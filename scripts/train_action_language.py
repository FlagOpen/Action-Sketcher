import pytorch_lightning as L
import torch
import os
import math

from argparse import ArgumentParser
from lerobot.configs.policies import PreTrainedConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Resize
from termcolor import cprint

from pi0.modeling import PI0Policy

from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import FusedAdam
from torchvision import transforms
import numpy as np

from robotics_dataset.dataset_complete_1_libero import RoboticsDataset, robotics_collate_fn


# Default paths (can be overridden by command line arguments)
JSON_PATH = ""
REASONING_JSON_PATH = ""
NORMALIZATION_PATH = ""
VALID_JSON_PATH = ""
VALID_REASONING_JSON_PATH = ""

# Default paths (will be overridden by command line arguments)
def to_device_dtype(d, device, dtype):
    for key, value in d.items():
        if isinstance(value, dict):
            to_device_dtype(value, device, dtype)
        elif isinstance(value, torch.Tensor):
            if key not in ["action_is_pad"]:
                d[key] = value.to(device=device, dtype=dtype)
            else:
                d[key] = value.to(device=device)
        else:
            pass
    return d



class LightningTrainingWrapper(L.LightningModule):
    def __init__(self, args, config=None, policy=None):
        super().__init__()
        self.policy = policy
        self.config = config
        self.args = args

    def configure_model(self):
        if self.policy is None:
            self.config.device = self.device
            self.policy = PI0Policy(self.config)

    def forward(self, batch):
        return self.policy(batch)[0]

    def training_step(self, batch, batch_idx):
        loss,loss_dict = self.policy(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("text_loss", loss_dict["text_loss"],prog_bar = True)
        self.log("action_loss", loss_dict["l2_loss"],prog_bar = True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss,loss_dict = self.policy(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_text_loss", loss_dict["text_loss"],prog_bar = True)
        self.log("val_action_loss", loss_dict["l2_loss"],prog_bar = True)
        return loss

    def configure_optimizers(self):
        optimizer = FusedAdam(
            self.policy.get_optim_params(), lr=self.args.lr, weight_decay=self.args.weight_decay, eps=1e-6, adam_w_mode=True
        )
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def train(args):
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255.0),
    ])
    
    # Create dataset with arguments from command line
    dataset_kwargs = {
        'json_path': args.json_path,
        'reasoning_json_path': args.reasoning_json_path,
        'normalization_path': args.normalization_path,
        'reasoning_only': args.reasoning_only,
        'use_vprompt': args.use_vprompt,
        'balance_sampling': args.balance_sampling,
        'visual_reasoning': args.visual_reasoning,
        'action_only': args.action_only,
        'add_augmentation': args.add_augmentation,
    }
    
    if hasattr(args, 'no_reference'):
        dataset_kwargs['no_reference'] = args.no_reference
    if hasattr(args, 'main_task_prompt'):
        dataset_kwargs['main_task_prompt'] = args.main_task_prompt
    
    dataset = RoboticsDataset(
        data_root=args.data_root,**dataset_kwargs)

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        persistent_workers=True,
        collate_fn=robotics_collate_fn,
        sampler = dataset.get_sampler() if hasattr(dataset, "get_sampler") else None,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    # Validation dataset
    val_dataset_kwargs = {
        'json_path': args.valid_json_path,
        'reasoning_json_path': args.valid_reasoning_json_path,
        'normalization_path': args.normalization_path,
        'reasoning_only': args.reasoning_only,
        'use_vprompt': args.use_vprompt,
        'balance_sampling': args.balance_sampling,
        'visual_reasoning': args.visual_reasoning,
        'action_only': args.action_only,
        'add_augmentation': args.add_augmentation,
    }
    
    if hasattr(args, 'no_reference'):
        val_dataset_kwargs['no_reference'] = args.no_reference
    if hasattr(args, 'main_task_prompt'):
        val_dataset_kwargs['main_task_prompt'] = args.main_task_prompt
    

    world_size = max(1, args.num_gpus * args.num_nodes)
    effective_global_batch = args.batch_size * args.acc_grad_batches * world_size
    scaled_lr = args.lr_base * math.sqrt(effective_global_batch / args.baseline_global_batch)

    steps_per_epoch_batches = math.ceil(len(dataloader))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch_batches / max(1, args.acc_grad_batches))
    total_optimizer_steps = optimizer_steps_per_epoch * args.max_epochs
    warmup_steps = max(1000, int(args.warmup_ratio * total_optimizer_steps))

    args.lr = args.lr_base
    args.warmup_steps = int(warmup_steps)

    cprint(
        f"[LR Tuning] world_size={world_size}, eff_global_batch={effective_global_batch}, lr={args.lr:.6g}, warmup_steps={args.warmup_steps}",
        "yellow",
    )

    callback = ModelCheckpoint(
        dirpath=os.path.join(args.ckpt_dir, args.exp_name),
        filename="{epoch}-{step}-{train_loss_epoch:.6f}",
        save_top_k=-1,
        every_n_epochs=args.every_n_epochs,
    )

    csv_logger = CSVLogger(save_dir = os.path.join(args.log_dir,'csv'), name = args.exp_name)
    tb_logger = TensorBoardLogger(save_dir = os.path.join(args.log_dir,'tb'), name = args.exp_name)

    loggers = [csv_logger, tb_logger]

    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "stage3_gather_16bit_weights_on_model_save": False,
        "bf16": {"enabled": True},
        "activation_checkpointing": {
            "partition_activations": True,
            "contiguous_memory_optimization": True,
            "cpu_checkpointing": False,
        },
    }
    deepspeed_strategy = DeepSpeedStrategy(config=ds_config)

    trainer = L.Trainer(
        accelerator="cuda",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy=deepspeed_strategy,
        max_epochs=args.max_epochs,
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        precision="bf16-mixed",
        accumulate_grad_batches=args.acc_grad_batches,
        callbacks=[callback],
        logger=loggers
    )

    with trainer.init_module():
        config = PreTrainedConfig.from_pretrained(args.pretrained_model_path)

        config.device = "cpu"
        config.freeze_vision_encoder = args.freeze_vision_encoder
        config.train_expert_only = args.train_expert_only
        config.train_state_proj = args.train_state_proj

    if args.resume_checkpoint:
        cprint(f"Resuming from lightning checkpoint: {args.resume_checkpoint}", "green")
        training_policy = LightningTrainingWrapper(args=args, config=config)
        trainer.fit(training_policy, dataloader, ckpt_path=args.resume_checkpoint)
    else:
        cprint(f"Resuming from pretrained checkpoint: {args.pretrained_model_path}", "green")
        policy = PI0Policy.from_pretrained(
            args.pretrained_model_path,
            config=config,
        ).train()

        print("=" * 80)
        print("Inspecting model parameters...")
        
        total_params = 0
        trainable_params = 0
        
        for name, param in policy.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"[✅ Trainable]: {name}")
            else:
                print(f"[🧊 Frozen]:    {name}")
        
        print("-" * 80)
        print("Summary:")
        print(f"  - Total Parameters:     {total_params / 1e6:.2f}M")
        print(f"  - Trainable Parameters: {trainable_params / 1e6:.2f}M")
        print(f"  - Frozen Parameters:    {(total_params - trainable_params) / 1e6:.2f}M")
        print(f"  - Percentage Trainable: {100 * trainable_params / total_params:.2f}%")
        print("=" * 80)
        
        training_policy = LightningTrainingWrapper(args=args, policy=policy, config=config)
        trainer.fit(training_policy, dataloader)


if __name__ == "__main__":
    parser = ArgumentParser(description="Train PI0 (tuned for DeepSpeed-ZeRO3)")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--ckpt_dir", type=str, default="./ckpts")
    parser.add_argument("--exp_name", type=str, default="114_stack_blocks_noaugment_stage3")
    parser.add_argument("--every_n_epochs", type=int, default=10)
    parser.add_argument("--max_epochs", type=int, default=400)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--acc_grad_batches", type=int, default=4)
    parser.add_argument("--lr_base", type=float, default=5e-5*math.sqrt(6))
    parser.add_argument("--baseline_global_batch", type=int, default=32)
    parser.add_argument("--warmup_ratio", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--freeze_vision_encoder", type=bool, default=True)
    parser.add_argument("--train_expert_only", type=bool, default=False)
    parser.add_argument("--train_state_proj", type=bool, default=False)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--pretrained_model_path", type=str, default="/share/project/xxq/pytorch_pi/ckpts/stack_blocks/stage2_epoch19")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="/share/project/caomingyu/stage1_train_data/high_quality_data_832k/llava_v1_5_lrv_mix832k_train.jsonl")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--freeze_action_modules", type=bool, default=True)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--lang_only", type=bool, default=True)
    
    # NEW DATASET ARGUMENTS
    parser.add_argument("--json_path", type=str, default=JSON_PATH)
    parser.add_argument("--reasoning_json_path", type=str, default=REASONING_JSON_PATH)
    parser.add_argument("--normalization_path", type=str, default=NORMALIZATION_PATH)
    parser.add_argument("--data_root", type=str, default="", help="Root directory for dataset files")
    parser.add_argument("--valid_json_path", type=str, default=VALID_JSON_PATH)
    parser.add_argument("--valid_reasoning_json_path", type=str, default=VALID_REASONING_JSON_PATH)
    parser.add_argument("--reasoning_only", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--balance_sampling", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--use_vprompt", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--visual_reasoning", type=lambda x: str(x).lower() == 'true', default=True)
    parser.add_argument("--action_only", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--add_augmentation", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--no_reference", type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument("--main_task_prompt", type=lambda x: str(x).lower() == 'true', default=False)
    
    args = parser.parse_args()
    train(args)