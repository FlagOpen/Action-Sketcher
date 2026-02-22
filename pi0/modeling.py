"""
本文件实现了基于 PaliGemma with Expert 的 π0 模型（PI0FlowMatching）以及其在
LeRobot 体系内的策略封装（PI0Policy）。

核心能力：
- 将多视角图像（含参考 ref/vp 图）与语言前缀拼接成 Transformer 前缀流；
- 将机器人状态与动作扩散变量拼接成 Transformer 后缀流；
- 支持语言自回归训练/推理与动作扩散训练/推理的联合；
- 通过 special tokens 区分前缀/后缀、动作/推理等不同阶段，并用 1D/2D mask 控制注意力流向；

重要数据约定：
- observation["image"][key] 形状为 (*b, 3, H, W) 且为 uint8 [0,255]，内部会归一化到 [-1,1]；
- observation["state"] 形状为 (*b, 1, S) 或 (*b, S) 的 float32，会 pad 到 config.max_state_dim；
- observation["action"] 形状为 (*b, T, A) 的 float32，会 pad 到 config.max_action_dim；
- observation["prompt"] 可为 List[str] 或 List[List[str]]，前者仅前缀，后者为 [前缀, 后缀]；

训练损失：
- 动作扩散损失（MSE）：对 u_t 与网络预测的 v_t 做逐元素 MSE；
- 语言 AR 损失（CE）：在语言区域上做 next-token 预测并按 mask 聚合；

关键掩码：
- lang_masks：语言 tokens 有效位；
- token_ar_mask：语言自回归边界（前缀为0，后缀为1）；
- token_loss_mask：语言损失的可学习区域；
- diffusion_loss_mask：若为 False，屏蔽该样本的扩散后缀（训练/注意力）。
"""

import einops
import numpy as np
import torch
import torch.nn.functional as F
from lerobot.common.policies.pi0.configuration_pi0_libero import PI0Config
from lerobot.common.policies.pretrained import PreTrainedPolicy
from torch import Tensor, nn
from transformers import AutoTokenizer

from .paligemma_with_expert import PaliGemmaWithExpertConfig, PaliGemmaWithExpertModel
from .utils import (
    create_sinusoidal_pos_embedding,
    make_att_2d_masks,
    resize_with_pad,
    sample_beta,
)

MODE = "vp"
# MODE = "ref"
# MODE = "noref"

if MODE == "vp":
    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "ref_0_rgb",
    )
elif MODE == "ref":
    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
        "ref_0_rgb",
    )
elif MODE == "noref":
    IMAGE_KEYS = (
        "base_0_rgb",
        "left_wrist_0_rgb",
        "right_wrist_0_rgb",
    )


class PI0Policy(PreTrainedPolicy):
    """π0 策略封装（供 LeRobot 使用）。

    作用：
    - 统一处理输入 batch（图像/状态/语言/动作），并调用底层 `PI0FlowMatching`；
    - 提供训练 `forward`，以及推理相关的若干接口；

    主要方法：
    - forward：训练时计算语言 CE 与动作扩散 MSE；
    - action_or_reasoning：根据首个生成 token 决定进入动作或语言分支；
    - generate_reasoning（deprecated）：仅语言生成的旧接口；
    - select_action（deprecated）：仅动作生成的旧接口；
    """

    config_class = PI0Config
    name = "torch_pi0"

    def __init__(
        self,
        config: PI0Config,
        # tokenizer_path: str = "google/paligemma-3b-pt-224",
        # tokenizer_path: str = "./ckpts/paligemma-3b-pt-224",
        test_mode = False,
        tokenizer_path: str = "./checkpoint",
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        self.config = config
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = PI0FlowMatching(config)
        self.test_mode = test_mode
        self.reset()

    def reset(self):
        return None

    def get_optim_params(self) -> dict:
        return self.parameters()
    
    @torch.no_grad
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        raise NotImplementedError("Currently not implemented for PI0")

    # Deprecated method, for reference only.
    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], noise: Tensor | None = None
    ):
        """[已弃用] 仅生成动作的示例接口。

        参数：
        - observation：
          - image：字典，包含 `base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`/`ref_0_rgb`，形状为 (*b, 3, H, W)；
          - state：(*b, 1, S) 或 (*b, S)；
          - 语言输入：传入 `prompt`（字符串列表），或显式传入 `lang_tokens` 与 `lang_masks`；

        说明：推荐使用 `action_or_reasoning` / `sample_actions` 等更完善的接口。
        """
        self.eval()

        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks, token_ar_mask, token_loss_mask, diffusion_loss_mask = self.prepare_language_fuse(observation)
        
        # 直接在lang_tokens中插入begin_of_action token
        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device
        
        # 为每个batch样本找到合适的插入位置（在padding之前）
        for i in range(batch_size):
            # 找到第一个padding位置（token为0且mask为False的位置）
            valid_length = lang_masks[i].sum().item()
            if valid_length < lang_tokens.shape[1]:
                # 在第一个padding位置插入begin_of_action token
                insert_pos = valid_length
                # 直接插入begin_of_action token（覆盖padding）
                lang_tokens[i, insert_pos] = self.config.begin_of_action_token
                # 更新mask
                lang_masks[i, insert_pos] = True
                # 更新ar_mask
                token_ar_mask[i, insert_pos] = 1
        
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, token_ar_mask, noise=noise
        )
        return actions


    # Deprecated method, for reference only.
    @torch.no_grad()
    def generate_reasoning(
        self,
        observation: dict[str, Tensor | list[str]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        return_text: bool = True,
    ):
        """[已弃用] 基于图像与文本前缀进行语言自回归生成。

        输入：
        - observation：包含 `image`/`state`/`prompt`；
          其中 `state` 仅用于尺寸/设备推断，不参与语言生成；
        - 生成参数：`max_new_tokens`/`do_sample`/`temperature`/`top_k`；

        输出：
        - 若 `return_text=True`，返回 {"texts": List[str], "tokens": LongTensor}；
        - 否则仅返回 tokens。
        推荐改用 `action_or_reasoning`，其在语言路径上功能更完整。
        """
        self.eval()

        images, img_masks = self.prepare_images(observation)

        prompts = observation.get("prompt", None)
        if prompts is None:
            raise ValueError("observation 中必须提供 'prompt': List[str]")

        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("'prompt' 必须是 List[str]，每个元素为一个样本的前缀文本。")

        device = images.device
        bsize = images.shape[0]
        if len(prompts) != bsize:
            raise ValueError("prompt 的 batch 大小与图像 batch 大小不一致。")

        # 为每个样本构造：<bos> prefix ... <end_of_prefix> <begin_of_reasoning>
        seqs: list[list[int]] = []
        max_len_limit = int(self.config.tokenizer_max_length)
        for i in range(bsize):
            prefix = prompts[i]
            toks = self.language_tokenizer.encode(prefix, add_special_tokens=True)
            # 移除尾部 eos
            if (
                len(toks) > 0
                and self.language_tokenizer.eos_token_id is not None
                and toks[-1] == self.language_tokenizer.eos_token_id
            ):
                toks = toks[:-1]

            toks.append(int(self.config.end_of_prefix_token))
            # 在推理时显式加入 <BEGIN_OF_REASONING>
            # if hasattr(self.config, "begin_of_reasoning_token"):
            #     toks.append(int(self.config.begin_of_reasoning_token))

            # 截断到上限
            toks = toks[:max_len_limit]
            seqs.append(toks)

        init_lengths = [len(s) for s in seqs]
        max_init_len = max(init_lengths)
        lang_tokens = torch.zeros((bsize, max_init_len), dtype=torch.long, device=device)
        lang_masks = torch.zeros((bsize, max_init_len), dtype=torch.bool, device=device)
        for i in range(bsize):
            li = init_lengths[i]
            lang_tokens[i, :li] = torch.tensor(seqs[i], dtype=torch.long, device=device)
            lang_masks[i, :li] = True

        # 自回归生成
        gen_ids = self.model.generate_reasoning(
            images=images,
            img_masks=img_masks,
            lang_tokens=lang_tokens,
            lang_masks=lang_masks,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

        if not return_text:
            return gen_ids

        # 解码文本（去除填充 eos 之后的部分）
        eos_id = int(self.config.paligemma_eos_token)
        texts: list[str] = []
        for i in range(gen_ids.shape[0]):
            row = gen_ids[i].tolist()
            try:
                cut = row.index(eos_id)
                row = row[:cut]
            except ValueError:
                pass
            text = self.language_tokenizer.decode(row, skip_special_tokens=True)
            texts.append(text)

        return {"texts": texts, "tokens": gen_ids}

    @torch.no_grad()
    def action_or_reasoning(
        self,
        observation: dict[str, Tensor | list[str]],
        max_new_tokens: int = 512,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
        return_text: bool = True,
        return_logits: bool = False
    ):
        """统一的推理入口：先用前三张图+前缀，预测首个 token 决策路径。

        流程：
        - 使用 [前三张图] + [语言前缀] 计算下一个 token 的分布；
        - 若 token==<BEGIN_OF_ACTION>，把 ref 图插到语言之后，进入动作生成分支；
        - 否则，保持语言路径进行自回归生成；

        返回：
        - 动作分支：{"boa_mask": Bool[b], "actions": (*b', T, A), "is_action": True}
        - 语言分支：{"texts": List[str], "tokens": LongTensor, "is_action": False}
        """
        self.eval()

        # --- 步骤 1：准备图像、状态和语言前缀 ---
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)

        prompts = observation.get("prompt", None)


        if prompts is None:
            raise ValueError("observation 中必须提供 'prompt': List[str]")
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError("'prompt' 必须是 List[str]，每个元素为一个样本的前缀文本。")

        device = images.device
        dtype = images.dtype
        bsize = images.shape[0]

        # 将字符串列表格式的 prompt 编码为 token IDs，并在末尾添加 <EOP> (end of prefix)
        seqs: list[list[int]] = []
        max_len_limit = int(self.config.tokenizer_max_length)
        for i in range(bsize):
            prefix = prompts[i]
            toks = self.language_tokenizer.encode(prefix, add_special_tokens=True)
            # 移除 tokenizer 可能自动添加的 eos token
            if (
                len(toks) > 0
                and self.language_tokenizer.eos_token_id is not None
                and toks[-1] == self.language_tokenizer.eos_token_id
            ):
                toks = toks[:-1]
            toks.append(int(self.config.end_of_prefix_token))
            toks = toks[:max_len_limit]
            seqs.append(toks)

        # 将 tokens 列表转换为 padded tensor
        init_lengths = [len(s) for s in seqs]
        max_init_len = max(init_lengths)
        lang_tokens = torch.zeros((bsize, max_init_len), dtype=torch.long, device=device)
        lang_masks = torch.zeros((bsize, max_init_len), dtype=torch.bool, device=device)
        for i in range(bsize):
            li = init_lengths[i]
            lang_tokens[i, :li] = torch.tensor(seqs[i], dtype=torch.long, device=device)
            lang_masks[i, :li] = True

        # --- 步骤 2：生成决策 token (第一个 token) ---
        # 这个阶段只使用前置图像（不含 reference image），以模拟 "先观察再决策"
        boa_token_id = int(self.config.begin_of_action_token)
        eos_token_id = int(self.config.paligemma_eos_token)

        # a. 图像嵌入并拆分前置图像与参考图像
        num_images = images.shape[1]
        images_reshaped = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb_all = self.model.paligemma_with_expert.embed_image(images_reshaped)
        num_patch = img_emb_all.shape[1]
        img_emb_all = einops.rearrange(img_emb_all, "(b n) l d -> b n l d", b=bsize)
        img_emb_all = img_emb_all.to(dtype=dtype) * (img_emb_all.shape[-1] ** 0.5)

        img_masks_patch = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
      
        # 决策时只用前 n-1 张图
        front_img_emb = einops.rearrange(img_emb_all[:, : num_images - 1, :, :], "b n l d -> b (n l) d")
        front_img_masks = einops.rearrange(img_masks_patch[:, : num_images - 1, :], "b n l -> b (n l)")

        # b. 语言嵌入
        lang_emb = self.model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb = lang_emb.to(dtype=dtype) * np.sqrt(lang_emb.shape[-1])

        # c. 构造送入 Transformer 的前缀序列：[前置图像] + [语言前缀]
        prefix_embs = torch.cat([front_img_emb, lang_emb], dim=1)
        prefix_pad_masks = torch.cat([front_img_masks, lang_masks], dim=1)
        prefix_ar_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.int32, device=device)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # d. Transformer 前向传播，得到 hidden states
        (prefix_out, _), past_key_values = self.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=False,
        )

        # e. 提取语言部分的最后一个 token 的 hidden state，并计算 logits
        front_img_len = front_img_emb.shape[1]
        num_lang_embs = lang_tokens.shape[1]
        lang_hidden = prefix_out[:, front_img_len : front_img_len + num_lang_embs, :]
        lm_head = self.model.paligemma_with_expert.paligemma.language_model.lm_head
        logits = lm_head(lang_hidden).to(dtype=torch.float32)

        # 找到每个样本语言部分的真实结尾位置
        prefix_lang_lens = lang_masks.to(dtype=torch.int64).sum(dim=1)
        last_indices = (prefix_lang_lens - 1).clamp_min(0)
        gather_indices = last_indices.view(bsize, 1, 1).expand(bsize, 1, logits.size(-1))
        last_logits = torch.gather(logits, dim=1, index=gather_indices).squeeze(1)

        # f. 强制约束：第一个生成的 token 只能是 <BOA> 或 <BOR>
        allowed_ids = [int(self.config.begin_of_action_token)]
        if hasattr(self.config, "begin_of_reasoning_token"):
            allowed_ids.append(int(self.config.begin_of_reasoning_token))
        allowed_ids = sorted(set(allowed_ids))
        allowed_tensor = torch.tensor(allowed_ids, dtype=torch.long, device=device)
        decision_logits = last_logits.gather(1, allowed_tensor.unsqueeze(0).expand(bsize, -1))

        # 将 logits 中所有不在允许列表里的 token 的概率设为 -inf
        masked_logits = torch.full_like(last_logits, fill_value=-float("inf"))
        gathered_vals = last_logits.gather(1, allowed_tensor.unsqueeze(0).expand(bsize, -1))
        masked_logits.scatter_(1, allowed_tensor.unsqueeze(0).expand(bsize, -1), gathered_vals)
        last_logits = masked_logits
        print(last_logits)
        # g. 从约束后的 logits 中采样下一个 token
        def sample_next_token(curr_logits: torch.Tensor) -> torch.Tensor:   
            if do_sample:
                cl = curr_logits
                if temperature is not None and temperature > 0:
                    cl = cl / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    values, _ = torch.topk(cl, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    cl = torch.where(cl < min_values, torch.full_like(cl, -float("inf")), cl)
                probs = F.softmax(cl, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                return torch.argmax(curr_logits, dim=-1)

        next_tokens = sample_next_token(last_logits)
        # 判断每个样本是否选择了动作分支
        boa_mask = next_tokens.eq(boa_token_id)

        return_dict = {}
        if return_logits:
            return_dict["decision_logits"] = decision_logits
            return_dict["allowed_token_ids"] = allowed_tensor
        # --- 步骤 3：根据决策 token，进入不同分支 ---
        if torch.any(boa_mask):
            # --- 动作生成分支 ---
            # a. 将决策出的 <BOA> token 追加到原始语言序列后面
            L_old = lang_tokens.shape[1]
            L_new = L_old + 1
            lang_tokens_aug = torch.zeros((bsize, L_new), dtype=torch.long, device=device)
            lang_masks_aug = torch.zeros((bsize, L_new), dtype=torch.bool, device=device)
            token_ar_mask_aug = torch.zeros((bsize, L_new), dtype=torch.int32, device=device)
            lang_tokens_aug[:, :L_old] = lang_tokens
            lang_masks_aug[:, :L_old] = lang_masks
            lang_tokens_aug[:, L_old] = next_tokens
            lang_masks_aug[:, L_old] = True
            token_ar_mask_aug[:, L_old] = (next_tokens == boa_token_id).to(torch.int32)

            # b. 只选择那些需要生成动作的样本进行处理
            boa_indices = torch.nonzero(boa_mask, as_tuple=False).squeeze(-1)
            images_boa = images.index_select(0, boa_indices)
            img_masks_boa = img_masks.index_select(0, boa_indices)
            state_boa = state.index_select(0, boa_indices)
            lang_tokens_boa = lang_tokens_aug.index_select(0, boa_indices)
            lang_masks_boa = lang_masks_aug.index_select(0, boa_indices)
            token_ar_mask_boa = token_ar_mask_aug.index_select(0, boa_indices)

            # c. 调用底层的动作采样函数（此时会用到包括 ref image 在内的所有图像）
            actions_boa = self.model.sample_actions(
                images_boa,
                img_masks_boa,
                lang_tokens_boa,
                lang_masks_boa,
                state_boa,
                token_ar_mask_boa,
            )
            is_action = True

            return_dict.update({
                "boa_mask": boa_mask,
                "actions": actions_boa,
                "is_action": True
            })
            #return {"boa_mask": boa_mask, "actions": actions_boa,"is_action": is_action}
            return return_dict

        # --- 语言生成分支 ---
        # 调用底层的语言生成函数，继续自回归解码
        gen_ids = self.model.generate_reasoning(
            images=images, # 此时会用到所有图像
            img_masks=img_masks,
            lang_tokens=lang_tokens, # 语言部分只送入前缀
            lang_masks=lang_masks,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )

        return_dict.update({
            "tokens": gen_ids,
            "is_action": False
        })

        if not return_text:
            return return_dict

        # 将生成的 token IDs 解码为文本
        texts: list[str] = []
        for i in range(gen_ids.shape[0]):
            row = gen_ids[i].tolist()
            try:
                cut = row.index(eos_token_id)
                row = row[:cut]
            except ValueError:
                pass
            text = self.language_tokenizer.decode(row, skip_special_tokens=True)
            texts.append(text)
        is_action = False
        return_dict["texts"] = texts
        return return_dict

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """训练前向：计算并返回总损失与日志项。

        输入 batch 需要包含：
        - image：多视角图像（字典），内部会被转换为 [-1,1] 并 resize+pad；
        - state：机器人状态（会 pad 到 max_state_dim）；
        - prompt 或 (lang_tokens, lang_masks)：语言输入；
        - action：动作序列（会 pad 到 max_action_dim）；

        返回：
        - (loss, loss_dict)
          - loss 为组合损失：diffusion_loss_coeff * diffusion + text_loss_coeff * text
          - loss_dict 记录分项：l2_loss、text_loss、以及可选的中间可视化项
        """
        # --- 步骤 1：准备所有模态的输入 ---
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        # 从 batch["prompt"] 解析出 tokens 和各种 masks
        lang_tokens, lang_masks, token_ar_mask, token_loss_mask, diffusion_loss_mask = self.prepare_language_fuse(batch)
        actions, action_dim = self.prepare_action(batch)
        noise = batch.get("noise", None)
        time = batch.get("time", None)

        loss_dict = {}
        # --- 步骤 2：调用底层模型 PI0FlowMatching 的 forward ---
        action_losses, text_loss_per_batch = self.model.forward(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask, diffusion_loss_mask, state, actions, noise, time
        )

        # --- 步骤 3：后处理与损失计算 ---
        # 如果数据中有 episode 边界的 padding 标记，则将 padding 部分的损失置零
        actions_is_pad = batch.get("action_is_pad", None)
        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_losses = action_losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = action_losses.clone()

        # 移除因维度对齐而添加的 padding 动作维度的损失
        action_losses = action_losses[:, :, :action_dim]
        loss_dict["losses"] = action_losses.clone()

        # 计算用于反向传播的总损失
        diffusion_loss = action_losses.mean()
        text_loss = text_loss_per_batch.mean()
        # 总损失 = 动作扩散损失 + 语言建模损失
        loss = self.config.diffusion_loss_coeff * diffusion_loss + self.config.text_loss_coeff * text_loss
        
        # 记录用于日志的标量损失值
        loss_dict["l2_loss"] = diffusion_loss.item()
        loss_dict["text_loss"] = text_loss.item()

        return loss, loss_dict

    def prepare_images(self, observation: dict[str, Tensor]):
        """图像预处理：归一化到 [-1,1]，按配置尺寸 pad，并堆叠为 (*b, n, c, h, w)。

        返回：
        - images：(*b, n, 3, H, W)
        - img_masks：(*b, n) 布尔，表示该视角是否存在
        """
        dtype = observation["state"].dtype
        bsize = observation["state"].shape[0]
        images, img_masks = [], []
        # 确定哪些预期的图像 key 存在于当前 observation 中
        present_img_keys = [key for key in IMAGE_KEYS if key in observation["image"]]
        missing_img_keys = [key for key in IMAGE_KEYS if key not in present_img_keys]

        for key in present_img_keys:
            # 读取图像
            img = observation["image"][key]
            # 归一化：从 uint8 [0, 255] 转为 float32 [-1.0, 1.0]
            img = img.to(dtype) / 127.5 - 1.0
            # 缩放和填充：将不同尺寸的图像处理为统一尺寸
            img = resize_with_pad(
                img, *self.config.resize_imgs_with_padding, pad_value=-1.0
            )
            images.append(img)
            # 标记该图像存在
            img_masks.append(torch.ones((bsize,), dtype=torch.bool, device=img.device))

        for key in missing_img_keys:
            # 对于缺失的图像，用 -1.0 填充的占位符代替
            img = torch.full_like(images[0], fill_value=-1.0) # 使用已处理图像的尺寸
            images.append(img)
            # 标记该图像缺失
            img_masks.append(torch.zeros((bsize,), dtype=torch.bool, device=img.device))

        # 将图像列表堆叠成一个 tensor
        images = torch.stack(images, dim=1)  # (*b, n, c, h, w)
        img_masks = torch.stack(img_masks, dim=1)  # (*b, n)

        return images, img_masks

    def prepare_state(self, observation: dict[str, Tensor]):
        """对状态向量右侧进行零填充，长度齐到 config.max_state_dim。"""
        state = observation["state"]
        # F.pad 的 (0, pad_len) 表示在最后一个维度上，左侧不填充，右侧填充 pad_len
        state = F.pad(state, (0, self.config.max_state_dim - state.shape[-1]))
        return state

    def prepare_action(self, observation: dict[str, Tensor]):
        """对动作向量维度进行右侧零填充，长度齐到 config.max_action_dim。

        返回：
        - action：(*b, T, max_action_dim)
        - action_dim：原始动作维度（未填充前）
        """
        action = observation["action"]
        action_dim = action.shape[-1]
        # 在最后一个维度上进行填充
        action = F.pad(action, (0, self.config.max_action_dim - action_dim))
        return action, action_dim


    def prepare_language_fuse(self, observation: dict[str, Tensor]):
            """准备并融合语言输入（前缀/后缀），输出 tokens 与各类 mask。

            两种输入形式：
            - 直接给 prompt（List[str] 或 List[List[str]]）；
            - 或直接给 lang_tokens 与 lang_masks；

            约定：
            - prompt[0] 为语言前缀；
            - prompt[1] 若以 <BEGIN_OF_REASONING> 开头，则进入语言后缀；若以 <BEGIN_OF_ACTION> 开头，则进入动作后缀；
            - 会生成：
              - lang_tokens：拼接的 token 序列
              - lang_masks：有效位
              - token_ar_mask：自回归（前缀0，后缀1）
              - token_loss_mask：计算语言 CE 的可学习区域
              - diffusion_loss_mask：样本级别是否计算扩散（动作）损失
            """
            # --- 步骤 1：获取输入 ---
            lang_tokens = observation.get("lang_tokens", None)
            lang_masks = observation.get("lang_masks", None)
            prompts = observation.get("prompt", None)
            device = observation["state"].device

            # 检查输入合法性：必须提供 prompt 或 (lang_tokens, lang_masks)
            if prompts is None and (lang_tokens is None or lang_masks is None):
                raise ValueError(
                    "Either 'prompt' or ('lang_tokens', 'lang_masks') must be provided in the observation."
                )

            # 预处理：清理 prompt 中可能存在的 <image> 占位符
            if prompts is not None:
                cleaned_prompts = []
                for item in prompts:
                    if isinstance(item, str):
                        cleaned_prompts.append(item.replace("<image>", ""))
                    elif isinstance(item, (list, tuple)):
                        cleaned_prompts.append([(
                            (s.replace("<image>", "") if isinstance(s, str) else s)
                        ) for s in item])
                    else:
                        cleaned_prompts.append(item)
                prompts = cleaned_prompts

            batch_size = len(prompts)

            # --- 步骤 2：初始化用于存储 batch 数据的列表 ---
            all_tokens: list[list[int]] = []
            all_token_masks: list[list[bool]] = []
            all_ar_masks: list[list[int]] = []
            all_text_loss_masks: list[list[bool]] = []
            all_diffusion_loss_masks: list[bool] = []

            max_len = self.config.tokenizer_max_length
            
            # --- 步骤 3：逐样本处理 ---
            for i in range(batch_size):
                raw_item = prompts[i]
                
                # a. 解析 prompt 格式
                # prompt 支持两种格式：
                # 1. str: 仅包含前缀文本，用于纯动作推理。
                # 2. List[str]: [前缀, 后缀]，用于联合训练或推理。
                if isinstance(raw_item, str):
                    prompt_list = [raw_item]
                elif isinstance(raw_item, list) and all(isinstance(p, str) for p in raw_item):
                    prompt_list = raw_item
                else:
                    raise ValueError(
                        "Each prompt element must be either a string (prefix-only) or a list of strings (prefix + suffix)."
                    )

                # b. 处理前缀 (prefix)
                #    i. 对前缀文本进行编码，并添加 BOS (begin of sentence) token。
                prefix = prompt_list[0]
                prefix_tokens = self.language_tokenizer.encode(prefix, add_special_tokens=True)

                #    ii. 若 tokenizer 自动在末尾添加了 EOS (end of sentence)，则移除。
                if (
                    len(prefix_tokens) > 0
                    and self.language_tokenizer.eos_token_id is not None
                    and prefix_tokens[-1] == self.language_tokenizer.eos_token_id
                ):
                    prefix_tokens = prefix_tokens[:-1]

                #    iii. 在前缀末尾追加 <EOP> (end of prefix) 特殊 token 作为分界。
                prefix_tokens.append(self.config.end_of_prefix_token)

                # c. 处理后缀 (suffix)
                #    根据后缀的起始 special token，判断是推理（reasoning）还是动作（action）分支。
                if len(prompt_list) > 1 and prompt_list[1].startswith("<BEGIN_OF_REASONING>"):
                    # 语言分支 (reasoning):
                    # i. 提取 <BOR> 之后的内容，并编码为 tokens。
                    suffix_text = prompt_list[1].removeprefix("<BEGIN_OF_REASONING>")
                    suffix_tokens = [self.config.begin_of_reasoning_token]
                    suffix_tokens += self.language_tokenizer.encode(
                        suffix_text, add_special_tokens=False
                    )
                    # ii. 在结尾添加 EOS token，表示生成结束。
                    suffix_tokens.append(self.config.paligemma_eos_token)
                    # iii. diffusion_loss_mask 设为 False，表示该样本不计算动作损失。
                    diff_mask_val = False
                elif len(prompt_list) > 1 and prompt_list[1].startswith("<BEGIN_OF_ACTION>"):
                    # 动作分支 (action):
                    # i. 后缀仅包含 <BOA> (begin of action) token。
                    suffix_tokens = [self.config.begin_of_action_token]
                    # ii. diffusion_loss_mask 设为 True，表示该样本需要计算动作损失。
                    diff_mask_val = True
                else:
                    # 如果有后缀但不是以上两种情况，则数据格式错误
                    raise AssertionError("The suffix format is incorrect. It should start with <BEGIN_OF_REASONING> or <BEGIN_OF_ACTION>.")
                    

                # d. 拼接前缀与后缀 tokens
                tokens = prefix_tokens + suffix_tokens

                # e. 生成各类掩码
                #    - token_mask: 有效 token 的位置 (True)。
                token_mask = [True] * len(tokens)
                #    - ar_mask: 自回归掩码，前缀为 0（双向可见），后缀为 1（单向可见）。
                ar_mask = [0] * len(prefix_tokens) + [1] * len(suffix_tokens)
                #    - text_loss_mask: 仅在后缀（推理内容）上计算语言损失。
                text_loss_mask = [False] * len(prefix_tokens) + [True] * len(suffix_tokens)

                # f. 对齐长度：过长则截断，过短则用 0 填充
                if len(tokens) < max_len:
                    pad_len = max_len - len(tokens)
                    tokens.extend([0] * pad_len)
                    token_mask.extend([False] * pad_len)
                    ar_mask.extend([0] * pad_len)
                    text_loss_mask.extend([False] * pad_len)
                else:
                    tokens = tokens[:max_len]
                    token_mask = token_mask[:max_len]
                    ar_mask = ar_mask[:max_len]
                    text_loss_mask = text_loss_mask[:max_len]

                # g. 将处理好的单个样本结果存入 batch 列表
                all_tokens.append(tokens)
                all_token_masks.append(token_mask)
                all_ar_masks.append(ar_mask)
                all_text_loss_masks.append(text_loss_mask)
                all_diffusion_loss_masks.append(diff_mask_val)

            # --- 步骤 4：将列表批量转换为 Tensor ---
            lang_tokens = torch.tensor(all_tokens, dtype=torch.long, device=device)
            lang_masks = torch.tensor(all_token_masks, dtype=torch.bool, device=device)
            token_ar_mask = torch.tensor(all_ar_masks, dtype=torch.int32, device=device)
            token_loss_mask = torch.tensor(all_text_loss_masks, dtype=torch.bool, device=device)
            diffusion_loss_mask = torch.tensor(all_diffusion_loss_masks, dtype=torch.bool, device=device)

            return (
                lang_tokens,
                lang_masks,
                token_ar_mask,
                token_loss_mask,
                diffusion_loss_mask,
            )

class PI0FlowMatching(nn.Module):
    """π0 主体模型：语言-视觉前缀 + 状态-动作后缀 的联合 Transformer。

    参考：
    - 论文: https://www.physicalintelligence.company/download/pi0.pdf
    - Jax 代码: https://github.com/Physical-Intelligence/openpi

    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    ┌──────────────────────────────┐
    │               actions        │
    │               ▲              │
    │              ┌┴─────┐        │
    │  kv cache    │Gemma │        │
    │  ┌──────────►│Expert│        │
    │  │           │      │        │
    │ ┌┴────────┐  │x 10  │        │
    │ │         │  └▲──▲──┘        │
    │ │PaliGemma│   │  │           │
    │ │         │   │  robot state │
    │ │         │   noise          │
    │ └▲──▲─────┘                  │
    │  │  │                        │
    │  │  image(s)                 │
    │  language tokens             │
    └──────────────────────────────┘
     结构要点：
    - 前缀流：多图像 + 语言 tokens；
    - 后缀流：状态 + 动作扩散轨迹；
    - 通过 1D/2D 掩码控制前后缀彼此的可见性；
    - 语言使用 CE 训练，动作使用 Flow Matching（MSE）训练。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # paligemma with action expert
        print(f"freeze vision_encoder {self.config.freeze_vision_encoder}")
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_with_export_config
        )

        # projection layers
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(
            self.config.max_action_dim, self.config.proj_width
        )
        self.action_out_proj = nn.Linear(
            self.config.proj_width, self.config.max_action_dim
        )

        self.action_time_mlp_in = nn.Linear(
            self.config.proj_width * 2, self.config.proj_width
        )
        self.action_time_mlp_out = nn.Linear(
            self.config.proj_width, self.config.proj_width
        )

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    
    def embed_img_lang(
        self, images, img_masks, lang_tokens, lang_masks,token_ar_mask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """图像+语言前缀嵌入。

        输入：
        - images：(*b, n, 3, H, W) 已归一化至 [-1,1]
        - img_masks：(*b, n) 图像存在标记
        - lang_tokens/lang_masks：语言 tokens 与有效位
        - token_ar_mask：语言自回归掩码（前缀0，后缀1）

        返回：
        - embs：拼接后的前缀嵌入；
        - pad_masks：对应的 1D 有效位；
        - ar_masks：对应的 1D 自回归边界；
        - lang_start_index：语言在前缀嵌入中的起始位置（便于抽取语言 hidden）
        """
        bsize = images.shape[0]
        device = images.device
        dtype = images.dtype

        # 1. 图像嵌入
        # a. 将 batch 与 num_images 维度合并，送入图像编码器。
        num_images = images.shape[1]
        images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb_all = self.paligemma_with_expert.embed_image(images)
        # b. 恢复 batch 维度，并将 patch 维度与 num_images 维度合并。
        num_patch = img_emb_all.shape[1]
        img_emb_all = einops.rearrange(img_emb_all, "(b n) l d -> b (n l) d", b=bsize)
        # c. 对嵌入进行尺度缩放，这是 Transformer 常见做法。
        img_emb_all = img_emb_all.to(dtype=dtype) * (img_emb_all.shape[-1] ** 0.5)

        # 2. 将图像掩码扩展到 patch 级别。
        img_masks_patch = einops.repeat(img_masks, "b n -> b n l", l=num_patch)

        # 3. 分离 "前置图像" (front_img) 与 "参考图像" (ref_img)。
        #    关键设计：推理时，模型先基于前置图像和语言指令，预测出动作意图或初步推理。
        #    然后，可以基于这个初步结果对参考图像进行处理（如画上箭头作为 visual prompt），
        #    再将处理后的参考图送入模型进行精确的动作生成。
        #    因此，在构建初始序列时，参考图（通常是第4张）被刻意放在语言 token 之后，
        #    以模拟这种 "先看场景、再思考、后用参考图执行" 的流程。
        if num_images >= 4:
            # a. 恢复 num_images 维度，以便分离。
            img_emb_4d = einops.rearrange(img_emb_all, "b (n l) d -> b n l d", n=num_images)
            # b. 前 n-1 张为 front_img，最后 1 张为 ref_img。
            front_img_emb = einops.rearrange(img_emb_4d[:, : num_images - 1, :, :], "b n l d -> b (n l) d")
            ref_img_emb = einops.rearrange(img_emb_4d[:, num_images - 1 : num_images, :, :], "b n l d -> b (n l) d")

            # c. 对应地分离掩码。
            front_img_masks = einops.rearrange(img_masks_patch[:, : num_images - 1, :], "b n l -> b (n l)")
            ref_img_masks = einops.rearrange(img_masks_patch[:, num_images - 1 : num_images, :], "b n l -> b (n l)")

            # d. 为图像序列创建自回归掩码。
            #    front_img 内部双向可见（mask=0），ref_img 的首个 token 设为 1，表示新的自回归段落开始。
            img_ar_mask_front = torch.zeros_like(front_img_masks, dtype=torch.int32, device=device)
            img_ar_mask_ref = torch.zeros_like(ref_img_masks, dtype=torch.int32, device=device)
            if img_ar_mask_ref.shape[1] > 0:
                img_ar_mask_ref[:, 0] = 1
        else:
            # 若图像不足4张，则全部视为 front_img，ref_img 为空。
            front_img_emb = img_emb_all
            ref_img_emb = torch.zeros((bsize, 0, img_emb_all.shape[-1]), dtype=img_emb_all.dtype, device=device)
            front_img_masks = einops.rearrange(img_masks_patch, "b n l -> b (n l)")
            ref_img_masks = torch.zeros((bsize, 0), dtype=torch.bool, device=device)
            img_ar_mask_front = torch.zeros_like(front_img_masks, dtype=torch.int32, device=device)
            img_ar_mask_ref = torch.zeros((bsize, 0), dtype=torch.int32, device=device)

        # 4. 语言嵌入。
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]
        lang_emb = lang_emb.to(dtype=dtype) * np.sqrt(lang_emb.shape[-1])
        lang_ar_mask = token_ar_mask.to(device=device, dtype=torch.int32)

        # 5. 组装最终的前缀序列：[前置图像] + [语言] + [参考图像]
        embs = torch.cat([front_img_emb, lang_emb, ref_img_emb], dim=1)
        pad_masks = torch.cat([front_img_masks, lang_masks, ref_img_masks], dim=1)
        ar_masks = torch.cat([img_ar_mask_front, lang_ar_mask, img_ar_mask_ref], dim=1)
        
        # 6. 记录语言 token 在拼接后序列中的起始位置，便于后续提取语言部分的 hidden states。
        lang_start_index = front_img_emb.shape[1]
        return embs, pad_masks, ar_masks, lang_start_index

    def embed_state_action(self, state, noisy_actions, timestep):
        """状态+动作后缀嵌入。

        - state：(*b, S)
        - noisy_actions：(*b, T, A)
        - timestep：(*b,) in [0,1]
        返回后缀 token 的嵌入、pad mask 与注意力 mask（控制前缀不可看后缀、state 不看 action）。
        """
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        # 1. 将状态向量投影到 Transformer 的隐藏维度。
        state_emb = self.state_proj(state)

        # 2. 将 [0,1] 的 diffusion timestep 编码为正弦位置编码。
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.config.proj_width,
            min_period=4e-3,
            max_period=4.0,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        # 3. 将带噪动作投影到隐藏维度。
        action_emb = self.action_in_proj(noisy_actions)
        # 4. 将时间编码扩展到与动作序列等长，并与动作嵌入拼接。
        time_emb = einops.repeat(time_emb, "b d -> b n d", n=action_emb.shape[1])
        action_time_emb = torch.cat([action_emb, time_emb], dim=-1)

        # 5. 使用一个小型 MLP 进一步融合动作与时间信息。
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        action_time_dim = action_time_emb.shape[1]

        # 6. 拼接状态嵌入与（动作-时间）嵌入，构成完整的后缀序列。
        embs = torch.cat([state_emb, action_time_emb], dim=1)
        # 后缀序列所有 token 都是有效的，因此 pad_masks 全为 True。
        pad_masks = torch.ones(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )

        # 7. 创建后缀内部的注意力掩码（1D）。
        #    - `att_masks` 为 0 表示双向可见，为 1 表示单向（自回归）。
        #    - state token (第0位) 和 action_time_emb 的首位设为 1，作为自回归段的起点。
        #    - 其余 action token 内部双向可见。
        #    - 这个 1D mask 后续会与前缀的 mask 一起送入 `make_att_2d_masks` 生成最终的 2D 注意力矩阵。
        att_masks = torch.zeros(
            (bsize, action_time_dim + 1), device=device, dtype=torch.bool
        )
        att_masks[:, :2] = True

        return embs, pad_masks, att_masks
    
    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        token_ar_mask,
        token_loss_mask,
        diffusion_loss_mask,
        state,
        actions,
        noise=None,
        time=None,
    ) -> tuple[Tensor, torch.Tensor]:
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device

        # --- 1. 准备 Flow Matching 训练样本 ---
        # a. 若未提供，则采样高斯噪声。
        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        # b. 若未提供，则采样 diffusion 时间步 t。
        if time is None:
            time = self.sample_time(bsize, device).to(dtype)

        # c. 构造线性插值的带噪动作 x_t，以及 flow matching 的目标速度场 u_t。
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # --- 2. 嵌入前缀与后缀序列 ---
        # a. 嵌入图像和语言，得到前缀流的嵌入、填充掩码、自回归掩码，以及语言部分的起始索引。
        prefix_embs, prefix_pad_masks, prefix_att_masks, lang_start_index = self.embed_img_lang(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask)
        # b. 嵌入状态和带噪动作，得到后缀流的嵌入和掩码。
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_state_action(
            state, x_t, time)
        
        # c. 对于仅做语言推理的样本（diffusion_loss_mask=False），屏蔽其整个后缀流，
        #    使其在注意力计算中不可见，也等效于不参与计算。
        if diffusion_loss_mask is not None:
            diff_mask_bool = diffusion_loss_mask.to(dtype=torch.bool).view(-1, 1)
            suffix_pad_masks = suffix_pad_masks & diff_mask_bool
            suffix_att_masks = suffix_att_masks & diff_mask_bool

        suffix_att_masks = suffix_att_masks.to(dtype=torch.int32)

        # --- 3. 拼接前后缀，送入 Transformer ---
        # a. 拼接前缀和后缀的填充掩码与自回归掩码。
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # b. 基于 1D 的 pad_masks 和 att_masks 生成最终的 2D 注意力矩阵。
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # c. 生成位置编码 ID。
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # d. 执行 Transformer 前向传播。
        (prefix_out, suffix_out), _past = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        # --- 4. 计算动作扩散损失 ---
        # a. 从后缀输出中提取动作部分，并通过输出投影层得到预测的速度场 v_t。
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        v_t = self.action_out_proj(suffix_out)
        # b. 计算预测速度 v_t 与目标速度 u_t 之间的 MSE 损失。
        action_losses = F.mse_loss(u_t, v_t, reduction="none")

        # c. 对于推理样本，通过乘以0来屏蔽其扩散损失。
        if diffusion_loss_mask is not None:
            action_losses =  action_losses * diffusion_loss_mask.view(bsize, 1, 1).to(action_losses.dtype)

        # --- 5. 计算语言自回归损失 ---
        # a. 从前缀输出中，根据之前记录的 lang_start_index，提取语言 token 对应的 hidden states。
        lang_hidden = prefix_out[:, lang_start_index : lang_start_index + lang_tokens.shape[1], :]

        # b. 将 hidden states 通过 lm_head 转换为词表 logits。
        lm_head = self.paligemma_with_expert.paligemma.language_model.lm_head
        logits = lm_head(lang_hidden)
        logits = logits.to(dtype=torch.float32)

        # c. 经典的 next-token 预测设置：logits 右移一位，labels 左移一位。
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lang_tokens[:, 1:].contiguous()
        # d. 最终的损失计算目标区域，是同时满足 lang_masks 和 token_loss_mask 的部分。
        target_mask = (lang_masks[:, 1:] & token_loss_mask[:, 1:]).to(dtype=torch.bool)

        # e. 计算每个 token 的交叉熵损失。
        vocab_size = shift_logits.size(-1)
        ce_per_token = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction="none",
        ).view(bsize, -1)

        # f. 使用 target_mask 屏蔽无需计算损失的 token，并按样本求平均得到最终的语言损失。
        masked_ce = ce_per_token * target_mask.to(dtype=ce_per_token.dtype)
        denom = target_mask.sum(dim=1).clamp_min(1).to(dtype=ce_per_token.dtype)
        text_loss_per_batch = masked_ce.sum(dim=1) / denom

        return action_losses, text_loss_per_batch

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        token_ar_mask=None,
        noise=None,
    ) -> Tensor:
        """动作推理：给定图像+语言前缀与状态，解码得到 (*b, T, A) 的动作序列。"""
        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            actions_shape = (
                bsize,
                self.config.n_action_steps,
                self.config.max_action_dim,
            )
            noise = torch.randn(actions_shape, device=device, dtype=dtype)

        # 如果token_ar_mask为None，创建默认的mask（全部为0，表示不使用自回归）
        if token_ar_mask is None:
            token_ar_mask = torch.zeros_like(lang_masks, dtype=torch.int32, device=device)

        # 构建前缀并建立KV缓存（不复用跨调用的 KV）
        prefix_embs, prefix_pad_masks, prefix_att_masks, _lang_start_index = self.embed_img_lang(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )
            

        dt = torch.tensor(-1.0 / self.config.num_steps, dtype=dtype, device=device)
        x_t = noise
        time = torch.tensor(1.0, dtype=dtype, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)

            v_t = self.predict_velocity(
                state, prefix_pad_masks, past_key_values, x_t, expanded_time
            )

            # Euler step
            x_t += dt * v_t
            time += dt

        return x_t

    def predict_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        """在时间 t 预测速度场 v_t（后缀分支）。"""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_state_action(
            state, x_t, timestep
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=(
                {layer_idx: {"key_states": kv["key_states"], "value_states": kv["value_states"]}
                 for layer_idx, kv in past_key_values.items()}
                if past_key_values is not None else None
            ),
            inputs_embeds=[None, suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

    @torch.no_grad()
    def generate_reasoning(
        self,
        images: torch.Tensor,
        img_masks: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        max_new_tokens: int = 512,
        eos_token_id: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """语言路径的增量解码（含 KV 缓存）。

        步骤：
        - 一次性编码 [前三张图 + 语言前缀] 构建 KV 缓存；
        - 逐 token 追加进行生成，mask 限定只能看有效前缀与已生成 token；
        - 结束后可将 ref 图追加进 KV（如需要）。
        """
        device = images.device
        dtype = images.dtype
        bsize = images.shape[0]

        if eos_token_id is None:
            eos_token_id = int(self.config.paligemma_eos_token)

        # 每个样本的有效前缀长度（语言tokens部分）
        prefix_lang_lens = lang_masks.to(dtype=torch.int64).sum(dim=1)

        # 1) 仅用前三张图 + 语言前缀建立KV缓存（不包含最后一张ref图）
        num_images = images.shape[1]
        # 图像嵌入（分离前三张与最后一张）
        images_reshaped = einops.rearrange(images, "b n c h w -> (b n) c h w")
        img_emb_all = self.paligemma_with_expert.embed_image(images_reshaped)
        num_patch = img_emb_all.shape[1]
        img_emb_all = einops.rearrange(img_emb_all, "(b n) l d -> b n l d", b=bsize)
        img_emb_all = img_emb_all.to(dtype=dtype) * (img_emb_all.shape[-1] ** 0.5)

        # patch 级别的图像mask
        img_masks_patch = einops.repeat(img_masks, "b n -> b n l", l=num_patch)

        if num_images >= 4:
            front_img_emb = einops.rearrange(img_emb_all[:, : num_images - 1, :, :], "b n l d -> b (n l) d")
            ref_img_emb = einops.rearrange(img_emb_all[:, num_images - 1 : num_images, :, :], "b n l d -> b (n l) d")
            front_img_masks = einops.rearrange(img_masks_patch[:, : num_images - 1, :], "b n l -> b (n l)")
            ref_img_masks = einops.rearrange(img_masks_patch[:, num_images - 1 : num_images, :], "b n l -> b (n l)")
        else:
            front_img_emb = einops.rearrange(img_emb_all, "b n l d -> b (n l) d")
            ref_img_emb = torch.zeros((bsize, 0, img_emb_all.shape[-1]), dtype=img_emb_all.dtype, device=device)
            front_img_masks = einops.rearrange(img_masks_patch, "b n l -> b (n l)")
            ref_img_masks = torch.zeros((bsize, 0), dtype=torch.bool, device=device)

        # 语言嵌入（仅prefix tokens）
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        num_lang_embs = lang_emb.shape[1]
        lang_emb = lang_emb.to(dtype=dtype) * np.sqrt(lang_emb.shape[-1])

        # 组装前缀（不含ref）：[前3图] + [语言prefix]
        prefix_embs = torch.cat([front_img_emb, lang_emb], dim=1)
        prefix_pad_masks = torch.cat([front_img_masks, lang_masks], dim=1)
        prefix_ar_masks = torch.zeros_like(prefix_pad_masks, dtype=torch.int32, device=device)

        # 构造前缀注意力与位置
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 建立KV缓存
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # 计算第一步的next-token分布（语言区域从 front_img_len 开始）
        front_img_len = front_img_emb.shape[1]
        lang_hidden = prefix_out[:, front_img_len : front_img_len + num_lang_embs, :]
        lm_head = self.paligemma_with_expert.paligemma.language_model.lm_head
        logits = lm_head(lang_hidden).to(dtype=torch.float32)
        last_indices = (prefix_lang_lens - 1).clamp_min(0)
        gather_indices = last_indices.view(bsize, 1, 1).expand(bsize, 1, logits.size(-1))
        last_logits = torch.gather(logits, dim=1, index=gather_indices).squeeze(1)

        def sample_next_token(curr_logits: torch.Tensor) -> torch.Tensor:
            if do_sample:
                if temperature is not None and temperature > 0:
                    curr_logits = curr_logits / max(temperature, 1e-5)
                if top_k and top_k > 0:
                    values, _ = torch.topk(curr_logits, top_k)
                    min_values = values[:, -1].unsqueeze(-1)
                    curr_logits = torch.where(
                        curr_logits < min_values,
                        torch.full_like(curr_logits, -float("inf")),
                        curr_logits,
                    )
                probs = F.softmax(curr_logits, dim=-1)
                return torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                return torch.argmax(curr_logits, dim=-1)

        next_tokens = sample_next_token(last_logits)
        finished = next_tokens.eq(eos_token_id)

        # 记录已生成token；并维护每个样本的下一个位置id
        generated: list[list[int]] = [[] for _ in range(bsize)]
        for i in range(bsize):
            generated[i].append(int(next_tokens[i].item()))

        # 前缀KV长度（所有样本一致）：等于输入prefix_embs的总长度
        prefix_total_len = prefix_embs.shape[1]
        kv_len = prefix_total_len
        # RoPE位置：为每个样本维护下一个token应使用的位置（忽略padding位置）
        next_pos = prefix_pad_masks.to(dtype=torch.int64).sum(dim=1)

        # 继续生成后续token
        for _ in range(max_new_tokens - 1):
            if torch.all(finished):
                break

            # 对已完成样本喂入eos，保持batch一致
            step_tokens = torch.where(
                finished, torch.full_like(next_tokens, eos_token_id), next_tokens
            ).view(bsize, 1)

            # 单步语言token嵌入（与前缀同尺度）
            step_embs = self.paligemma_with_expert.embed_language_tokens(step_tokens)
            step_embs = step_embs.to(dtype=dtype) * np.sqrt(step_embs.shape[-1])

            # 单步注意力mask：允许关注所有有效的前缀位置 + 已生成token + 当前token；屏蔽padding
            kv_len = kv_len + 1  # 追加1个token后KV长度+1
            step_mask = torch.zeros((bsize, 1, kv_len), dtype=torch.bool, device=device)
            # 允许前缀中有效token
            step_mask[:, :, :prefix_total_len] = prefix_pad_masks.unsqueeze(1)
            # 允许已生成token（包括当前步）
            step_mask[:, :, prefix_total_len:kv_len] = True

            # 位置id：按各样本有效前缀计数递增
            step_pos_ids = next_pos.view(bsize, 1)

            (step_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=step_mask,
                position_ids=step_pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=[step_embs, None],
                use_cache=True,
                fill_kv_cache=False,
            )

            step_logits = lm_head(step_out).to(dtype=torch.float32).squeeze(1)
            next_tokens = sample_next_token(step_logits)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, eos_token_id), next_tokens
            )

            for i in range(bsize):
                if not finished[i]:
                    token_id = int(next_tokens[i].item())
                    generated[i].append(token_id)
                    if token_id == eos_token_id:
                        finished[i] = True

            # 已完成样本的位置不再递增
            next_pos = next_pos + (~finished).to(next_pos.dtype)

        # 2) 语言生成完成后，将最后一张ref图插入到结束语言之后（追加到KV缓存）
        ref_len = ref_img_emb.shape[1]
        if ref_len > 0:
            # 构造注意力mask：允许关注所有既有token（prefix_total_len + 已生成的token）以及ref内部双向可见
            total_prev_len = kv_len
            ref_mask = torch.zeros((bsize, ref_len, total_prev_len + ref_len), dtype=torch.bool, device=device)
            # 允许前缀有效位置
            ref_mask[:, :, :prefix_total_len] = prefix_pad_masks.unsqueeze(1)
            # 允许已生成token区间
            if total_prev_len > prefix_total_len:
                ref_mask[:, :, prefix_total_len:total_prev_len] = True
            # 允许ref内部相互可见
            ref_mask[:, :, total_prev_len : total_prev_len + ref_len] = True

            # 位置id为当前next_pos开始，连续ref_len个位置
            ref_pos_ids = next_pos.view(bsize, 1) + torch.arange(ref_len, device=device, dtype=next_pos.dtype).view(1, ref_len)

            # 追加ref到KV缓存
            (_, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=ref_mask,
                position_ids=ref_pos_ids,
                past_key_values=past_key_values,
                inputs_embeds=[ref_img_emb, None],
                use_cache=True,
                fill_kv_cache=False,
            )
            kv_len = kv_len + ref_len

        # 输出：对齐为同长度，使用eos填充
        gen_lengths = [len(g) for g in generated]
        if max(gen_lengths, default=0) == 0:
            return torch.zeros((bsize, 0), dtype=torch.long, device=device)

        max_gen = max(gen_lengths)
        out = torch.full((bsize, max_gen), fill_value=eos_token_id, dtype=torch.long, device=device)
        for i in range(bsize):
            if gen_lengths[i] > 0:
                out[i, : gen_lengths[i]] = torch.tensor(generated[i], dtype=torch.long, device=device)

        return out
