
<h1 align="center">Action-Sketcher: From Reasoning to Action via Visual Sketches for Long-Horizon Robotic Manipulation</h1>

<h3 align="center">Make intent visible. Make action reliable.</h3>


<p align="center">
  <a href="https://arxiv.org/pdf/2601.xxxxx"><img src="https://img.shields.io/badge/arXiv-2601.xxxxx-b31b1b.svg" alt="arXiv"></a>
  &nbsp;
  <a href="https://action-sketcher.github.io/"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="Project Homepage"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/🤗%20Dataset-Stay%20tuned-green.svg" alt="Benchmark"></a>
  &nbsp;
  <a href="#"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Stay%20tuned-yellow" alt="Weights"></a>
</p>

## 🔥 Overview

<div style="text-align: center; background-color: white;">
    <img src="assets/teasor.png" width=100% >
</div>

**Action-Sketcher** operates in a ***See-Think-Sketch-Act*** loop, where a foundation model first performs temporal and spatial reasoning to decompose a high-level instruction (e.g., "Clean the objects on the table") into a subtask and a corresponding **Visual Sketch**. This sketch, composed of primitives like points, boxes, and arrows, serves as an explicit, human-readable plan that guides a low-level policy to generate robust action sequences. This methodology enables three key capabilities: ***(bottom left)*** long-horizon planning through task decomposition, ***(bottom middle)*** explicit spatial reasoning by grounding instructions in scene geometry, and ***(bottom right)*** seamless human-in-the-loop adaptability via direct sketch correction and intent supervision.


## 🗞️ News
- **`2026-01-05`**: ✨ ***Codes, Dataset and Weights are coming soon! Stay tuned for updates***.
- **`2026-01-05`**: 🔥 We released our [Project Page](https://action-sketcher.github.io/) of **Action-Sketcher**.


## 🎯 TODO
- [ ] Release the model checkpoints and inference codes used in our paper *(About 2 week)*.
- [ ] Release the full dataset and training codes *(About 1 month)*.
- [ ] Release the Dataset Generation Pipeline *(Maybe 1 month or more)*.


## 🤖 Method

<div align="center"> 
    <img src="assets/method.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>

The Action-Sketcher framework is **model-agnostic** and can be integrated with any VLA model with an event-driven loop that (i) summarizes the next subtask, (ii) emits a compact Visual Sketch (points, boxes, arrows, relations) that externalizes spatial intent, and (iii) synthesizes an action chunk conditioned on that sketch and the robot state. The explicit intermediate supports targeted supervision, on-the-fly correction, and reliable long-horizon execution within a single-model architecture.

## ✨ Experiments
<div align="center"> 
    <img src="assets/result.png" alt="Logo" style="width=100%;vertical-align:middle">
</div>


## 📑 Citation

If you find our work helpful, feel free to cite it:
```
@article{tan2026action,
  title={Action-Sketcher: From Reasoning to Action via Visual Sketches for Long-Horizon Robotic Manipulation},
  author={Tan, Huajie and Co, Peterson and Xu, Yijie and Rong, Shanyu and Ji, Yuheng and Chi, Cheng and others},
  journal={arXiv preprint arXiv:2601.xxxxx},
  year={2026}
}
```