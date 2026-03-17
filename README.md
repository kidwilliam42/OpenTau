<p align="center">
  <a href="https://www.tensor.auto">
    <img src="assets/logo.png" alt="Logo">
  </a>
</p>

<p align="center">
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/cpu_test.yml?query=branch%3Amain"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/cpu_test.yml/badge.svg?branch=main" alt="CPU Tests"></a>
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/gpu_test.yml"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/gpu_test.yml/badge.svg" alt="Nightly GPU Tests"></a>
  <a href="https://github.com/TensorAuto/OpenTau/actions/workflows/regression_test.yml"><img src="https://github.com/TensorAuto/OpenTau/actions/workflows/regression_test.yml/badge.svg" alt="Nightly Regression Tests"></a>
  <a href="https://opentau.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/opentau/badge/?version=latest" alt="Documentation"></a>
  <a href="https://pypi.org/project/opentau/"><img src="https://img.shields.io/pypi/v/opentau" alt="Version"></a>
  <a href="https://pypi.org/project/opentau/"><img src="https://img.shields.io/pypi/status/opentau" alt="Status"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/pypi/pyversions/opentau" alt="Python versions"></a>
  <a href="https://github.com/TensorAuto/OpenTau/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"></a>
  <a href="https://hub.docker.com/r/tensorauto/opentau"><img src="https://img.shields.io/docker/v/tensorauto/opentau?label=Docker" alt="Docker"></a>
  <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" alt="pre-commit"></a>
</p>

# OpenTau - Train VLA models with state-of-the-art techniques by Tensor

At Tensor, we are pushing the frontier of large foundation models for physical AI. In robot learning, a vision-language-action (VLA) model is a multimodal foundation model that integrates vision, language, and action. Today, VLA represents the leading approach for embodied AI, spanning autonomous driving, robot manipulation, and navigation.

OpenTau is Tensor’s open-source training toolchain for frontier VLA models—designed to make training reproducible, accessible, and scalable. At Tensor, we believe in open research and reproducible progress for the robotics community. By open-sourcing our training toolchain, we aim to expand knowledge sharing and accelerate scientific progress that others can reproduce.

Whether you use the official OpenPi codebase or LeRobot’s reimplementation, you may still be missing key components. OpenTau implements these key capabilities in one place:

- Co-training on an adjustable mixture of heterogeneous datasets
- Discrete actions for fast VLM convergence in $\pi_{0.5}$
- Knowledge insulation between the VLM backbone and the action expert
- Dropout in the VLM to reduce overfitting
- A reinforcement learning pipeline described in $\pi^*_{0.6}$
- And more...

OpenTau ($\tau$) is a tool developed by *[Tensor][1]* to bridge this gap, and we also use it internally to train our proprietary in-house models. Our goal is to help you train VLAs on any dataset while fully leveraging state-of-the-art techniques. We plan to continuously upgrade this repository to keep pace with the state of the art in the robotics community.

|                                                 Features |         OpenPi          |             LeRobot              | **OpenTau** |
|---------------------------------------------------------:|:-----------------------:|:--------------------------------:|:-----------:|
|                  Co-training with Heterogeneous Datasets |            ❌            |                ❌                 |      ✅      |
|                 Discrete Actions Training in $\pi_{0.5}$ |            ❌            |                ❌                 |      ✅      |
| Knowledge Insulation (KI) between VLM and Action Decoder |            ❌            |                ❌                 |      ✅      |
|                              Dropout Layers in PaliGemma | ✅ (Jax) <br>❌ (PyTorch)|                ❌                |      ✅     |
|                        Multi-Node and Multi-GPU Training |            ❌            |                ✅                 |      ✅      |
|                 Fully Functioning $\pi_{0.5}$ Checkpoint |            ✅            | ❌ <br> (Missing Text Embeddings) |      ✅      |
|                       Visualize dataset with URDF models |            ❌            |                ❌                 |      ✅      |
|            Simulation Environments for Evaluating Models |            ❌            |                ✅                 |      ✅      |
|                 Create Validation Splits During Training |            ❌            |                ❌                 |      ✅      |
|    $\pi^{*}_{0.6}$ style Reinforcement Learning Pipeline |            ❌            |                ❌                 |      ✅      |
|                        Post-training on Human Data (Beta)|            ❌            |                ❌                 |      ✅      |
|                                                Framework |      Jax / PyTorch       |             PyTorch               |   PyTorch    |

## Quick Start
If you are familiar with LeRobot, getting started with OpenTau is very easy.
Because OpenTau is a fork of the popular LeRobot repository, any LeRobot-compliant policy and dataset can be used directly with OpenTau.
Check out our [documentation](https://opentau.readthedocs.io/) to get started quickly.
We provide a [quick start guide](https://opentau.readthedocs.io/en/latest/getting_started.html) to help you get started with OpenTau.

For using local notebooks to train and evaluate models, find the notebooks at [notebooks/pi05_training.ipynb](https://github.com/TensorAuto/OpenTau/blob/main/notebooks/pi05_training.ipynb) and [notebooks/pi05_evaluation_only.ipynb](https://github.com/TensorAuto/OpenTau/blob/main/notebooks/pi05_evaluation_only.ipynb).

For using the Google Colab notebooks to train and evaluate models, find the colab notebooks here: [pi05_training](https://colab.research.google.com/drive/1DeU0lNnEzs1KHo0Nkgh4YKBr-xu9moBM?usp=sharing) and [pi05_evaluation_only](https://colab.research.google.com/drive/1U_AyuH9WYMT4anEWvsOtIT7g01jA0WGm?usp=sharing) respectively.

## Checkpoints
We provide fully functioning $\pi_{0.5}$ checkpoints trained with high success rates. We plan to release more models in the near future.

| Model Checkpoint              | Description                                                                                                   | Success Rate (%)                                                   |
|-------------------------------|---------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| [TensorAuto/tPi0.5-libero][2] | A $\pi_{0.5}$ model checkpoint trained on the LIBERO dataset with discrete actions and knowledge insulation.  | 98.4% (10) <br> 97.6% (Goal) <br> 100% (Object) <br> 98% (Spatial) |
| [TensorAuto/pi05_base][5]     | A $\pi_{0.5}$ model checkpoint converted from the official openpi checkpoint, with language embeddings added. | N/A                                                                |
| More coming soon...           |                                                                                                               |                                                                    |

## 与上游 OpenTau 官方代码的主要差异

本仓库在 [OpenTau 官方仓库](https://github.com/TensorAuto/OpenTau) 基础上做了三项主要改动。下面用通俗的语言解释每项改动做了什么、为什么要做。

---

### 改动一：给机器人加了一个"大脑"——层级任务规划

**一句话概括：** 让一个会"看图说话"的 AI（Qwen3-VL）充当指挥官，把复杂任务拆成一步步的小指令，再交给机器人手臂去执行。

**通俗解释：**

想象你让机器人完成"把两个摩卡壶都放到炉子上"这个任务。官方的 OpenTau 只有一个"手"——低层策略（π₀.₅），它只知道根据当前看到的画面做出下一个动作，但不会主动拆解任务。这就好比让一个工人干活，但没有工头告诉他先做什么、后做什么。

我们加的"大脑"就是这个工头。它的工作流程是：
1. **看一眼当前场景**（通过摄像头图像）
2. **想一想下一步该做什么**（比如"先把右边的摩卡壶对准炉子"）
3. **把指令发给"手"去执行**（低层策略执行 10-30 步动作）
4. **执行完了再看一眼**，决定下一个子任务
5. **循环往复**，直到任务完成或步数用完

**涉及的文件：**

| 文件 | 作用 |
|---|---|
| `src/opentau/agents/hierarchical_agent.py` | "大脑"和"手"之间的协调逻辑 |
| `src/opentau/planner/qwen3_vl_planner.py` | "大脑"本身——调用 Qwen3-VL 模型进行规划 |
| `src/opentau/planner/qwen_prompts.yaml` | 给"大脑"的指令模板（提示词），定义了它该怎么思考 |
| `src/opentau/scripts/hierarchical_eval.py` | 运行评估的主脚本 |
| `configs/examples/pi05_hierarchical_eval_config.json` | 评估用的配置参数 |
| `src/opentau/configs/default.py` | 可调节的参数定义（每步多少动作、最多规划几次等） |

我们还准备了三种不同风格的提示词，适用于不同场景：
- **通用型**：适合大多数任务
- **简短操作型**：指令更简洁，适合简单操作任务
- **保守操作型**：更谨慎，会先对准再抓取，适合精细操作

---

### 改动二：把模型搬到本地——支持断网运行

**一句话概括：** 官方代码运行时会从 HuggingFace 网站下载模型，但我们的服务器没有外网，所以把所有模型提前下载好放在本地，并修改代码直接读取本地文件。

**通俗解释：**

官方代码里写的是"去 HuggingFace 网站下载 `google/paligemma-3b-pt-224` 这个模型"，就像写了个网址让程序自己去下。但我们的服务器连不上外网，所以：
1. 先在能上网的电脑上把模型下载好
2. 拷贝到服务器的 `/home/yjc/models/` 目录下
3. 把代码里所有"网址"改成"本地文件路径"

改动涉及三个模型：

| 模型 | 用途 | 本地路径 |
|---|---|---|
| PaliGemma | 机器人的"眼睛"（VLM 骨干网络），负责理解图像 | `/home/yjc/models/paligemma-3b-pt-224` |
| Qwen3-VL | 机器人的"大脑"（高层规划器），负责规划任务 | `/home/yjc/models/Qwen3-VL-4B-Instruct` |
| Fast tokenizer | 离散动作 tokenizer，仅在模型初始化时加载以构建网络结构，**推理时不参与动作生成** | `/home/yjc/models/fast` |

> **注意：** π₀.₅ 有两条动作输出路径——基于扩散模型（Flow Matching）的连续动作和基于 Fast tokenizer 的离散动作。**推理时只使用扩散模型生成动作**，Fast tokenizer 仅在训练时提供辅助监督信号（CE loss）。初始化时仍需加载 Fast tokenizer，因为需要其 `vocab_size` 来构建与 checkpoint 权重匹配的网络结构。如果只需要纯扩散模型（不含离散动作分支），应使用 π₀ 而非 π₀.₅。

另外还加了一个便利功能：如果将来想换模型路径，不用改代码，只需设置环境变量 `OPENTAU_PALIGEMMA_ID` 即可。

---

### 改动三：让"大脑"更聪明——提示词优化

**一句话概括：** 修改了给 AI 规划器的指令，解决了它"偷懒"的问题——以前做了一半就说"我做完了"，现在会认真检查是否真的全做完了。

**通俗解释：**

改之前的问题：任务是"把两个摩卡壶都放到炉子上"，规划器放了一个之后就说"任务完成了"。这就好比老师让你做 10 道题，你做了 1 道就交卷了。

原因是提示词里只写了"如果任务完成了就返回 done"，但没有强调怎么才算"完成"。

改进后的提示词增加了四条规则：
1. **看图确认**：必须看着摄像头图像亲眼确认所有东西都到位了，才能说"完成"
2. **逐个检查**：如果任务说"两个都要放好"，就必须确认两个都放好了，不能只放一个就算完
3. **失败重试**：如果上一步可能没做成功（比如没抓住），就重试，而不是跳过
4. **不要自欺欺人**：不能因为"尝试过了"就当作"做到了"，必须用眼睛（图像）来验证

这个改动效果明显：
- 改之前：平均只执行 40 步就放弃，成功率 0%
- 改之后：平均执行 456 步持续尝试，成功率提升到 50%

## Acknowledgements

This project builds on the $\pi$ series of [papers][3] and many other open-source efforts—especially [LeRobot][4]—for re-implementing the $\pi$ models and helping standardize training infrastructure. OpenTau extends these foundations to provide a more accessible, comprehensive toolchain for training vision-language-action agents.

[1]:	https://www.tensor.ai
[2]:	https://huggingface.co/TensorAuto/tPi0.5-libero
[3]:	https://www.pi.website/blog
[4]:	https://huggingface.co/lerobot
[5]:    https://huggingface.co/TensorAuto/pi05_base
