# RPBE-VLA（MemoryVLA × LIBERO-Mem）跑通说明 — 只跑我们的方法

目标：在 LIBERO-Mem 上跑 **RPBE 臂（`--arm gamma-rpbe`）** 的训练 + 评测。
本文件自包含，按顺序照做即可。**第一步先跑 T3 任务数据集。**

---

## 0. 硬件与前提

- 单张 **A100 40GB** 可用即可（实测训练占 ~32 GB）
- 磁盘：**至少 200 GB**（openvla 基座 30G + Llama-2 13.5G + T3 数据 18.4G + metainfo 0.7G + checkpoint 若干 1.85G）
- 需要能连 **HuggingFace 镜像**（`hf-mirror.com`），见 §3

### 0.1 配套脚本（随本文件一起给你）

目录 `vla_eval/` 下 6 个文件，**必须一起拿到**：

| 文件 | 用途 |
|---|---|
| `tiered_eval_t1.py` | **LIBERO 评测器**（名字虽叫 t1，已泛化，任何 task 都能跑） |
| `run_tiered.sh` | 评测启动器，**内置 `ulimit -n 65536`**（见 §7.5） |
| `finalize_run_dir.py` | 补齐 run 目录的 `config.json` / `dataset_statistics.json`（见 §7.6） |
| `compare_images.py` | 验证图像翻转方向的实测脚本（见 §7.1） |
| `watch_steps.py` | 按指定步数保存 checkpoint 快照 |
| `run_t1_rpbe.sh` | 训练启动器模板（路径需按你的环境改） |

脚本里的路径可用环境变量覆盖，**不用改代码**：

```bash
export RPBE_VLA_ROOT=<ROOT>          # 所有默认路径的根
export LIBERO_MEM_REPO=<libero-mem clone 路径>
export MEMVLA_DIR=<PRSS2>/third_party/memoryvla
export PRSS2_SRC=<PRSS2>/src
export RPBE_PYTHON=<ROOT>/env_memvla/bin/python
```

---

## 1. 代码

```
git clone https://github.com/fengshiyangrouguan/PRSS2.git
cd PRSS2
git checkout develop_VLA        # 确认 HEAD = 29c1368
```

**跟 VLA 相关的全部代码在这两处：**
- `third_party/memoryvla/` — MemoryVLA 整棵树（训练入口 `train_libero_mem_rpbe.py` 在这里）
- `src/rpbe_embodied/` — RPBE 的 embodied 适配层（Gamma、可行性投影）

其余目录（`old/`、`papers/`、`scripts/`）跟 VLA 无关，可忽略。整份 clone 约 42 MB。

> 注意：远端另有 `fix/avg-lora-clock` 等分支，**跑 RPBE 不要用它们**，就用 `develop_VLA`。

---

## 2. 权重（都不在 git 里，要单独下）

| 用途 | 来源 | 落地路径 | 大小 |
|---|---|---|---|
| **VLA 基座**（必须） | HF `openvla/openvla-7b-prismatic` | `<ROOT>/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt` | 30.17 GB |
| **Llama-2-7b-hf**（必须） | HF `NousResearch/Llama-2-7b-hf` | `<ROOT>/Llama-2-7b-hf/` | 13.5 GB |

**两个坑：**

1. **基座不要在 HF 上找 `openvla/openvla-7b`** —— 那个仓现在只放 safetensors 分片，**没有** `checkpoints/*.pt`。要的是 **`openvla/openvla-7b-prismatic`**。
2. **Llama-2 用 `NousResearch/Llama-2-7b-hf`**，不要用官方 `meta-llama/Llama-2-7b-hf`（gated，镜像下会 403）。权重的 config/tokenizer 才是关键，实际权重会被 VLA checkpoint 覆盖。
   把 `Llama-2-7b-hf` 里的 `pytorch_model*.bin` 删掉可省约 17 GB（只用 safetensors）。

`Llama-2-7b-hf` 的路径靠环境变量传给代码：`LLAMA2_LOCAL_PATH=<ROOT>/Llama-2-7b-hf`

---

## 3. 数据集

HF 仓：**`libero-mem/LIBERO-Mem`**（不是标准 LIBERO，两边任务定义不兼容）。内容：

- 10 个任务 hdf5，共 **211.5 GB**（不要全下）
- **`metainfo.json`（0.714 GB）—— 必需，漏了直接跑不起来**

**T3 要下的两个文件：**

```
KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times_demo.hdf5   18.4 GB
metainfo.json                                                                     0.714 GB
```

下到同一目录，例如 `<ROOT>/datasets/LIBERO-Mem/`。

**为什么 `metainfo.json` 非下不可**：数据加载器第一件事就是读它，并且从里面取 `task_description`。没有它，哪怕 hdf5 完整也会当场报 `ValueError: task ... missing from metainfo.json`。

**HF 访问方式**（直连通常不通）：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
用 `huggingface_hub.snapshot_download` 或 `hf download`，带 `--include` 只拉上面两个文件。

> 训练脚本的 `--task-filter` **默认值就是 `KITCHEN_SCENE1_3`**，所以 T3 是原生支持的，不用改代码。

---

## 4. 环境

### 4.1 conda 在这类机器上通常是坏的
如果 `conda create` 报 `UnavailableInvalidChannel: HTTP 403`，那是 .condarc 里配的国内镜像挂了。**别折腾 conda，用 uv：**

```bash
pip install uv
uv venv --python 3.10 <ROOT>/env_memvla
```

### 4.2 装依赖

```bash
UVM="uv pip install --python <ROOT>/env_memvla/bin/python"

# torch（按显卡架构选 cu128；A100 用这个没问题）
$UVM --index-url https://download.pytorch.org/whl/cu128 \
     --extra-index-url https://pypi.org/simple \
     torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# 训练侧
$UVM "accelerate>=0.25.0" "draccus>=0.8.0" einops json-numpy jsonlines \
     matplotlib "peft==0.11.1" protobuf rich "sentencepiece==0.1.99" \
     "timm==0.9.10" "tokenizers==0.19.1" "transformers==4.40.1" wandb \
     "huggingface-hub==0.29.3" "numpy==1.26.4" "opencv-python==4.11.0.86" \
     h5py pillow flask transforms3d
```

**不需要装：**
- **flash-attn** —— 代码里被强制关掉了（`prismatic/models/backbones/llm/llama2.py` 里 `use_flash_attention_2 = False`）
- **tensorflow** —— RLDS 那条链被 `try/except ImportError` 挡住了，训练走 HDF5

### 4.3 评测还要仿真栈（同一个 env 里装）

```bash
# 系统库（无头渲染必需，缺了会报 eglQueryString 相关错误）
apt-get install -y libegl1 libegl-mesa0 libgl1 libglx-mesa0 libosmesa6 \
                   libglew2.2 libglib2.0-0 libsm6 libxext6 libxrender1 libxi6

$UVM mujoco==3.1.6 bddl==1.0.1 gym==0.25.2 cloudpickle==2.1.0 future easydict thop
$UVM -e <LIBERO_MEM_REPO>/thirdparty/robosuite
```

其中 **`<LIBERO_MEM_REPO>` 是仿真环境**，要单独 clone：

```bash
git clone https://github.com/libero-mem/libero-mem.git     # 约 1.5 GB，自带 assets 和 bddl
```

**不要用标准 LIBERO**（`Lifelong-Robot-Learning/LIBERO`）—— 它的任务名是
`KITCHEN_SCENE1_put_the_black_bowl_on_the_plate`，而 LIBERO-Mem 的是
`KITCHEN_SCENE1_1_...` / `KITCHEN_SCENE1_3_...`，**是另一套 benchmark，没有我们的任务定义**。

**`libero` 包不要 pip 装**（它的 setup.py 布局有问题，装了也 import 不到）。用 `PYTHONPATH` 指过去即可。

---

## 5. 训练

```bash
cd <PRSS2>/third_party/memoryvla

export PYTHONPATH=<PRSS2>/src
export RPBE_EMBODIED_PATH=<PRSS2>/src
export LLAMA2_LOCAL_PATH=<ROOT>/Llama-2-7b-hf
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ulimit -n 65536                      # ← 见 §7.5，必须

<ROOT>/env_memvla/bin/python -u train_libero_mem_rpbe.py \
  --pretrained-checkpoint <ROOT>/openvla-7b-prismatic/checkpoints/step-295000-epoch-40-loss=0.2200.pt \
  --data-root <ROOT>/datasets/LIBERO-Mem \
  --task-filter KITCHEN_SCENE1_3 \
  --arm gamma-rpbe \
  --run-dir <ROOT>/runs/t3_rpbe_seed42 \
  --max-steps 20000 \
  --eval-every 2000 \
  --checkpoint-every 1000 \
  --snapshot-steps "10000,15000,20000" \
  --no-fullstate 1 \
  --seed 42
```

**必须 cd 到 `third_party/memoryvla`**（训练入口会 `sys.path.insert(0, HERE)`）。

**期望看到的关键行**（用来判断跑对了）：
```
scope=full | task params: 922.13M | gamma params: 0.57M      ← gamma 必须非 0
== training loop start (arm=gamma-rpbe) ==
[gamma proj] ... proj_feasible=True ...
[gamma step] ...
```
`gamma params: 0.00M` 说明退化成 avg 基线了，是错的。

**速度/时间**：约 1.26 秒/optimizer step（A100 40G）→ **20000 步约 7 小时**。
RPBE 的可行性投影只比 avg 慢约 8.6%。

**跑挂/中断后重启**：`--init-from-weights <某个 .pt>` 只加载权重、optimizer 全新、从 0 开始（不是续训）。

---

## 6. 评测（LIBERO 仿真 rollout）

标准 memoryvla 的 `evaluation/libero/eval_libero.py` **跑不了 LIBERO-Mem**：
- 它的 `max_steps` 字典里没有 `libero_mem` 分支 → NameError
- 它遍历整个 suite，第一个缺 `.pruned_init` 的任务就崩

**用我们改好的**（`tiered_eval_t1.py`，名字虽叫 t1 但已泛化，任何 task 都能跑）：

```bash
cd <LIBERO_MEM_REPO>
export PYTHONPATH=<PRSS2>/third_party/memoryvla:<LIBERO_MEM_REPO>
export MUJOCO_GL=egl
ulimit -n 65536

<ROOT>/env_memvla/bin/python <SCRIPTS>/vla_eval/tiered_eval_t1.py \
  --ckpt <ROOT>/runs/t3_rpbe_seed42/snapshot_15000.pt \
  --task KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times \
  --n 20 --maxsteps 600 \
  --out-dir <ROOT>/eval/t3_rpbe_15k
```

**评测前先确认 checkpoint 目录里有这两个文件**（缺了会报错）：
```
<run_dir>/config.json
<run_dir>/dataset_statistics.json
```
正常跑完的 run 会自动写；**如果训练是被中途 kill 的，必须补跑**（见 §7.6）。

输出：逐 demo 的 JSONL + summary（`complete` / `strict` / `overshot` 计数）。

**评测协议**（LIBERO-Mem 官方口径，勿改）：
```
num_trials_per_task = 20
num_steps_wait      = 20
seed                = 7
resolution          = 256
```

---

## 7. 会踩的坑（都是实测踩过的，不是推测）

### 7.1 图像必须「上下翻转」，不是 rot180
LIBERO-Mem 的 hdf5 里存的是仿真画面的**垂直翻转**。实测比对：
```
sim 帧 vs hdf5 帧的 mean|diff|
  vflip   7.5    ← 就是这个
  rot180  44.0
  identity 89.1
```
所以送模型前是 `obs["agentview_image"][::-1, :]`。

**memoryvla 官方的 `get_libero_image` 用的是 `img[::-1, ::-1]`（rot180），对 LIBERO-Mem 是错的** —— 会喂左右镜像的图，表现为"机器人能动、能做一半、但永远完不成任务"。**这是最容易中的坑。**

**换数据集/任务时先自测一遍**（下完 T3 数据后建议就跑，30 秒）：
```bash
export RPBE_VLA_ROOT=<ROOT> LIBERO_MEM_REPO=<libero-mem clone>
export RPBE_TASK=KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times
<ROOT>/env_memvla/bin/python vla_eval/compare_images.py
```
它会把 sim 渲染的帧和 hdf5 里存的帧按 identity / hflip / vflip / rot180 四种比一遍，**哪一行数值最小就是正确变换**。不确认这一步，后面所有 rollout 数字都是废的。

### 7.2 子目标状态机必须手动驱动
LIBERO-Mem 的顺序型目标，`env.step()` 返回的 `done` **结构性地永远为 False**：
```python
done = self._check_success()          # inc 默认 False
# 而 sequence 分支只在 inc=True 时才推进计时器
if live_time > 5 and nonlive_time > 15: ... return True
else:
    if inc: live_time += 1            # ← 永不执行
```
必须自己每步调：
```python
base = env.env                        # wrapper 没有 __getattr__ 转发
base.reset_subgoal_progress()         # 每个 demo 开始
base._overshot = False                # 官方 reset 不清这个，会跨 demo 污染
base._check_success(inc=True)         # 每个 env.step 之前
```

### 7.3 不要拿 `_check_success` 的返回值当成功
它在**中间子目标达成时也返回 True**。据此 break 会把一条本来会成功的轨迹截断。
正确做法：只有 `len(base._satisfied_subgoals) >= n_subgoals` 才算成功。

### 7.4 不要因为 `env.step` 返回 done 就 break
同 7.3 的理由。

### 7.5 fd 上限（第 20 个 demo 必崩）
容器默认 `ulimit -n 1024`，而 wandb 临时目录会泄漏 fd。跑到第 20 个 demo 时 robosuite 打不开 mesh 文件，报
`resource not found: .../link1_vis.obj`。
**启动脚本里加 `ulimit -n 65536`。**

### 7.6 中途停掉的 run 没有 `dataset_statistics.json`
训练脚本只在**全部训练结束后**才写它。如果是被 kill 的，评测前要先补：

```bash
export RPBE_VLA_ROOT=<ROOT> RPBE_TASK_FILTER=KITCHEN_SCENE1_3
<ROOT>/env_memvla/bin/python vla_eval/finalize_run_dir.py \
  --run-dir <ROOT>/runs/t3_rpbe_seed42 \
  --task-filter KITCHEN_SCENE1_3
```
（同时会补 `config.json`，评测加载器也需要它。脚本会打印最终校验，必须看到 `OK`。）

### 7.7 checkpoint 是「训练存档」不是「可部署模型」
`_ckpt_dict()` 只存 `requires_grad` 的参数（约 880 个张量 / 1.85 GB），**不含冻结的 7B 主干**。
所以**不能**直接喂给 memoryvla 的 `deploy.py`（它要完整 nested state dict）。
我们的 `tiered_eval_t1.py` 已经内置了两阶段恢复（base + delta + 重挂 LoRA + 覆盖 norm_stats），直接用即可。

### 7.8 T3 的 `.pruned_init` 可能没有
`benchmark.get_task_init_states()` 要 `<suite>/<task>.pruned_init`，而仓库只带了 4 个任务的。
我们的评测器走 `metainfo.json` 的 `initial_state` 字段，**不受影响**。

---

## 8. 参考数字（T1 上已跑出来的，用来判断 T3 结果是否合理）

任务 `KITCHEN_SCENE1_1`（1 个子目标），20 次 rollout：

| 臂 | complete (SR) | strict | overshot |
|---|---|---|---|
| `avg`（官方平均合并基线） | 8/19 = 42.1% | 4/19 | 4 |
| `gamma-rpbe` @15000 | 8/20 = 40.0% | 4/20 | 4 |
| `gamma-rpbe` @12000 | 6/20 = 30.0% | 3/20 | 3 |
| `gamma-rpbe` @18000 | 5/16（未跑完） | 2/16 | 3 |

**注意**：T1 是单子目标短程任务，两者打平（val 曲线也逐点重合）。**T3 有 3 个顺序子目标，是长程任务** —— 如果 RPBE 有意义，更该在这里体现。

`strict` vs `complete` 的区别：`complete` 是子目标达成；`strict` 额外要求达成后不 overshoot（最终目标不能连续 5 帧失效）。**报告时两个都要给。**

---

## 9. 建议的执行顺序

1. clone PRSS2（§1）
2. 下权重（§2）—— 这一步最久，30 GB + 13.5 GB
3. **下 T3 数据 + metainfo.json（§3）** ← 你要求的第一步
4. 建 env（§4）
5. 起训练（§5），约 7 小时
6. 中途用 `--snapshot-steps` 留 checkpoint
7. 训练完或到达目标步数后评测（§6）
8. 报告 `complete` / `strict` / `overshot` 三个数，以及成功的 demo id 列表

---

## 10. 有任何一步报错

把**完整报错 + 当时的命令行 + 相关日志尾部**发回来。特别注意区分：
- **环境/加载类错误** → 多半是 §7 里的坑
- **`proj_feasible=False` 或 `aborted=True`** → 这是 **RPBE 方法层面的信号**（可行性投影没解出来），不是崩溃，要单独报
