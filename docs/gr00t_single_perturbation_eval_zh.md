# GR00T 单任务 / 单扰动评测说明

这份文档对应当前仓库里的 GR00T 评测器：

- 环境：`envs/gr00t`
- 模型：`gr00t`
- 默认配置：[gr00t.yaml](/home/wangshuo/VLA-Arena/vla_arena/configs/evaluation/gr00t.yaml)

默认推荐跑法不是把 GR00T 模型直接塞进 `VLA-Arena` 的 Python 进程，而是：

1. 在 `VLA-Arena` 里启动评测
2. 由评测器自动拉起 `../Isaac-GR00T/gr00t/eval/run_gr00t_server.py`
3. 通过 ZeroMQ client 在 VLA-Arena 任务上评测 GR00T

这样和你现在手动分别跑 `Isaac-GR00T` / `VLA-Arena` 的方式最接近，也最稳。

## 1. 先改 5 个关键字段

### `policy_mode`

推荐保持：

```yaml
policy_mode: "server"
```

这会让评测器优先连接现有 GR00T server；如果没连上，就按 YAML 里的配置自动启动一个。

### `gr00t_root`

指定 Isaac-GR00T 仓库根目录，例如：

```yaml
gr00t_root: "../Isaac-GR00T"
```

如果你要评测 `Libero_GR00T` 里的 GR00T N1.6，也可以直接写：

```yaml
gr00t_root: "/home/wangshuo/Libero_GR00T"
```

### `model_path`

指定你要评测的 GR00T checkpoint，例如：

```yaml
model_path: "../Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial"
```

如果是 `Libero_GR00T` 里的 N1.6 checkpoint，可以写：

```yaml
model_path: "/home/wangshuo/Libero_GR00T/examples/LIBERO/checkpoint/libero_spatial"
```

### `embodiment_tag`

在 VLA-Arena 上，推荐用：

```yaml
embodiment_tag: "LIBERO_PANDA"
```

### `task_suite_name`

决定你测哪个 benchmark suite，也支持“基础 suite + 单个扰动”：

```yaml
task_suite_name: "safety_static_obstacles+blur"
```

## 2. 任务选择字段

### `task_name`

只跑一个任务：

```yaml
task_name: "pick_the_apple_and_place_it_on_the_bowl_1"
```

也可以写自然语言任务文本，只要和该 suite 里的任务精确匹配。

### `task_names`

一次跑多个任务：

```yaml
task_names:
  - "pick_the_apple_and_place_it_on_the_bowl_1"
  - "pick_the_tomato_and_place_it_on_the_plate_1"
```

## 3. 支持的扰动后缀

当前 `gr00t` evaluator 已支持通过 `task_suite_name` 后缀直接打开这些扰动：

- `+light`
- `+blur`
- `+noise`
- `+color`
- `+camera`
- `+layout`
- `+layout_random`
- `+lang1`
- `+lang2`
- `+lang3`
- `+lang4`

例如：

```yaml
task_suite_name: "safety_static_obstacles+camera"
```

注意：

- `+unseen` 目前还没有在 `gr00t` evaluator 里实现
- `layout_random` 会跳过固定 init state，直接依赖环境自身随机摆放

## 4. 最常用命令

默认工作目录：

`/home/wangshuo/VLA-Arena`

执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/gr00t \
  vla-arena eval --model gr00t --config vla_arena/configs/evaluation/gr00t.yaml
```

## 5. 一个最接近你当前需求的例子

如果你要测：

- GR00T
- `safety_static_obstacles`
- level 1
- 单任务 `pick_the_apple_and_place_it_on_the_bowl_1`
- `blur` 扰动

那么 YAML 可以写成：

```yaml
policy_mode: "server"
gr00t_root: "../Isaac-GR00T"
model_path: "../Isaac-GR00T/checkpoints/GR00T-N1.7-LIBERO/libero_spatial"
embodiment_tag: "LIBERO_PANDA"

task_suite_name: "safety_static_obstacles+blur"
task_level: 1
task_name: "pick_the_apple_and_place_it_on_the_bowl_1"
task_names: null
num_trials_per_task: 1
```

然后直接执行上一节的命令即可。

## 6. 结果位置

- 聚合 JSON：`./results/gr00t_json_<timestamp>.json`
- 日志：`./experiments/logs/EVAL-*.txt`
- 视频：`./rollouts/gr00t/<date>/`

## 7. 如果你想手动起 server

也可以先单独启动：

```bash
cd /home/wangshuo/Isaac-GR00T
uv run python gr00t/eval/run_gr00t_server.py \
  --model-path checkpoints/GR00T-N1.7-LIBERO/libero_spatial \
  --embodiment-tag LIBERO_PANDA \
  --host 127.0.0.1 \
  --port 5555 \
  --use-sim-policy-wrapper
```

然后把 `gr00t.yaml` 里保持：

```yaml
policy_mode: "server"
auto_start_policy_server: false
host: "127.0.0.1"
port: 5555
```

这样 `VLA-Arena` 只负责连这个 server，不会重复拉起新进程。
