# OpenPI 单任务 / 单扰动评测说明

这份文档只解决一个实际问题：

你现在想用 `pi0` / OpenPI，在 VLA-Arena 里测：

- 整个 suite
- 某一个具体任务
- 某一个具体任务在某一个具体扰动下的结果

并且你希望知道：

- 该改哪些 YAML 字段
- 该执行什么命令
- 结果在哪看

本文档对应当前仓库里的 OpenPI 评测器：

- 环境：`envs/openpi`
- 模型：`openpi`
- 默认配置：[openpi.yaml](/home/wangshuo/VLA-Arena/vla_arena/configs/evaluation/openpi.yaml)

## 1. 先记住 3 个关键字段

### `task_suite_name`

负责选“任务族”。

例如：

- `safety_static_obstacles`
- `distractor_dynamic_distractors`
- `extrapolation_unseen_objects`
- `extrapolation_task_workflows`
- `extrapolation_preposition_combinations`
- `long_horizon`

也支持“基础 suite + 单个扰动”：

```yaml
task_suite_name: "extrapolation_unseen_objects+light"
```

### `task_name`

负责在这个 suite 里只选一个任务。

例如：

```yaml
task_name: "pick the tomato and place it on the plate"
```

或者用内部 task id：

```yaml
task_name: "pick_the_tomato_and_place_it_on_the_plate_0"
```

### `task_names`

负责一次选多个任务。

例如：

```yaml
task_names:
  - "pick the tomato and place it on the plate"
  - "pick the apple and place it on the plate"
```

## 2. 这 3 个字段分别控制什么

最重要的一点：

- `task_suite_name` 决定“在哪一类 benchmark 上测”
- `task_name` / `task_names` 决定“在这类 benchmark 里具体测哪一个或哪几个任务”

所以最常见的 4 种模式就是：

1. 测整个 suite
2. 测 suite 里的一个任务
3. 测整个 suite + 一个扰动
4. 测 suite 里的一个任务 + 一个扰动

## 3. 扰动怎么选

当前 OpenPI 已经支持用 `task_suite_name` 后缀直接打开单个扰动。

支持的后缀是：

- `+light`
- `+blur`
- `+noise`
- `+color`
- `+camera`
- `+layout`
- `+layout_random`
- `+unseen`
- `+lang1`
- `+lang2`
- `+lang3`
- `+lang4`

它们分别表示：

- `light`
  随机调整亮度、对比度、饱和度、色温

- `blur`
  对输入图像施加高斯模糊

- `noise`
  对输入图像加高斯噪声

- `color`
  物体颜色随机化

- `camera`
  相机位置偏移

- `layout`
  切换到同一任务预生成的其他初始布局，优先使用和默认布局差异最大的那些摆放

- `layout_random`
  不使用固定 init state，而是在更大的桌面工作区内随机重新摆放可移动物体

- `unseen`
  把单任务里主要操作的那个物体替换成一个未见过的物体类别，并跳过原始固定 init state

- `lang1` 到 `lang4`
  语言扰动，等价于打开 instruction replacement，并设置不同 `replacement_level`

## 4. language perturbation 为什么看起来“不叫 language”

因为 OpenPI 这里原本的命名不是 `language: true/false`，而是：

- `use_replacements`
- `replacement_probability`
- `replacement_level`

也就是说，语言扰动本来就有，只是名字叫“instruction replacements”。

如果你手动开语言扰动，原生写法是：

```yaml
use_replacements: true
replacement_probability: 1.0
replacement_level: 1
```

如果你想少改字段，直接用：

```yaml
task_suite_name: "extrapolation_unseen_objects+lang1"
```

就可以达到同样效果。

## 5. 最常用的具体操作方式

下面所有命令都默认你在仓库根目录：

`/home/wangshuo/VLA-Arena`

### 5.1 测整个 suite，不加扰动

例如：

“测 `safety_static_obstacles` 这个 suite 的 L0”

YAML 这样写：

```yaml
task_suite_name: "safety_static_obstacles"
task_level: 0
task_name: null
task_names: null
```

执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

### 5.2 测整个 suite，只加一种扰动

例如：

“测 `extrapolation_unseen_objects` 在 light perturbation 下的结果”

YAML：

```yaml
task_suite_name: "extrapolation_unseen_objects+light"
task_level: 0
task_name: null
task_names: null
```

执行命令和上面完全一样：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

### 5.3 只测一个具体任务，不加扰动

例如：

“只测 `pick the tomato and place it on the plate` 这个任务”

YAML：

```yaml
task_suite_name: "safety_static_obstacles"
task_level: 0
task_name: "pick the tomato and place it on the plate"
task_names: null
```

执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

### 5.4 只测一个具体任务，并且只加一种扰动

这是你现在最关心的用法。

例如：

“测 `pick the tomato and place it on the plate` 这个任务，在 light perturbation 下的结果”

YAML：

```yaml
task_suite_name: "safety_static_obstacles+light"
task_level: 0
task_name: "pick the tomato and place it on the plate"
task_names: null
```

执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

### 5.5 一次只测几个你自己挑的任务

例如：

“只测 tomato 和 apple 这两个任务”

YAML：

```yaml
task_suite_name: "safety_static_obstacles"
task_level: 0
task_name: null
task_names:
  - "pick the tomato and place it on the plate"
  - "pick the apple and place it on the plate"
```

### 5.6 一次只测几个你自己挑的任务，并加一种扰动

例如：

“只测 tomato 和 apple，并加 blur”

YAML：

```yaml
task_suite_name: "safety_static_obstacles+blur"
task_level: 0
task_name: null
task_names:
  - "pick the tomato and place it on the plate"
  - "pick the apple and place it on the plate"
```

## 6. 推荐优先用内部 task id

如果自然语言文本在同一个 level 里有重复，字符串匹配可能会歧义。

所以更稳的方式是优先用内部 task id。

例如：

```yaml
task_name: "pick_the_tomato_and_place_it_on_the_plate_0"
```

如果你要查某个 suite 里具体有哪些任务，直接看这份总表：

[vla_arena_task_catalog_zh.md](/home/wangshuo/VLA-Arena/docs/vla_arena_task_catalog_zh.md)

## 7. 你以后最常改的其实就这几行

如果你只是在切换“任务”和“扰动”，通常只需要改：

```yaml
task_suite_name: "safety_static_obstacles+light"
task_level: 0
task_name: "pick the tomato and place it on the plate"
task_names: null
```

别的字段可以先不动。

## 8. 原生 true/false perturbation 开关还能不能用

能用。

也就是说，下面这种老写法仍然有效：

```yaml
task_suite_name: "safety_static_obstacles"
task_level: 0
task_name: "pick the tomato and place it on the plate"

adjust_light: true
blur: false
add_noise: false
randomize_color: false
camera_offset: false

use_replacements: false
```

`layout` 稍微特殊一点，它不是图像后处理，而是切到别的 init state。

所以 `+layout` 的手动等价写法是：

```yaml
task_suite_name: "safety_static_obstacles"
layout_perturbation: true
init_state_selection_mode: "episode_idx"
init_state_offset: 0
init_state_offset_random: false
```

含义是：

- 不再固定用默认第 0 个初始布局
- 先把其他预生成布局按“和默认布局差异大小”排序
- 每个 episode 从差异最大的那些布局开始轮换

`layout_random` 则更激进一些，它会跳过固定 init state，直接依赖环境在 reset 时重新随机摆放。

手动等价写法是：

```yaml
task_suite_name: "safety_static_obstacles"
layout_random: true
```

含义是：

- 不读取 `.pruned_init` 里的固定布局
- 每个 episode reset 时都重新随机摆放
- 采样范围比原始 task region 更大，肉眼更容易看出布局变化

`unseen` 的逻辑则是：

- 对当前单任务的主操作物体做运行时替换
- 例如把 `apple` 改成 `kiwi` / `donut` / `bagel` 之类未见物体
- 同时跳过旧 `.pruned_init`，因为原始固定状态和新物体不再严格对应

手动等价写法是：

```yaml
task_suite_name: "safety_static_obstacles"
unseen_object_perturbation: true
unseen_object_category: "kiwi"   # 可选；null / auto 表示自动选一个
```

更推荐的简写是：

```yaml
task_suite_name: "safety_static_obstacles+unseen"
```

只是如果你已经在 `task_suite_name` 里写了 `+light`、`+blur`、`+layout`、`+layout_random`、`+unseen`、`+lang1`，那就不需要再手动改对应字段了。

建议：

- 你想少改字段：用 `task_suite_name: "<suite>+<扰动>"`
- 你想完全显式控制：直接改各个 true/false 字段

## 9. 结果去哪看

### 实时日志

```bash
tail -f /home/wangshuo/VLA-Arena/experiments/logs/<日志文件>.txt
```

### 最新结果 JSON

```bash
ls -lt /home/wangshuo/VLA-Arena/results/openpi_json_*.json | head
```

### rollout 视频

```bash
find /home/wangshuo/VLA-Arena/rollouts/openpi/$(date +%Y_%m_%d) -type f | sort
```

## 10. 一个你可以直接照抄的完整示例

目标：

- 模型：`pi0`
- suite：`safety_static_obstacles`
- level：`0`
- task：`pick the tomato and place it on the plate`
- perturbation：`light`

把 [openpi.yaml](/home/wangshuo/VLA-Arena/vla_arena/configs/evaluation/openpi.yaml) 改成：

```yaml
inference_mode: "websocket"
train_config_name: "pi0_vla_arena_low_mem_finetune"
policy_checkpoint_dir: "VLA-Arena/pi0-vla-arena-fintuned-LoRA"
policy_checkpoint_step: "latest"
auto_start_policy_server: true

host: "0.0.0.0"
port: 8000
resize_size: 224
replan_steps: 5

task_suite_name: "safety_static_obstacles+light"
task_level: 0
task_name: "pick the tomato and place it on the plate"
task_names: null
num_steps_wait: 10
num_trials_per_task: 10

save_video_mode: "first_success_failure"
use_local_log: true
local_log_dir: "./experiments/logs"
seed: 7
result_json_path: default
```

然后执行：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 \
  uv run --project envs/openpi \
  vla-arena eval --model openpi --config vla_arena/configs/evaluation/openpi.yaml
```

这就是“对具体任务 + 具体干扰项做测量”的完整操作。

## 11. 注意事项

### 11.1 不能只写扰动名

下面这种不行：

```yaml
task_suite_name: "light"
task_suite_name: "blur"
```

必须带基础 suite：

```yaml
task_suite_name: "safety_static_obstacles+light"
task_suite_name: "extrapolation_unseen_objects+blur"
task_suite_name: "long_horizon+layout"
task_suite_name: "safety_static_obstacles+layout_random"
task_suite_name: "safety_static_obstacles+unseen"
```

### 11.2 代理环境

如果你的 shell 里设置了：

- `HTTP_PROXY`
- `HTTPS_PROXY`

OpenPI 的本地 websocket 连接可能失败。

所以建议统一使用本文档里的命令模板，也就是：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  ...
```
