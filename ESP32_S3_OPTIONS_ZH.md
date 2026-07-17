# ESP32-S3 自定义唤醒词方案选择

> 核验日期：2026-07-16。以下结论以 ESP32-S3 端侧、持续监听、离线推理为目标。

## 结论

如果要做**任意中文自定义词**并保留开源、可复现能力，继续使用本项目的
microWakeWord 路线最合适。它输出 ESPHome 可加载的 INT8 `.tflite`，但模型质量
依赖真实录音、困难负样本和设备评测，不能只靠输入一段文字就保证量产效果。

如果只想先确认 ESP32-S3 的麦克风和固件链路，可直接使用 ESPHome 官方英文
模型；如果有 NVIDIA GPU 且希望用图形界面快速训练，可先试第三方 Docker
trainer；如果固件是 ESP-SR/小智，则必须使用 WakeNet `.bin`，不能加载本项目的
`.tflite`。

## 可选方案

| 方案 | 自定义词 | 中文 | 产物/固件 | 开箱程度 | 建议 |
|---|---:|---:|---|---:|---|
| ESPHome 官方 microWakeWord 模型 | 否 | 暂无官方中文模型 | `.tflite` + JSON / ESPHome | 最高 | 先验证硬件 |
| 本项目 microWakeWord 训练链 | 是 | 是 | `.tflite` + JSON / ESPHome | 中 | 开源自定义词首选 |
| TaterTotterson NVIDIA Docker trainer | 是 | 取决于 Piper 语音 | `.tflite` + JSON / ESPHome | 高 | 有本地 NVIDIA GPU 时快速试用 |
| 乐鑫 WakeNet9 定制 | 是 | 是 | `wn9_*.bin` / ESP-SR、部分小智固件 | 商业服务 | 量产 ESP-SR 产品 |

### 1. 立即验证硬件：ESPHome 官方模型

官方 v2 模型库当前提供 `okay_nabu`、`hey_jarvis`、`hey_mycroft` 和 `alexa`，
以及 VAD 模型。无需下载到仓库，ESPHome 会解析模型清单：

```yaml
micro_wake_word:
  microphone: wake_mic
  vad:
  models:
    - model: okay_nabu
  on_wake_word_detected:
    - logger.log:
        format: "wake word detected: %s"
        args: ['wake_word.c_str()']
```

这只能验证音频输入、内存和推理链路，不能把 `okay_nabu` 改名后当成自定义词。

### 2. 任意中文词：使用本项目

仓库已包含“星星”示例。建议实际使用 3～6 个音节的短语，例如“你好星星”；
“星星”这类两音节常用词更容易被普通对话误触发。

```bash
python scripts/init_wake.py \
  --name my_wake \
  --phrases "你好星星" \
  --language zh

python scripts/synth_positives.py --project my_wake
python scripts/synth_hard_negatives.py --project my_wake
python scripts/download_hf_negatives.py --out data/negative_datasets
python scripts/build_features.py --project my_wake --download-aug-corpora
python scripts/train_microwakeword.py --project my_wake
```

训练后必须加入目标设备麦克风录制的独立正样本和长时负样本，再选择阈值：

```bash
python scripts/seed_eval_tasks.py \
  --project my_wake \
  --bulk-audio-dir data/raw/negatives \
  --bulk-stream-minutes 60

python -m eval.runner \
  --project my_wake \
  --model models/my_wake-wakeword-v0.tflite \
  --out eval/results/my_wake-v0__latest.json

python scripts/emit_manifest.py \
  --project my_wake \
  --eval-json eval/results/my_wake-v0__latest.json
```

把 JSON 和它引用的 `.tflite` 发布在同一目录，然后在 ESPHome 中引用 **JSON
清单地址**，不是直接写模型 URL：

```yaml
micro_wake_word:
  microphone: wake_mic
  vad:
  models:
    - model: https://example.com/my_wake/manifest.json
```

### 3. 更快的第三方训练器

[`TaterTotterson/microWakeWord-Trainer-Nvidia-Docker`](https://github.com/TaterTotterson/microWakeWord-Trainer-Nvidia-Docker)
提供预构建 NVIDIA/CUDA 镜像、Web UI、Piper 合成、真实录音/误触样本导入，以及
ESPHome `.tflite` + JSON 输出。固定版本比追踪 `latest` 更稳妥：

```bash
docker pull ghcr.io/tatertotterson/microwakeword:v11
docker run -d \
  --gpus all \
  --network host \
  -e REC_PORT=8789 \
  -v "$(pwd):/data" \
  ghcr.io/tatertotterson/microwakeword:v11
```

打开 `http://localhost:8789`。这是第三方项目，使用前仍需核验数据许可证、中文
Piper 发音、评测集隔离和模型清单内容。本项目更适合需要明确数据来源、云训练和
可重复评测的场景。

### 4. ESP-SR / 小智固件：WakeNet9

WakeNet9/9l 原生支持 ESP32-S3，单模型最多 5 个唤醒词。官方开放的“Hi 乐鑫”等
模型可直接使用；自定义词由乐鑫提供服务。官方要求客户自备方案通常需要 2 万条
以上合格语料，训练调优约 2～3 周，费用和量产规模需联系销售。

WakeNet 产物是专有 `wn9_*.bin`。microWakeWord 的 `.tflite` 与它的前处理、模型
加载器和固件接口都不同，不能互换。若目标是 `78/xiaozhi-esp32` 一类 ESP-SR
固件，应选择 WakeNet 定制或修改固件集成 ESPHome/microWakeWord 运行时。

## 不建议直接用于 ESP32-S3 的方案

- `openWakeWord` 更适合 Raspberry Pi/Linux；其官方说明中也提到量化模型在
  ESP32-S3 上处理单个 80 ms 帧可能需要数秒，无法满足持续实时监听。
- 普通音频分类 TFLite 即使写着“ESP32”，如果输入特征不是 ESPHome 的 40 维
  MicroFrontend 流，或者缺少 v2 JSON 清单，也不能直接放进 `micro_wake_word`。
- 来源不明的社区 `.tflite` 不应仅凭文件尺寸判断兼容性；至少检查 INT8 输入输出、
  streaming state、10 ms feature step、tensor arena 和独立 FAR/FRR 结果。

## 官方资料

- [ESPHome micro_wake_word 配置与 v2 清单](https://esphome.io/components/micro_wake_word/)
- [ESPHome 官方模型库](https://github.com/esphome/micro-wake-word-models/tree/main/models/v2)
- [OHF microWakeWord 训练框架](https://github.com/OHF-Voice/micro-wake-word)
- [乐鑫 WakeNet9 文档](https://docs.espressif.com/projects/esp-sr/zh_CN/latest/esp32s3/wake_word_engine/README.html)
- [乐鑫唤醒词定制流程](https://docs.espressif.com/projects/esp-sr/zh_CN/latest/esp32s3/wake_word_engine/ESP_Wake_Words_Customization.html)
