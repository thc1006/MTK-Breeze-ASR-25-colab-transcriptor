# 台灣在地化中文語音轉文字轉錄器

使用 MediaTek Breeze-ASR-25 官方模型，支援時間戳記對齊，可在 RTX 3050 4GB 等低 VRAM 環境下執行。

## 功能

- 語音轉文字：使用 MediaTek-Research/Breeze-ASR-25 官方模型
- 時間戳記：每段文字標註起始與結束時間
- SRT 字幕：標準字幕格式輸出
- 自適應 GPU 配置：自動偵測 VRAM 選擇最佳設定
- 音頻增強：降噪、響度正規化

## 系統需求

- Python 3.10+
- NVIDIA GPU (建議 4GB+ VRAM) 或 CPU
- CUDA 12.x (GPU 模式)

## 安裝

```bash
# 1. Clone
git clone https://github.com/yourusername/MTK-Breeze-ASR-25-colab-transcriptor.git
cd MTK-Breeze-ASR-25-colab-transcriptor

# 2. 安裝 PyTorch (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. 安裝依賴
pip install -r requirements.txt
```

首次執行會下載模型（約 3GB），之後會快取。

## 使用方式

### 語音轉文字

```bash
# 自動偵測 GPU，選擇最佳配置
python transcribe_cli.py audio.m4a --mode auto

# 速度優先（NF4 量化 + beam=1，低 VRAM 適用）
python transcribe_cli.py audio.m4a --mode speed

# 品質優先（float16 + beam=3）
python transcribe_cli.py audio.m4a --mode quality

# 輸出 SRT 字幕
python transcribe_cli.py audio.m4a --srt

# 指定輸出路徑
python transcribe_cli.py audio.m4a -o result.txt

# 英文轉錄
python transcribe_cli.py audio.mp3 --lang en

# 強制 CPU
python transcribe_cli.py audio.m4a --cpu
```

### 音頻增強

```bash
python cli.py input.wav -o output.wav
```

## 輸出格式

### TXT（時間戳記）
```
[00:00:01.25 - 00:00:07.25] 第一個是我們剛剛講到的那個無人機...
```

### SRT（字幕）
```srt
1
00:00:01,250 --> 00:00:07,250
第一個是我們剛剛講到的那個無人機...
```

## 效能參考（RTX 3050 Laptop 4GB）

以下為 6 分鐘台灣華語音檔的實測數據，實際速度依硬體和音檔內容而異。

| 模式 | 量化 | 6 分鐘音檔耗時 |
|------|------|----------------|
| speed (beam=1) | NF4 4-bit | 約 4-5 分鐘 |
| quality (beam=3) | float16 | 較慢 |
| auto | 依 GPU 決定 | 依 GPU 決定 |

VRAM 占用：NF4 量化約 930MB，可在 4GB GPU 上穩定執行。

## 自適應 GPU 分級

系統會自動偵測 GPU 的 VRAM 和 compute capability，從內建分級表選擇配置：

| 分級 | VRAM | 量化 | beam |
|------|------|------|------|
| low | 4GB 以下 | NF4 4-bit | 1 |
| mid_low | 5-6GB | INT8 | 2 |
| mid | 7-8GB | 無 | 3 |
| mid_high+ | 9GB 以上 | 無 | 5 |

量化載入失敗時會自動降級：NF4 -> INT8 -> float16 -> CPU。

## 專案結構

```
.
├── transcribe_cli.py  # 語音轉文字 CLI
├── cli.py             # 音頻增強 CLI
├── gui.py             # 音頻增強 GUI
├── src/
│   ├── transcriber.py # Breeze-ASR-25 轉錄器（HF Transformers + Silero-VAD）
│   ├── enhancer.py    # 音頻增強器
│   └── config.py      # 音頻增強配置
└── config/
    └── default.yaml   # 音頻增強預設配置
```

## 技術細節

- 模型：MediaTek-Research/Breeze-ASR-25（官方 HuggingFace 權重）
- 後端：HuggingFace Transformers pipeline
- VAD：Silero-VAD（語音區段偵測，跳過靜音段）
- 量化：bitsandbytes NF4/INT8（低 VRAM GPU 用）
- 支援格式：.wav, .mp3, .m4a, .flac, .ogg, .aac

## 注意事項

- Windows 終端機可能顯示亂碼（cp950 編碼），輸出檔案是正確的 UTF-8
- bitsandbytes 在部分 Windows 環境可能不支援，會自動降級為 float16
- 長音檔（超過 30 分鐘）建議使用 speed 模式

## Colab 版本

沒有本地 GPU 可使用 Google Colab：
https://colab.research.google.com/drive/1RgRKhBo9vBAQ3ZUqt4APBzsT-u1ECB18

## License

MIT License
