# MTK Breeze-ASR-25 台灣中文語音轉文字轉錄器

## 專案狀態：可用

本專案使用 HuggingFace Transformers 載入官方 MediaTek-Research/Breeze-ASR-25 模型，
搭配 Silero-VAD 做語音區段偵測，支援 bitsandbytes 量化降級。
自適應 GPU 優化：自動偵測 VRAM 和 compute capability，選擇最佳配置。

---

## 快速開始

### 轉錄音檔（最常用）

```bash
# 自動偵測 GPU，選擇最佳配置（推薦）
python transcribe_cli.py audio.m4a --mode auto

# 速度優先模式（NF4 量化 + beam=1，適合低 VRAM）
python transcribe_cli.py audio.m4a

# 品質優先模式
python transcribe_cli.py audio.m4a --mode quality

# 同時輸出 SRT 字幕
python transcribe_cli.py audio.m4a --srt

# 指定輸出檔案
python transcribe_cli.py audio.m4a -o output.txt

# 英文轉錄
python transcribe_cli.py audio.mp3 --lang en

# 手動指定 pipeline 分段長度（秒），0=不分段
python transcribe_cli.py audio.m4a --mode auto --chunk-length 0
```

### 音頻增強（降噪/響度調整）

```bash
python cli.py input.wav -o output.wav
```

---

## 專案結構

```
MTK-Breeze-ASR-25-colab-transcriptor/
├── transcribe_cli.py     # 語音轉文字 CLI（主要工具）
├── cli.py                # 音頻增強 CLI
├── gui.py                # 音頻增強 GUI
├── src/
│   ├── transcriber.py    # Breeze-ASR-25 轉錄器（HF Transformers + Silero-VAD）
│   ├── enhancer.py       # 音頻增強器
│   └── config.py         # 音頻增強配置
├── config/
│   └── default.yaml      # 音頻增強預設配置
├── requirements.txt      # Python 依賴
├── 01312.m4a            # 測試音檔
└── *.ipynb              # Google Colab 版本（參考用）
```

---

## 自適應 GPU 分級系統

系統自動偵測 GPU 的 VRAM 和 compute capability (CC)，從 `_VRAM_TIERS` 查表選擇最佳配置：

| Tier | VRAM | dtype | quant_mode | beam | chunk_s | 範例 GPU |
|------|------|-------|-----------|------|---------|---------|
| `low` | <= 4GB | float16 | bnb_4bit | 1 | 30 | RTX 3050 Laptop 4GB |
| `mid_low` | 5-6GB | float16 | bnb_8bit | 2 | 30 | RTX 4050 6GB |
| `mid` | 7-8GB | float16 | none | 3 | 30 | RTX 4060 8GB |
| `mid_high` | 9-11GB | float16 | none | 5 | 0 | RTX 3080 10GB |
| `high` | 12-24GB | bfloat16* | none | 5 | 0 | RTX 3060 12GB, RTX 4080 16GB |
| `very_high` | 25-48GB | bfloat16* | none | 5 | 0 | A100 40GB, L40S 48GB |
| `ultra` | > 48GB | bfloat16* | none | 5 | 0 | A100 80GB, H100 80GB |

*CC >= 8.0 (Ampere+) 且 high 以上層級時自動升級為 bfloat16

- `chunk_s=30`: pipeline 以 30 秒分段處理（低 VRAM 情境）
- `chunk_s=0`: pipeline 整段處理（高 VRAM，完整上下文傳遞）

### 量化降級鏈

每次載入模型時，若指定的量化模式失敗，會自動嘗試下一級：

```
bnb_4bit -> bnb_8bit -> float16 + device_map="auto" -> 純 CPU float32
```

---

## 硬體配置（開發環境）

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX 3050 Laptop |
| VRAM | 4GB |
| CC | 8.6 (Ampere) |
| 自動選擇 | low tier: float16 + bnb_4bit, beam=1, chunk=30s |

---

## 技術細節

### 模型
- **Breeze-ASR-25**: MediaTek 台灣華語優化模型
- **來源**: `MediaTek-Research/Breeze-ASR-25` (官方 HuggingFace 權重)
- **後端**: HuggingFace Transformers + `pipeline("automatic-speech-recognition")`
- **VAD**: Silero-VAD (外掛預處理，跳過靜音段)
- **量化**: bitsandbytes NF4/INT8 (低 VRAM GPU)

### 配置類別

- **`GPUConfig`**: 自適應配置（`--mode auto` 時由 `_auto_config()` 產生）
- **`RTX3050Config`**: 速度優先固定配置（`--mode speed`，NF4 量化）
- **`QualityConfig`**: 品質優先固定配置（`--mode quality`，float16）

---

## 常用指令

### 安裝依賴
```bash
pip install -r requirements.txt
```

### 轉錄指令選項
```bash
python transcribe_cli.py --help

# 主要參數：
#   --mode {speed,quality,auto}  優化模式（auto 會自動偵測 GPU）
#   --chunk-length SECS          Pipeline 分段長度（秒），0=不分段
#   --srt                        輸出 SRT 字幕
#   --lang {zh,en}              語言
#   -o OUTPUT                   輸出路徑
#   --cpu                       強制 CPU
```

---

## 輸出格式

### TXT（時間戳記）
```
[00:00:01.25 - 00:00:07.25] 第一個是我們剛剛講到的那個無人機...
[00:00:07.25 - 00:00:12.78] 然後第二個可能就是我之前有提到...
```

### SRT（字幕）
```srt
1
00:00:01,250 --> 00:00:07,250
第一個是我們剛剛講到的那個無人機...

2
00:00:07,250 --> 00:00:12,780
然後第二個可能就是我之前有提到...
```

---

## 注意事項

1. **首次執行會下載模型**（約 3GB），之後會快取
2. **Windows 終端機顯示亂碼是正常的**（cp950 編碼問題），但輸出檔案是正確的 UTF-8
3. **支援格式**: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`, `.aac`
4. **建議**：長音檔（>30 分鐘）建議使用 `--mode speed`
5. **量化注意**：bitsandbytes 在 Windows 上可能不支援，會自動降級為 float16

---

## 開發備註

- 不要使用 emoji（Windows cp950 編碼問題）
- 程式碼風格：繁體中文註解
- 測試檔案：`01312.m4a`（約 6 分鐘台灣華語對話）

---

## 相關連結

- [Breeze-ASR-25 HuggingFace (官方)](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [Silero-VAD](https://github.com/snakers4/silero-vad)
- [Google Colab 版本](https://colab.research.google.com/drive/1RgRKhBo9vBAQ3ZUqt4APBfsT-u1ECB18)
