# MTK Breeze-ASR-25 台灣中文語音轉文字轉錄器

## 專案狀態：可用

本專案支援自適應 GPU 優化：自動偵測 VRAM 和 compute capability，選擇最佳配置。
涵蓋從 RTX 3050 4GB 到 H100 80GB+ 的所有 NVIDIA GPU。

---

## 快速開始

### 轉錄音檔（最常用）

```bash
# 自動偵測 GPU，選擇最佳配置（推薦）
python transcribe_cli.py audio.m4a --mode auto

# 速度優先模式（固定 beam=1，適合低 VRAM）
python transcribe_cli.py audio.m4a

# 品質優先模式
python transcribe_cli.py audio.m4a --mode quality

# 同時輸出 SRT 字幕
python transcribe_cli.py audio.m4a --srt

# 指定輸出檔案
python transcribe_cli.py audio.m4a -o output.txt

# 英文轉錄
python transcribe_cli.py audio.mp3 --lang en

# 手動指定 batch size（需要足夠 VRAM）
python transcribe_cli.py audio.m4a --mode auto --batch 8
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
│   ├── transcriber.py    # Breeze-ASR-25 轉錄器（自適應 GPU 優化）
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

| Tier | VRAM | compute_type | beam | batch | 範例 GPU |
|------|------|-------------|------|-------|---------|
| `low` | <= 4GB | int8_float16 | 1 | - | RTX 3050 Laptop 4GB |
| `mid_low` | 5-6GB | int8_float16 | 3 | - | RTX 4050 6GB |
| `mid` | 7-8GB | int8_float16 | 5 | - | RTX 3070 8GB, RTX 4060 8GB |
| `mid_high` | 9-11GB | float16 | 5 | - | RTX 3080 10GB |
| `high` | 12-24GB | bfloat16* | 5 | 12 | RTX 3060 12GB, RTX 4080 16GB |
| `very_high` | 25-48GB | bfloat16* | 5 | 24 | A100 40GB, L40S 48GB |
| `ultra` | > 48GB | bfloat16* | 5 | 32 | A100 80GB, H100 80GB |

*CC >= 8.0 (Ampere+) 時自動升級為 bfloat16，否則維持 float16

**BatchedInferencePipeline**: high 以上層級自動啟用 faster-whisper 批次推論，可達 3x 額外加速。

---

## 硬體配置（開發環境）

| 項目 | 規格 |
|------|------|
| GPU | NVIDIA RTX 3050 Laptop |
| VRAM | 4GB |
| CC | 8.6 (Ampere) |
| 自動選擇 | low tier: int8_float16, beam=1 |
| 實測速度 | ~40s 完成 6 分鐘音檔 |

---

## 技術細節

### 模型
- **Breeze-ASR-25**: MediaTek 台灣華語優化模型
- **來源**: `SoybeanMilk/faster-whisper-Breeze-ASR-25`
- **基於**: Whisper large-v2

### 配置類別

- **`GPUConfig`**: 自適應配置（`--mode auto` 時由 `_auto_config()` 產生）
- **`RTX3050Config`**: 速度優先固定配置（`--mode speed`）
- **`QualityConfig`**: 品質優先固定配置（`--mode quality`）

### 效能數據（RTX 3050 4GB）

| 模式 | 時間 | 加速比 |
|------|------|--------|
| speed (beam=1) | ~40s | 2.9x |
| quality (beam=3) | ~80s | 1.9x |
| 未優化 (beam=5) | 150.4s | 1x |

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
#   --batch BATCH                批次推論 batch size（需要足夠 VRAM）
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

---

## 開發備註

- 不要使用 emoji（Windows cp950 編碼問題）
- 程式碼風格：繁體中文註解
- 測試檔案：`01312.m4a`（約 6 分鐘台灣華語對話）

---

## 相關連結

- [Breeze-ASR-25 HuggingFace](https://huggingface.co/MediaTek-Research/Breeze-ASR-25)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Google Colab 版本](https://colab.research.google.com/drive/1RgRKhBo9vBAQ3ZUqt4APBfsT-u1ECB18)
