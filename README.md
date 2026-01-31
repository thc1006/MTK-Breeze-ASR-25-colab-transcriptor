# 台灣在地化的中文語音轉文字轉錄器

MediaTek Breeze-ASR-25 + 時間戳記對齊 台灣本土中/英文語音轉文字轉錄器 Colab 實作

> Google Colab 版本：https://colab.research.google.com/drive/1RgRKhBo9vBAQ3ZUqt4APBfsT-u1ECB18

- **GPU 加速** - 使用 CUDA 進行高速處理
- **智能響度正規化** - LUFS 標準化（支援串流平台標準）
- **動態範圍壓縮** - 讓大小聲更均衡
- **AI 降噪** - 移除背景雜音
- **頻譜增強 (EQ)** - 可調整低/中/高頻
- **批次處理** - 一次處理整個資料夾
- **CLI + GUI** - 命令行與圖形介面雙支援

## 安裝

### 系統需求

- Python 3.8+
- NVIDIA GPU with CUDA support (建議 4GB+ VRAM)
- FFmpeg

### 安裝步驟

```bash
# 1. Clone repo
git clone https://github.com/yourusername/audio-enhancer-gpu.git
cd audio-enhancer-gpu

# 2. 建立虛擬環境（建議）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安裝 PyTorch (根據你的 CUDA 版本)
# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 4. 安裝其他依賴
pip install -r requirements.txt

# 5. 安裝套件
pip install -e .
```

### 安裝 FFmpeg

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (使用 Chocolatey)
choco install ffmpeg
```

## 使用方式

### CLI 命令行

```bash
# 處理單一檔案
audio-enhance input.wav -o output.wav

# 處理整個資料夾
audio-enhance ./recordings/ -o ./enhanced/

# 自訂參數
audio-enhance input.wav -o output.wav \
    --loudness -14 \
    --denoise 0.8 \
    --compress \
    --eq-mid 3.0

# 查看所有選項
audio-enhance --help
```

### Python API

```python
from src.enhancer import AudioEnhancer
from src.config import EnhanceConfig

# 建立配置
config = EnhanceConfig(
    target_loudness=-14.0,
    enable_denoise=True,
    denoise_strength=0.75,
    enable_compression=True,
    enable_eq=True,
    mid_boost=2.0
)

# 初始化增強器
enhancer = AudioEnhancer(config)

# 處理單一檔案
stats = enhancer.process("input.wav", "output.wav")
print(f"增益: {stats['gain']:.1f} dB")

# 批次處理
results = enhancer.process_batch(
    input_dir="./recordings",
    output_dir="./enhanced"
)
```

### GUI 圖形介面

```bash
# 啟動 GUI
audio-enhance-gui
# 或
python gui.py
```

## 參數說明

| 參數 | 說明 | 預設值 | 範圍 |
|------|------|--------|------|
| `--loudness` | 目標響度 (LUFS) | -14.0 | -30 ~ -5 |
| `--denoise` | 降噪強度 | 0.75 | 0 ~ 1 |
| `--compress` | 啟用動態壓縮 | True | - |
| `--threshold` | 壓縮閾值 (dB) | -20 | -40 ~ -5 |
| `--ratio` | 壓縮比 | 4.0 | 1 ~ 10 |
| `--eq-bass` | 低頻增益 (dB) | 2.0 | -6 ~ 6 |
| `--eq-mid` | 中頻增益 (dB) | 2.0 | -6 ~ 6 |
| `--eq-treble` | 高頻增益 (dB) | 1.0 | -6 ~ 6 |
| `--format` | 輸出格式 | wav | wav/mp3/flac |

### 預設設定檔

可以修改 `config/default.yaml` 來更改預設參數：

```yaml
loudness:
  target: -14.0
  peak_ceiling: -1.0

compression:
  enabled: true
  threshold: -20.0
  ratio: 4.0

denoise:
  enabled: true
  strength: 0.75

eq:
  enabled: true
  bass: 2.0
  mid: 2.0
  treble: 1.0

output:
  format: wav
  sample_rate: 44100
```

## 進階用法

### 使用設定檔

```bash
# 使用自訂設定檔
audio-enhance input.wav -o output.wav --config my_config.yaml
```

### Docker 使用

```bash
# 建構 image
docker build -t audio-enhancer .

# 執行（需要 NVIDIA Docker）
docker run --gpus all -v $(pwd)/audio:/data audio-enhancer \
    /data/input.wav -o /data/output.wav
```

## 支援格式

**輸入**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`, `.wma`

**輸出**: `.wav`, `.mp3`, `.flac`

## 常見問題

### CUDA 不可用

```bash
# 檢查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

如果返回 `False`，請確認：
1. 已安裝 NVIDIA 驅動
2. PyTorch 版本與 CUDA 版本匹配

### 記憶體不足

處理超長音檔時可能出現 OOM，可以：
```bash
# 使用分段處理模式
audio-enhance input.wav -o output.wav --chunk-mode
```

## License

MIT License

## Contributing

歡迎提交 Issue 和 Pull Request！
