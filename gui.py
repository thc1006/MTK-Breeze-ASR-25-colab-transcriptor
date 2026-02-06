#!/usr/bin/env python3
"""
Graphical user interface for Audio Enhancer.

Usage:
    python gui.py
    audio-enhance-gui
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config import EnhanceConfig
from src.enhancer import AudioEnhancer


class AudioEnhancerGUI:
    """Graphical user interface for Audio Enhancer."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("GPU Audio Enhancer")
        self.root.geometry("600x700")
        self.root.resizable(True, True)
        
        # Variables
        self.input_files: List[Path] = []
        self.output_dir: Optional[Path] = None
        self.processing = False
        
        # Config variables
        self.target_loudness = tk.DoubleVar(value=-14.0)
        self.enable_compression = tk.BooleanVar(value=True)
        self.compression_threshold = tk.DoubleVar(value=-20.0)
        self.compression_ratio = tk.DoubleVar(value=4.0)
        self.enable_denoise = tk.BooleanVar(value=True)
        self.denoise_strength = tk.DoubleVar(value=0.75)
        self.enable_eq = tk.BooleanVar(value=True)
        self.bass_boost = tk.DoubleVar(value=2.0)
        self.mid_boost = tk.DoubleVar(value=2.0)
        self.treble_boost = tk.DoubleVar(value=1.0)
        self.output_format = tk.StringVar(value="wav")
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create GUI widgets."""
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== File Selection =====
        file_frame = ttk.LabelFrame(main_frame, text="檔案選擇", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="選擇檔案", command=self._select_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="選擇資料夾", command=self._select_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="清除", command=self._clear_files).pack(side=tk.LEFT, padx=5)
        
        self.file_listbox = tk.Listbox(file_frame, height=4)
        self.file_listbox.pack(fill=tk.X, pady=5)
        
        # Output directory
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X)
        ttk.Label(output_frame, text="輸出目錄:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(output_frame, text="未選擇", foreground="gray")
        self.output_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(output_frame, text="選擇", command=self._select_output).pack(side=tk.RIGHT)
        
        # ===== Loudness Settings =====
        loud_frame = ttk.LabelFrame(main_frame, text="響度設定", padding="5")
        loud_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(loud_frame, text="目標響度 (LUFS):").pack(anchor=tk.W)
        loud_slider_frame = ttk.Frame(loud_frame)
        loud_slider_frame.pack(fill=tk.X)
        ttk.Scale(
            loud_slider_frame, from_=-30, to=-5, 
            variable=self.target_loudness, orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(loud_slider_frame, textvariable=self.target_loudness, width=6).pack(side=tk.RIGHT)
        
        # ===== Compression Settings =====
        comp_frame = ttk.LabelFrame(main_frame, text="動態壓縮", padding="5")
        comp_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            comp_frame, text="啟用壓縮", 
            variable=self.enable_compression
        ).pack(anchor=tk.W)
        
        comp_settings = ttk.Frame(comp_frame)
        comp_settings.pack(fill=tk.X)
        
        ttk.Label(comp_settings, text="閾值 (dB):").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(
            comp_settings, from_=-40, to=-5,
            variable=self.compression_threshold, orient=tk.HORIZONTAL
        ).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(comp_settings, textvariable=self.compression_threshold, width=6).grid(row=0, column=2)
        
        ttk.Label(comp_settings, text="壓縮比:").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(
            comp_settings, from_=1, to=10,
            variable=self.compression_ratio, orient=tk.HORIZONTAL
        ).grid(row=1, column=1, sticky=tk.EW)
        ttk.Label(comp_settings, textvariable=self.compression_ratio, width=6).grid(row=1, column=2)
        
        comp_settings.columnconfigure(1, weight=1)
        
        # ===== Denoise Settings =====
        denoise_frame = ttk.LabelFrame(main_frame, text="降噪", padding="5")
        denoise_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            denoise_frame, text="啟用降噪",
            variable=self.enable_denoise
        ).pack(anchor=tk.W)
        
        denoise_settings = ttk.Frame(denoise_frame)
        denoise_settings.pack(fill=tk.X)
        ttk.Label(denoise_settings, text="強度 (0-1):").pack(side=tk.LEFT)
        ttk.Scale(
            denoise_settings, from_=0, to=1,
            variable=self.denoise_strength, orient=tk.HORIZONTAL
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(denoise_settings, textvariable=self.denoise_strength, width=6).pack(side=tk.RIGHT)
        
        # ===== EQ Settings =====
        eq_frame = ttk.LabelFrame(main_frame, text="EQ 頻譜增強", padding="5")
        eq_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(
            eq_frame, text="啟用 EQ",
            variable=self.enable_eq
        ).pack(anchor=tk.W)
        
        eq_settings = ttk.Frame(eq_frame)
        eq_settings.pack(fill=tk.X)
        
        ttk.Label(eq_settings, text="低頻 (dB):").grid(row=0, column=0, sticky=tk.W)
        ttk.Scale(
            eq_settings, from_=-6, to=6,
            variable=self.bass_boost, orient=tk.HORIZONTAL
        ).grid(row=0, column=1, sticky=tk.EW)
        ttk.Label(eq_settings, textvariable=self.bass_boost, width=6).grid(row=0, column=2)
        
        ttk.Label(eq_settings, text="中頻 (dB):").grid(row=1, column=0, sticky=tk.W)
        ttk.Scale(
            eq_settings, from_=-6, to=6,
            variable=self.mid_boost, orient=tk.HORIZONTAL
        ).grid(row=1, column=1, sticky=tk.EW)
        ttk.Label(eq_settings, textvariable=self.mid_boost, width=6).grid(row=1, column=2)
        
        ttk.Label(eq_settings, text="高頻 (dB):").grid(row=2, column=0, sticky=tk.W)
        ttk.Scale(
            eq_settings, from_=-6, to=6,
            variable=self.treble_boost, orient=tk.HORIZONTAL
        ).grid(row=2, column=1, sticky=tk.EW)
        ttk.Label(eq_settings, textvariable=self.treble_boost, width=6).grid(row=2, column=2)
        
        eq_settings.columnconfigure(1, weight=1)
        
        # ===== Output Format =====
        format_frame = ttk.LabelFrame(main_frame, text="輸出格式", padding="5")
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        format_options = ttk.Frame(format_frame)
        format_options.pack()
        for fmt in ["wav", "mp3", "flac"]:
            ttk.Radiobutton(
                format_options, text=fmt.upper(),
                variable=self.output_format, value=fmt
            ).pack(side=tk.LEFT, padx=10)
        
        # ===== Progress =====
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 10))
        
        self.status_label = ttk.Label(main_frame, text="就緒", foreground="gray")
        self.status_label.pack()
        
        # ===== Process Button =====
        self.process_btn = ttk.Button(
            main_frame, text="開始處理",
            command=self._start_processing
        )
        self.process_btn.pack(pady=10)
        
    def _select_files(self):
        """Open file dialog to select audio files."""
        filetypes = [
            ("Audio files", "*.wav *.mp3 *.flac *.m4a *.ogg *.aac"),
            ("All files", "*.*")
        ]
        files = filedialog.askopenfilenames(filetypes=filetypes)
        if files:
            self.input_files = [Path(f) for f in files]
            self._update_file_list()
    
    def _select_folder(self):
        """Open dialog to select a folder."""
        folder = filedialog.askdirectory()
        if folder:
            folder_path = Path(folder)
            supported = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
            self.input_files = [
                f for f in folder_path.iterdir()
                if f.suffix.lower() in supported
            ]
            self._update_file_list()
    
    def _clear_files(self):
        """Clear selected files."""
        self.input_files = []
        self._update_file_list()
    
    def _update_file_list(self):
        """Update the file listbox."""
        self.file_listbox.delete(0, tk.END)
        for f in self.input_files:
            self.file_listbox.insert(tk.END, f.name)
        
        if self.input_files:
            self.status_label.config(text=f"已選擇 {len(self.input_files)} 個檔案")
    
    def _select_output(self):
        """Select output directory."""
        folder = filedialog.askdirectory()
        if folder:
            self.output_dir = Path(folder)
            self.output_label.config(text=str(self.output_dir), foreground="black")
    
    def _build_config(self) -> EnhanceConfig:
        """Build configuration from GUI settings."""
        return EnhanceConfig(
            target_loudness=self.target_loudness.get(),
            enable_compression=self.enable_compression.get(),
            compression_threshold=self.compression_threshold.get(),
            compression_ratio=self.compression_ratio.get(),
            enable_denoise=self.enable_denoise.get(),
            denoise_strength=self.denoise_strength.get(),
            enable_eq=self.enable_eq.get(),
            bass_boost=self.bass_boost.get(),
            mid_boost=self.mid_boost.get(),
            treble_boost=self.treble_boost.get(),
            output_format=self.output_format.get()
        )
    
    def _start_processing(self):
        """Start processing in a separate thread."""
        if not self.input_files:
            messagebox.showwarning("警告", "請先選擇音頻檔案")
            return
        
        if not self.output_dir:
            messagebox.showwarning("警告", "請選擇輸出目錄")
            return
        
        if self.processing:
            return
        
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        
        # Run in thread
        thread = threading.Thread(target=self._process_files)
        thread.daemon = True
        thread.start()
    
    def _process_files(self):
        """Process files (runs in separate thread)."""
        try:
            config = self._build_config()
            enhancer = AudioEnhancer(config)
            
            total = len(self.input_files)
            success = 0
            
            for i, input_path in enumerate(self.input_files):
                # Update progress
                progress = (i / total) * 100
                self.progress_var.set(progress)
                self.status_label.config(text=f"處理中: {input_path.name}")
                self.root.update_idletasks()
                
                # Generate output path
                output_path = self.output_dir / f"{input_path.stem}_enhanced.{config.output_format}"
                
                try:
                    stats = enhancer.process(input_path, output_path)
                    success += 1
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")
            
            # Complete
            self.progress_var.set(100)
            self.status_label.config(text=f"完成！成功處理 {success}/{total} 個檔案")
            
            messagebox.showinfo(
                "完成",
                f"成功處理 {success}/{total} 個檔案\n\n輸出目錄: {self.output_dir}"
            )
            
        except Exception as e:
            messagebox.showerror("錯誤", str(e))
            
        finally:
            self.processing = False
            self.process_btn.config(state=tk.NORMAL)
    
    def run(self):
        """Run the GUI application."""
        self.root.mainloop()


def main():
    """Main entry point for GUI."""
    app = AudioEnhancerGUI()
    app.run()


if __name__ == "__main__":
    main()
