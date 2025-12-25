"""
人声音频分离studio - 专业音频分离工作站 (Fixed & Optimized)
整合 Demucs大模型，支持多轨道编辑、播放控制、自动保存分轨

依赖: pip install demucs torch torchaudio numpy scipy matplotlib pyaudio
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import os
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- 尝试导入大模型 ---
try:
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    AI_AVAILABLE = True
    print("Demucs大模型支持已就绪")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠ 模型库未安装，将使用基础频段分离算法: {e}")

# --- 尝试导入音频播放 ---
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("⚠ PyAudio 未安装，播放功能不可用。请安装: pip install pyaudio")

# --- Matplotlib 字体与样式配置 ---
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# --- 现代 DAW 配色方案 ---
COLORS = {
    "bg": "#121212",             # 极深灰背景
    "panel": "#1e1e1e",          # 面板背景
    "panel_light": "#2a2a2a",    # 浅色面板
    "accent": "#00bcd4",         # 提亮色 (青色)
    "accent_hover": "#26c6da",   # 悬停色
    "text": "#ffffff",           # 主文本
    "text_dim": "#b0b0b0",       # 次要文本
    "border": "#333333",         # 边框
    "grid": "#2b2b2b",           # 网格线
    
    # 轨道颜色
    "vocals": "#ff5252", # 红
    "drums": "#ffb74d",  # 橙
    "bass": "#7c4dff",   # 紫
    "other": "#69f0ae",  # 绿
    
    "playhead": "#ffeb3b",       # 播放头颜色 (黄)
    "muted": "#424242",          # 静音状态
}

# 轨道定义
TRACK_CONFIG = [
    {"name": "VOCALS", "color": COLORS["vocals"], "icon": "🎤"},
    {"name": "DRUMS", "color": COLORS["drums"], "icon": "🥁"},
    {"name": "BASS", "color": COLORS["bass"], "icon": "🎸"},
    {"name": "OTHER", "color": COLORS["other"], "icon": "🎹"},
    {"name": "TRACK 5", "color": "#607d8b", "icon": "○"},
    {"name": "TRACK 6", "color": "#607d8b", "icon": "○"},
]

TRACK_HEIGHT = 70
TRACK_HEADER_WIDTH = 140
PX_PER_SEC = 60  # 时间轴缩放比例
RULER_HEIGHT = 30  # 时间标尺高度（给数字留出空间，避免被遮挡）


class AudioClip:
    """可拖拽音频片段类"""
    def __init__(self, canvas, track_idx, duration, color, name, audio_data, sample_rate, app):
        self.canvas = canvas
        self.app = app
        self.track_idx = track_idx
        self.duration = duration
        self.color = color
        self.name = name
        self.audio_data = audio_data
        self.sample_rate = sample_rate

        self.muted = False
        self.start_time = 0

        self.x = 0
        self.y = track_idx * TRACK_HEIGHT
        self.width = duration * PX_PER_SEC
        self.height = TRACK_HEIGHT - 6 

        self._draw()
        self._bind_events()

    def _draw(self):
        fill_color = COLORS["muted"] if self.muted else self.color
        
        # 背景矩形
        self.rect_id = self.canvas.create_rectangle(
            self.x, self.y + 3,
            self.x + self.width, self.y + self.height + 3,
            fill=fill_color,
            outline=COLORS["text"],
            width=1,
            tags=("clip", f"clip_{id(self)}")
        )

        # 绘制迷你波形
        self._draw_mini_waveform()

        # 文本标签
        self.text_id = self.canvas.create_text(
            self.x + 8, self.y + 14,
            text=f"{self.name}",
            anchor="nw",
            fill="#000000" if not self.muted else "#888888",
            font=("Segoe UI", 9, "bold"),
            tags=("clip", f"clip_{id(self)}")
        )

        # 静音图标
        self._draw_mute_icon()

    def _draw_mini_waveform(self):
        if self.audio_data is None: return
        
        # 简化采样
        samples = 150
        step = max(1, len(self.audio_data) // samples)
        data = self.audio_data[::step]
        if len(data.shape) > 1: data = data[:, 0]
        
        max_val = np.max(np.abs(data)) if np.max(np.abs(data)) > 0 else 1
        data = data / max_val

        center_y = self.y + 3 + self.height / 2
        points = []
        for i, val in enumerate(data[:samples]):
            px = self.x + (i / samples) * self.width
            py = center_y - val * (self.height * 0.4)
            points.extend([px, py])

        if len(points) >= 4:
            self.wave_id = self.canvas.create_line(
                points,
                fill="#333333" if not self.muted else "#555555",
                width=1,
                tags=("clip", f"clip_{id(self)}")
            )

    def _draw_mute_icon(self):
        icon_x = self.x + self.width - 20
        icon_y = self.y + 10
        icon_text = "🔇" if self.muted else "🔊"
        
        self.mute_icon = self.canvas.create_text(
            icon_x, icon_y,
            text=icon_text, anchor="nw", font=("Segoe UI", 10),
            tags=("clip", f"clip_{id(self)}", "mute_icon")
        )
        self.canvas.tag_bind(self.mute_icon, "<Button-1>", self.toggle_mute)

    def _bind_events(self):
        tag = f"clip_{id(self)}"
        self.canvas.tag_bind(tag, "<Button-1>", self.on_press)
        self.canvas.tag_bind(tag, "<B1-Motion>", self.on_drag)
        self.canvas.tag_bind(tag, "<ButtonRelease-1>", self.on_release)
        self.canvas.tag_bind(tag, "<Button-3>", self.on_right_click)

    def on_press(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.orig_x = self.x
        self.orig_track = self.track_idx
        self.canvas.tag_raise(f"clip_{id(self)}")

    def on_drag(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        new_x = max(0, self.orig_x + dx)
        new_track = max(0, min(len(TRACK_CONFIG)-1, self.orig_track + int(dy / TRACK_HEIGHT)))
        
        move_x = new_x - self.x
        move_y = (new_track * TRACK_HEIGHT) - self.y
        
        self.canvas.move(f"clip_{id(self)}", move_x, move_y)
        self.x = new_x
        self.y = new_track * TRACK_HEIGHT
        self.track_idx = new_track
        # 拖动过程中实时更新 start_time，播放时可即时响应
        self.start_time = self.x / PX_PER_SEC

    def on_release(self, event):
        grid_sec = 0.1
        grid_px = grid_sec * PX_PER_SEC
        snapped_x = round(self.x / grid_px) * grid_px
        
        move_x = snapped_x - self.x
        self.canvas.move(f"clip_{id(self)}", move_x, 0)
        self.x = snapped_x
        self.start_time = self.x / PX_PER_SEC
        
        self.app.update_status(f"片段移动至: 轨道 {self.track_idx+1}, 时间 {self.start_time:.2f}s")

    def on_right_click(self, event):
        menu = tk.Menu(self.canvas, tearoff=0, bg=COLORS["panel"], fg=COLORS["text"])
        # 根据当前状态动态显示菜单文案
        mute_label = "🔈 取消静音" if self.muted else "🔇 静音"
        menu.add_command(label=mute_label, command=lambda: self.toggle_mute(None))
        menu.add_separator()
        menu.add_command(label="🗑️ 删除", command=self.delete)
        menu.tk_popup(event.x_root, event.y_root)

    def toggle_mute(self, event):
        self.muted = not self.muted
        self.canvas.delete(f"clip_{id(self)}")
        self._draw()
        self._bind_events()
        self.app.update_status(f"{self.name} {'已静音' if self.muted else '已取消静音'}")

    def delete(self):
        self.canvas.delete(f"clip_{id(self)}")
        if self in self.app.clips:
            self.app.clips.remove(self)


class AudioPlayer:
    """音频播放器 - 逻辑修复版"""
    def __init__(self, app):
        self.app = app
        self.p = pyaudio.PyAudio() if PYAUDIO_AVAILABLE else None
        self.stream = None
        self.playing = False
        self.paused = False
        self.current_time = 0.0
        self.stop_event = threading.Event()
        self.play_thread = None

    def get_mixed_audio_chunk(self, start_time, duration):
        if not self.app.clips:
            return None, 44100

        sr = self.app.clips[0].sample_rate if self.app.clips else 44100
        num_samples = int(duration * sr)
        mixed = np.zeros((num_samples, 2), dtype=np.float32)
        start_sample_global = int(start_time * sr)
        has_audio = False
        
        for clip in self.app.clips:
            if clip.muted or clip.audio_data is None: continue
            
            clip_start_sample = int(clip.start_time * sr)
            clip_end_sample = clip_start_sample + len(clip.audio_data)
            
            overlap_start = max(start_sample_global, clip_start_sample)
            overlap_end = min(start_sample_global + num_samples, clip_end_sample)
            
            if overlap_start < overlap_end:
                has_audio = True
                buf_start = overlap_start - start_sample_global
                buf_end = overlap_end - start_sample_global
                src_start = overlap_start - clip_start_sample
                src_end = overlap_end - clip_start_sample
                
                audio_chunk = clip.audio_data[src_start:src_end]
                
                if len(audio_chunk.shape) == 1:
                    mixed[buf_start:buf_end, 0] += audio_chunk
                    mixed[buf_start:buf_end, 1] += audio_chunk
                elif audio_chunk.shape[1] == 1:
                    mixed[buf_start:buf_end, 0] += audio_chunk[:, 0]
                    mixed[buf_start:buf_end, 1] += audio_chunk[:, 0]
                else:
                    mixed[buf_start:buf_end] += audio_chunk

        if has_audio:
            peak = np.max(np.abs(mixed))
            if peak > 1.0: mixed /= peak
                
        return mixed, sr

    def play(self):
        if not self.p:
            messagebox.showerror("错误", "未检测到播放设备 (PyAudio)")
            return
        if self.playing and self.paused:
            self.paused = False
            return
        if self.playing: return

        self.playing = True
        self.paused = False
        self.stop_event.clear()
        self.play_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.play_thread.start()

    def _playback_loop(self):
        CHUNK_SIZE = 2048
        sr = self.app.clips[0].sample_rate if self.app.clips else 44100
             
        try:
            self.stream = self.p.open(
                format=pyaudio.paFloat32, channels=2, rate=sr, output=True,
                frames_per_buffer=CHUNK_SIZE
            )
            
            while not self.stop_event.is_set():
                if self.paused:
                    time.sleep(0.05)
                    continue
                
                if self.current_time >= self.app.total_duration:
                    break

                duration = CHUNK_SIZE / sr
                mixed_chunk, _ = self.get_mixed_audio_chunk(self.current_time, duration)
                
                if mixed_chunk is not None:
                    self.stream.write(mixed_chunk.astype(np.float32).tobytes())
                else:
                    self.stream.write(np.zeros((CHUNK_SIZE, 2), dtype=np.float32).tobytes())
                
                self.current_time += duration
                self.app.root.after(0, lambda t=self.current_time: self.app.update_playhead_ui(t))

        except Exception as e:
            print(f"Play Error: {e}")
        finally:
            self.playing = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            self.app.root.after(0, self.app.on_playback_stopped)

    def pause(self): self.paused = True
    def stop(self):
        self.stop_event.set()
        self.playing = False
        self.paused = False
        self.current_time = 0
        self.app.update_playhead_ui(0)
    def seek(self, t): self.current_time = t
    def cleanup(self):
        self.stop()
        if self.p: self.p.terminate()

class ModernStudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("音频分离工作站")
        self.root.geometry("1400x950")
        self.root.configure(bg=COLORS["bg"])
        
        self.file_path = ""
        self.audio_data = None
        self.sample_rate = 44100
        self.duration = 0
        self.clips = []
        self.total_duration = 60
        self.demucs_model = None
        self.scrubbing = False  # 时间轴拖动

        self.player = AudioPlayer(self)
        self._init_styles()
        self._init_ui()
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _init_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("Horizontal.TProgressbar", troughcolor=COLORS["grid"], background=COLORS["accent"], thickness=6, borderwidth=0)

    def _init_ui(self):
        self._create_toolbar()
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=2, pady=2)
        paned = tk.PanedWindow(main_container, orient="vertical", bg=COLORS["bg"], sashwidth=4, sashrelief="flat")
        paned.pack(fill="both", expand=True)
        self._create_waveform_panel(paned)
        self._create_track_panel(paned)
        self._create_transport_bar()

    def _create_toolbar(self):
        toolbar = tk.Frame(self.root, bg=COLORS["bg"], height=60, padx=15, pady=10)
        toolbar.pack(fill="x")
        
        tk.Label(toolbar, text="人声音频分离", font=("Segoe UI", 16, "bold"), bg=COLORS["bg"], fg="#ffffff").pack(side="left")
        tk.Label(toolbar, text=" studio", font=("Segoe UI", 16, "bold"), bg=COLORS["bg"], fg=COLORS["accent"]).pack(side="left")
        
        status_text = "Demus大模型已就绪" if AI_AVAILABLE else "基础模式"
        status_color = "#4caf50" if AI_AVAILABLE else "#ff9800"
        tk.Label(toolbar, text=f"  [{status_text}]", font=("Consolas", 9), bg=COLORS["bg"], fg=status_color).pack(side="left", padx=10, pady=5)

        btn_frame = tk.Frame(toolbar, bg=COLORS["bg"])
        btn_frame.pack(side="right")
        self._make_button(btn_frame, "📂 导入音频", self.load_file, bg=COLORS["panel_light"])
    def _make_button(self, parent, text, command, bg=COLORS["panel_light"]):
        btn = tk.Button(parent, text=text, command=command, bg=bg, fg=COLORS["text"], 
                        activebackground=COLORS["accent"], font=("Segoe UI", 10), relief="flat", padx=15, pady=4)
        btn.pack(side="left", padx=5)

    def _create_waveform_panel(self, parent):
        container = ttk.Frame(parent, style="Panel.TFrame")
        parent.add(container, height=230)
        
        header = tk.Frame(container, bg=COLORS["panel_light"], height=30)
        header.pack(fill="x")
        tk.Label(header, text="  📊 波形预览", bg=COLORS["panel_light"], fg=COLORS["text_dim"], font=("Segoe UI", 9, "bold")).pack(side="left", pady=5)
        
        self.fig, self.ax = plt.subplots(facecolor=COLORS["panel"])
        self.fig.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.25)
        self.canvas_wave = FigureCanvasTkAgg(self.fig, master=container)
        self.canvas_wave.get_tk_widget().pack(fill="both", expand=True)
        self._setup_ax_style(show_text=True)

    def _setup_ax_style(self, show_text=False):
        self.ax.set_facecolor(COLORS["bg"])
        self.ax.tick_params(axis='x', colors=COLORS['text_dim'])
        self.ax.tick_params(axis='y', colors=COLORS['text_dim'])
        for spine in self.ax.spines.values(): spine.set_color(COLORS['border'])
        self.ax.grid(True, color=COLORS['grid'], linestyle='--', alpha=0.5)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 10)
        if show_text:
            self.ax.text(5, 0, "导入音频以显示波形", ha='center', va='center', color=COLORS['text_dim'], fontsize=12)
        if hasattr(self, 'canvas_wave'): self.canvas_wave.draw()

    def _create_track_panel(self, parent):
        container = ttk.Frame(parent, style="Panel.TFrame")
        parent.add(container, height=450)
        
        header = tk.Frame(container, bg=COLORS["panel_light"], height=40)
        header.pack(fill="x", pady=1)
        tk.Label(header, text="  🎚️ 多轨编辑器", bg=COLORS["panel_light"], fg=COLORS["text"], font=("Segoe UI", 10, "bold")).pack(side="left", pady=8)
        
        self.btn_separate = tk.Button(header, text="开始分离", command=self.run_separation,
                                      bg=COLORS["accent"], fg="white", font=("Segoe UI", 9, "bold"),
                                      relief="flat", state="disabled")
        self.btn_separate.pack(side="right", padx=10, pady=5)

        content = tk.Frame(container, bg=COLORS["bg"])
        content.pack(fill="both", expand=True)

        # 左侧轨道区：添加与时间标尺等高的“左上角空白”，保证轨道与标尺对齐（仿 FL Studio）
        left_panel = tk.Frame(content, bg=COLORS["panel"])
        left_panel.pack(side="left", fill="y")

        self.header_ruler = tk.Canvas(left_panel, width=TRACK_HEADER_WIDTH, height=RULER_HEIGHT,
                                      bg=COLORS["panel_light"], highlightthickness=0)
        self.header_ruler.pack(fill="x")
        # 分割线，避免看起来“轨道盖住标尺”
        self.header_ruler.create_line(0, RULER_HEIGHT-1, TRACK_HEADER_WIDTH, RULER_HEIGHT-1, fill=COLORS["border"])

        self.track_headers = tk.Canvas(left_panel, width=TRACK_HEADER_WIDTH, bg=COLORS["panel"], highlightthickness=0)
        self.track_headers.pack(fill="y", expand=True)
        self._draw_track_headers()

        timeline_frame = tk.Frame(content, bg=COLORS["bg"])
        timeline_frame.pack(side="left", fill="both", expand=True)

        self.time_ruler = tk.Canvas(timeline_frame, height=RULER_HEIGHT, bg=COLORS["panel_light"], highlightthickness=0)
        self.time_ruler.pack(fill="x")
        self.timeline = tk.Canvas(timeline_frame, bg=COLORS["bg"], highlightthickness=0)
        self.timeline.pack(fill="both", expand=True)

        h_scroll = ttk.Scrollbar(timeline_frame, orient="horizontal", command=self._on_scroll)
        h_scroll.pack(fill="x")
        self.timeline.configure(xscrollcommand=h_scroll.set)
        
        self.timeline.bind("<Configure>", lambda e: self._draw_timeline())

        # 按住拖动时间轴可连续定位（scrub）
        self.timeline.bind("<ButtonPress-1>", self._on_timeline_press)
        self.timeline.bind("<B1-Motion>", self._on_timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self._on_timeline_release)

        self.time_ruler.bind("<ButtonPress-1>", self._on_timeline_press)
        self.time_ruler.bind("<B1-Motion>", self._on_timeline_drag)
        self.time_ruler.bind("<ButtonRelease-1>", self._on_timeline_release)

    def _draw_track_headers(self):
        self.track_headers.delete("all")
        for i, track in enumerate(TRACK_CONFIG):
            y = i * TRACK_HEIGHT
            self.track_headers.create_rectangle(0, y, TRACK_HEADER_WIDTH, y+TRACK_HEIGHT, fill=COLORS["panel"], outline=COLORS["border"])
            self.track_headers.create_rectangle(0, y, 6, y+TRACK_HEIGHT, fill=track["color"], outline="")
            self.track_headers.create_text(20, y+35, text=f"{track['icon']} {track['name']}", anchor="w", fill=COLORS["text"], font=("Segoe UI", 9, "bold"))
    def _draw_timeline(self):
        self.timeline.delete("grid")
        self.time_ruler.delete("all")
        w = max(self.timeline.winfo_width(), int(self.total_duration * PX_PER_SEC))
        h = len(TRACK_CONFIG) * TRACK_HEIGHT
        self.timeline.configure(scrollregion=(0, 0, w, h))

        for i in range(len(TRACK_CONFIG)):
            y = i * TRACK_HEIGHT
            bg = "#161616" if i % 2 == 0 else "#121212"
            self.timeline.create_rectangle(0, y, w, y+TRACK_HEIGHT, fill=bg, outline="", tags="grid")
            self.timeline.create_line(0, y+TRACK_HEIGHT, w, y+TRACK_HEIGHT, fill=COLORS["border"], tags="grid")

        for sec in range(int(self.total_duration) + 1):
            x = sec * PX_PER_SEC
            self.timeline.create_line(x, 0, x, h, fill=COLORS["grid"], tags="grid")
            self.time_ruler.create_line(x, RULER_HEIGHT-12, x, RULER_HEIGHT-1, fill=COLORS["text_dim"])
            self.time_ruler.create_text(x+2, 2, text=f"{sec}s", fill=COLORS["text_dim"], font=("Segoe UI", 8), anchor="n")
            sub_x = x + (PX_PER_SEC / 2)
            if sub_x < w: self.timeline.create_line(sub_x, 0, sub_x, h, fill="#1e1e1e", dash=(2,4), tags="grid")

        # 时间标尺底部边框（增强分隔感，避免“遮挡标尺”的观感）
        self.time_ruler.create_line(0, RULER_HEIGHT-1, w, RULER_HEIGHT-1, fill=COLORS["border"])

        # 确保网格在最底层，避免覆盖音频片段（修复“分离后音轨消失”）
        self.timeline.tag_lower("grid")

        self._draw_playhead_ui(self.player.current_time)

    def _create_transport_bar(self):
        bar = tk.Frame(self.root, bg=COLORS["panel"], height=80)
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)
        center_frame = tk.Frame(bar, bg=COLORS["panel"])
        center_frame.pack(expand=True)
        ctl_frame = tk.Frame(center_frame, bg=COLORS["panel"])
        ctl_frame.pack(side="left", padx=20)
        
        btn_props = {"bg": COLORS["panel"], "fg": COLORS["text"], "activebackground": COLORS["panel_light"], 
                     "bd": 0, "font": ("Segoe UI", 16)}

        tk.Button(ctl_frame, text="⏮", command=self.rewind, **btn_props).pack(side="left", padx=5)
        tk.Button(ctl_frame, text="⏹", command=self.stop, **btn_props).pack(side="left", padx=5)
        self.btn_play = tk.Button(ctl_frame, text="▶", command=self.play_pause, 
                                  bg=COLORS["accent"], fg="white", bd=0, font=("Segoe UI", 18), width=3)
        self.btn_play.pack(side="left", padx=10)
        tk.Button(ctl_frame, text="⏭", command=self.forward, **btn_props).pack(side="left", padx=5)

        info_frame = tk.Frame(center_frame, bg=COLORS["panel"])
        info_frame.pack(side="left", padx=20)
        self.lbl_time = tk.Label(info_frame, text="00:00.00", font=("Consolas", 18, "bold"), bg=COLORS["panel"], fg=COLORS["accent"])
        self.lbl_time.pack(side="left")
        tk.Label(info_frame, text=" / ", bg=COLORS["panel"], fg=COLORS["text_dim"]).pack(side="left")
        self.lbl_total = tk.Label(info_frame, text="00:00.00", font=("Consolas", 12), bg=COLORS["panel"], fg=COLORS["text_dim"])
        self.lbl_total.pack(side="left", pady=(6,0))

        self.status_label = tk.Label(self.root, text="就绪 - 请导入音频文件", bg=COLORS["bg"], fg=COLORS["text_dim"], font=("Segoe UI", 9), anchor="w")
        self.status_label.pack(side="bottom", fill="x", padx=5)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg")])
        if not path: return
        self.file_path = path
        self.update_status("正在加载音频...")
        threading.Thread(target=self._load_audio_thread, daemon=True).start()

    def _load_audio_thread(self):
        try:
            if AI_AVAILABLE:
                waveform, sr = torchaudio.load(self.file_path)
                self.audio_data = waveform.numpy().T
                self.sample_rate = sr
            else:
                sr, data = wavfile.read(self.file_path)
                if data.dtype != np.float32:
                    norm_factor = np.iinfo(data.dtype).max if data.dtype != np.float32 else 1.0
                    data = data.astype(np.float32) / norm_factor
                self.audio_data = data
                self.sample_rate = sr
            
            self.duration = len(self.audio_data) / self.sample_rate
            self.total_duration = max(60, self.duration + 5)
            self.root.after(0, self._on_audio_loaded)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("加载失败", str(e)))

    def _on_audio_loaded(self):
        # 载入新音频时，先清理旧片段（避免旧音轨残留/错乱）
        self._clear_clips_ui()

        self._draw_waveform()
        self._draw_timeline()
        self.btn_separate.config(state="normal")
        self.lbl_total.config(text=self._fmt_time(self.duration))

        # 若检测到已存在的分离结果（同级目录 *_vocals.wav 等），自动加载，避免每次都重新分离
        if self._try_load_existing_stems():
            self.update_status("检测到已分离文件：已自动载入分轨（无需重新分离）")
        else:
            self.update_status(f"已加载: {os.path.basename(self.file_path)}")



    def _clear_clips_ui(self):
        """清理当前工程里的片段与画布元素"""
        self.clips.clear()
        if hasattr(self, "timeline") and self.timeline is not None:
            self.timeline.delete("clip")

    def _load_wav_file_as_float(self, wav_path):
        """读取 wav 并统一转为 float32 [-1, 1] 的双声道数组"""
        sr, data = wavfile.read(wav_path)
        if np.issubdtype(data.dtype, np.integer):
            max_val = float(np.iinfo(data.dtype).max)
            data = data.astype(np.float32) / max_val
        else:
            data = data.astype(np.float32)

        # 统一为双声道
        if data.ndim == 1:
            data = np.column_stack((data, data))
        elif data.ndim == 2 and data.shape[1] == 1:
            data = np.column_stack((data[:, 0], data[:, 0]))

        return sr, data

    def _try_load_existing_stems(self):
        """若检测到同级目录已有分离结果，则自动载入（避免反复分离）"""
        if not self.file_path:
            return False

        base = os.path.splitext(self.file_path)[0]
        # 与保存逻辑保持一致：{base}_{stem}.wav
        stem_order = ["vocals", "drums", "bass", "other"]
        found = []
        for stem in stem_order:
            p = f"{base}_{stem}.wav"
            if os.path.exists(p):
                found.append((stem, p))

        if not found:
            return False

        # 清理旧片段并重新加载
        self._clear_clips_ui()

        max_dur = 0.0
        loaded = []

        for stem, p in found:
            try:
                sr, audio = self._load_wav_file_as_float(p)
            except Exception as e:
                print(f"⚠ 读取分离文件失败: {p} -> {e}")
                continue

            dur = len(audio) / sr
            max_dur = max(max_dur, dur)
            loaded.append((stem, sr, audio))

        if not loaded:
            return False

        # 需要更长时间轴时，扩展并重绘
        if max_dur + 5 > self.total_duration:
            self.total_duration = max(60, max_dur + 5)
            self._draw_timeline()

        idx_map = {"vocals": 0, "drums": 1, "bass": 2, "other": 3}
        for stem, sr, audio in loaded:
            self._add_clip_safe(audio, sr, stem, idx_map.get(stem, 0))

        return True

    def _draw_waveform(self):
        self.ax.clear()
        step = max(1, len(self.audio_data) // 8000)
        data = self.audio_data[::step]
        x = np.linspace(0, self.duration, len(data))
        
        if len(data.shape) > 1:
            self.ax.fill_between(x, data[:,0], alpha=0.8, color=COLORS["accent"], label="L")
            self.ax.fill_between(x, data[:,1], alpha=0.4, color="#ffffff", label="R")
        else:
            self.ax.fill_between(x, data, alpha=0.8, color=COLORS["accent"])
            
        self._setup_ax_style(show_text=False)
        self.ax.set_xlim(0, self.duration)

    def run_separation(self):
        self.btn_separate.config(state="disabled", text="⏳ 处理中...")
        self.update_status("正在分离中 (这也将保存分轨文件到本地)...")
        threading.Thread(target=self._separation_thread, daemon=True).start()

    def _separation_thread(self):
        try:
            if AI_AVAILABLE: self._separate_demucs()
            else: self._separate_basic()
            self.root.after(0, self._on_sep_done)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", str(e)))
            self.root.after(0, lambda: self.btn_separate.config(state="normal", text="⚡ 开始分离"))

    def _separate_demucs(self):
        if self.demucs_model is None:
            self.demucs_model = get_model("htdemucs")
            
        waveform, sr = torchaudio.load(self.file_path)
        if sr != self.demucs_model.samplerate:
            waveform = torchaudio.transforms.Resample(sr, self.demucs_model.samplerate)(waveform)
        if waveform.shape[0] == 1: waveform = waveform.repeat(2, 1)
        
        ref = waveform.mean(0)
        waveform = (waveform - ref.mean()) / ref.std()
        
        with torch.no_grad():
            sources = apply_model(self.demucs_model, waveform[None], progress=True)[0]
            
        sources = sources * ref.std() + ref.mean()
        source_names = ["drums", "bass", "other", "vocals"]
        
        self.root.after(0, self.clips.clear)
        self.root.after(0, lambda: self.timeline.delete("clip"))

        # --- 恢复的功能：保存文件 ---
        base_name = os.path.splitext(self.file_path)[0]

        for i, name in enumerate(source_names):
            audio = sources[i].cpu().numpy().T
            
            # 保存文件 (float32 -> int16)
            save_path = f"{base_name}_{name}.wav"
            # 简单的防止爆音限制
            max_val = np.max(np.abs(audio))
            if max_val > 0.99:
                 audio_norm = audio / max_val * 0.99
            else:
                 audio_norm = audio
            
            wavfile.write(save_path, self.demucs_model.samplerate, (audio_norm * 32767).astype(np.int16))
            print(f"已保存: {save_path}")

            self._add_clip_safe(audio, self.demucs_model.samplerate, name, i)

    def _separate_basic(self):
        data = self.audio_data
        if len(data.shape) == 1: data = np.column_stack((data, data))
        
        nyq = 0.5 * self.sample_rate
        def get_filter(cutoff, btype):
            b, a = butter(4, cutoff/nyq, btype=btype)
            return lfilter(b, a, data, axis=0)

        stems = {
            "bass": get_filter(200, 'low'),
            "drums": get_filter(2000, 'low') - get_filter(200, 'low'),
            "vocals": get_filter(2000, 'high')
        }
        
        self.root.after(0, self.clips.clear)
        self.root.after(0, lambda: self.timeline.delete("clip"))
        
        # --- 保存文件 ---
        base_name = os.path.splitext(self.file_path)[0]

        for i, (name, audio) in enumerate(stems.items()):
            save_path = f"{base_name}_{name}.wav"
            max_val = np.max(np.abs(audio))
            if max_val > 0.99: audio = audio / max_val * 0.99
            wavfile.write(save_path, self.sample_rate, (audio * 32767).astype(np.int16))
            
            self._add_clip_safe(audio, self.sample_rate, name, i)

    def _add_clip_safe(self, audio, sr, name, idx):
        duration = len(audio) / sr
        track_map = {"vocals":0, "drums":1, "bass":2, "other":3}
        mapped_idx = track_map.get(name.lower().split()[0], idx)
        track_cfg = TRACK_CONFIG[min(mapped_idx, len(TRACK_CONFIG)-1)]
        
        self.root.after(0, lambda: self.clips.append(
            AudioClip(self.timeline, mapped_idx, duration, track_cfg["color"], 
                     name.upper(), audio, sr, self)
        ))

    def _on_sep_done(self):
        self.btn_separate.config(state="normal", text="⚡ 开始分离")
        self.update_status("分离完成！分轨文件已保存在原目录。")
        messagebox.showinfo("完成", "音轨分离已完成。\n\nwav文件已保存在源音频同级目录下。")

    def play_pause(self):
        if self.player.playing:
            if self.player.paused:
                self.player.play()
                self.btn_play.config(text="⏸")
                self.update_status("播放中...")
            else:
                self.player.pause()
                self.btn_play.config(text="▶")
                self.update_status("已暂停")
        else:
            self.player.play()
            self.btn_play.config(text="⏸")
            self.update_status("播放中...")

    def stop(self):
        self.player.stop()
        self.on_playback_stopped()
        self.update_status("已停止")

    def on_playback_stopped(self):
        self.btn_play.config(text="▶")
        self.update_playhead_ui(0)

    def rewind(self):
        t = max(0, self.player.current_time - 5)
        self.player.seek(t)
        self.update_playhead_ui(t)

    def forward(self):
        t = min(self.total_duration, self.player.current_time + 5)
        self.player.seek(t)
        self.update_playhead_ui(t)

    def _seek_to_x(self, canvas, x):
        """把画布上的 x 坐标换算为时间并定位播放头（支持拖动 scrub）"""
        cx = canvas.canvasx(x)
        t = max(0.0, min(self.total_duration, cx / PX_PER_SEC))
        self.player.seek(t)
        self.update_playhead_ui(t)

    def _on_timeline_press(self, event):
        # 如果点击在片段上，交给片段自身的拖拽逻辑，避免“点片段却把播放头拖走”
        if event.widget == self.timeline:
            tags = self.timeline.gettags("current")
            if "clip" in tags:
                return
        self.scrubbing = True
        self._seek_to_x(event.widget, event.x)

    def _on_timeline_drag(self, event):
        if not self.scrubbing:
            return
        self._seek_to_x(event.widget, event.x)

    def _on_timeline_release(self, event):
        self.scrubbing = False

    def update_playhead_ui(self, t):
        self.lbl_time.config(text=self._fmt_time(t))
        self._draw_playhead_ui(t)

    def _draw_playhead_ui(self, t):
        self.timeline.delete("playhead")
        self.time_ruler.delete("playhead")
        x = t * PX_PER_SEC
        h = len(TRACK_CONFIG) * TRACK_HEIGHT
        self.time_ruler.create_polygon(x-6, RULER_HEIGHT-12, x+6, RULER_HEIGHT-12, x, RULER_HEIGHT-1, fill=COLORS["playhead"], tags="playhead")
        self.timeline.create_line(x, 0, x, h, fill=COLORS["playhead"], width=2, tags="playhead")
        
        view_left = self.timeline.canvasx(0)
        view_width = self.timeline.winfo_width()
        if x > view_left + view_width - 50 or x < view_left:
            pos = max(0, (x - 100) / (self.total_duration * PX_PER_SEC))
            self.timeline.xview_moveto(pos)
            self.time_ruler.xview_moveto(pos)

    def _fmt_time(self, s):
        m = int(s // 60)
        sec = s % 60
        return f"{m:02d}:{sec:05.2f}"

    def update_status(self, text):
        self.status_label.config(text=f" {text}")

    def _on_scroll(self, *args):
        self.timeline.xview(*args)
        self.time_ruler.xview(*args)

    def on_close(self):
        self.player.cleanup()
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass

    root = tk.Tk()
    app = ModernStudioApp(root)
    root.mainloop()