# Demucs-vocal-separation-studio
基于 Demucs 预训练大模型的本地人声分离工作站：一键分离人声/鼓/贝斯/其他轨道，提供波形可视化与结果管理，缺少 AI 依赖时可自动降级为基础频段分离

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Demucs-AI%20Powered-green.svg" alt="Demucs">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

一个基于 **Demucs 深度学习大模型** 的专业音频分离工作站，可以将音乐分离成人声、鼓点、贝斯、其他乐器四个独立音轨。

---
<img width="1918" height="986" alt="image" src="https://github.com/user-attachments/assets/3d1f06c4-a808-4222-9359-5db516503922" />


## ✨ 功能特点

- 🤖 **AI 智能分离** - 使用 Facebook 开源的 Demucs 大模型，分离效果专业级
- 🎚️ **多轨道编辑器** - 专业界面，支持拖拽、静音、删除
- 🎵 **实时播放** - 支持混音播放、暂停、快进快退
- 📊 **波形可视化** - 实时显示音频波形
- 💾 **自动保存** - 分离后自动保存各音轨为 WAV 文件
- 🎨 **现代 UI** - 深色主题，专业 DAW 风格界面

---

## 📦 安装依赖

打开命令提示符（CMD）或 PowerShell，进入项目目录后运行：

```bash
pip install -r requirements.txt
```

#### ⚠️ 常见问题

**PyAudio 安装失败？**

Windows 用户可以使用预编译的 wheel 文件：

```bash
pip install pipwin
pipwin install pyaudio
```

或者手动下载：https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio

---

## 🧠 关于 Demucs 大模型

### 什么是 Demucs？

Demucs 是 **Facebook AI Research (Meta AI)** 开发的 **深度学习音源分离模型**，能够将音乐智能分离成：
- 🎤 **Vocals** - 人声
- 🥁 **Drums** - 鼓点/打击乐
- 🎸 **Bass** - 贝斯/低音
- 🎹 **Other** - 其他乐器（吉他、钢琴等）

> 📖 **官方仓库**: https://github.com/facebookresearch/demucs
> 
> 📄 **论文**: [Hybrid Transformers for Music Source Separation](https://arxiv.org/abs/2211.08553)

### Demucs 版本说明

本项目使用的是 **Hybrid Transformer Demucs (htdemucs)** 模型，这是目前效果最好的版本：

| 模型名称 | 说明 | 参数量 |
|---------|------|--------|
| `htdemucs` | 混合 Transformer 架构（默认） | ~80M |
| `htdemucs_ft` | 精调版本，效果更好 | ~80M |
| `htdemucs_6s` | 支持 6 音轨分离 | ~80M |
| `mdx_extra` | MDX 比赛优胜模型 | ~60M |

更多模型信息请访问 [Demucs 官方文档](https://github.com/facebookresearch/demucs#separating-tracks)。

### 模型权重文件

首次运行程序进行分离时，Demucs 会 **自动从网络下载** 预训练模型权重（约 300MB）。

下载的模型会被缓存到以下位置：
- **Windows**: `C:\Users\<用户名>\.cache\torch\hub\checkpoints\`
- **Linux/Mac**: `~/.cache/torch/hub/checkpoints/`

### 手动下载模型（可选）

如果网络较慢，可以手动下载后放入缓存目录：

1. 下载 htdemucs 模型：
   - 模型名称：`htdemucs`
   - 下载地址：https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th

2. 将下载的 `htdemucs.th` 文件放入：
   - Windows: `C:\Users\<你的用户名>\.cache\torch\hub\checkpoints\955717e8-8726e21a.th`
   - 如果目录不存在，请手动创建



---

## 🚀 运行程序

在项目目录下运行：

```bash
python separation-studio.py
```

---

## 📖 使用说明

### 基本操作流程

1. **导入音频** - 点击右上角 "📂 导入音频" 按钮，选择 WAV/MP3/FLAC/OGG 文件
2. **开始分离** - 点击 "开始分离" 按钮，等待 AI 处理（首次会下载模型）
3. **查看结果** - 分离完成后，四个音轨会显示在多轨编辑器中
4. **播放试听** - 使用底部播放控制栏播放、暂停、快进

### 编辑功能

- **拖拽片段** - 直接拖动音频片段调整位置
- **静音/取消静音** - 点击片段右上角的 🔊 图标，或右键菜单
- **删除片段** - 右键点击片段，选择"删除"
- **时间轴定位** - 点击时间标尺或轨道区域跳转播放位置

### 输出文件

分离完成后，会在 **源文件同目录** 下生成：
- `原文件名_vocals.wav` - 人声
- `原文件名_drums.wav` - 鼓点
- `原文件名_bass.wav` - 贝斯
- `原文件名_other.wav` - 其他乐器

---

## 💻 系统要求

| 配置项 | 最低要求 | 推荐配置 |
|--------|----------|----------|
| 操作系统 | Windows 10 / macOS / Linux | Windows 10/11 |
| Python | 3.8+ | 3.10+ |
| 内存 | 8GB | 16GB+ |
| 显卡 | 无要求（CPU可运行） | NVIDIA GPU（CUDA加速） |
| 硬盘 | 500MB（含模型） | 1GB+ |

### GPU 加速（可选）

如果你有 NVIDIA 显卡，可以安装 CUDA 版本的 PyTorch 来加速分离过程：

```bash
# 先卸载 CPU 版本
pip uninstall torch torchaudio

# 安装 CUDA 版本（以 CUDA 11.8 为例）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

查看 PyTorch 官网获取适合你显卡的版本：https://pytorch.org/get-started/locally/

---

## 🔧 备用模式

如果 Demucs 安装失败，程序会自动切换到 **基础频段分离模式**：
- 使用传统的数字信号处理算法
- 按频率范围简单分离
- 效果不如 AI 模型，但无需额外依赖

---

## 📁 项目结构

```
人声分离/
├── 人声分离.py      # 主程序
├── requirements.txt  # 依赖列表
├── README.md         # 说明文档    
```

---

## ❓ 常见问题

### Q: 分离速度很慢？
A: CPU 模式下分离一首 3 分钟的歌曲约需 2-5 分钟，建议使用 NVIDIA GPU 加速，一般1分钟内可以分离完成

### Q: 提示 "模型库未安装"？
A: 运行 `pip install demucs torch torchaudio` 安装 AI 模型依赖。

### Q: PyAudio 安装报错？
A: Windows 用户请使用 `pipwin install pyaudio` 或下载预编译的 wheel 文件。

### Q: 分离效果不好？
A: 
- 确保使用的是 Demucs 模式（界面显示"Demucs大模型已就绪"）
- 原音频质量越高，分离效果越好
- 某些混音复杂的音乐可能效果一般

---

## 📜 开源协议

本项目基于 MIT 协议开源。

Demucs 模型由 Facebook AI Research 开发，遵循其开源协议。

---

## 🙏 致谢

- [Demucs](https://github.com/facebookresearch/demucs) - Facebook AI 音源分离模型
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [Matplotlib](https://matplotlib.org/) - 波形可视化

