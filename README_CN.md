# ComfyUI_RH_DreamID-V

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Plugin-blue" alt="ComfyUI Plugin">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p>

本项目是 [DreamID-V](https://github.com/bytedance/DreamID-V) 的 ComfyUI 插件版本，用于实现高保真视频人脸交换功能。

## ✨ 功能特点

- 🎭 **高保真人脸交换**：基于 Diffusion Transformer 的视频人脸交换技术
- 🎬 **视频驱动**：支持使用视频作为动作驱动源
- 🖼️ **参考图像**：使用单张人脸图像作为身份参考
- 🔧 **ComfyUI 集成**：完美集成到 ComfyUI 工作流中

## 📋 节点说明

本插件提供两个核心节点：

| 节点名称 | 功能说明 |
|---------|---------|
| `RunningHub_DreamID-V_Loader` | 加载 DreamID-V 模型管线 |
| `RunningHub_DreamID-V_Sampler` | 执行视频人脸交换采样 |

## 🛠️ 安装指南

### 方法一：通过 ComfyUI Manager 安装（推荐）

1. 安装 [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. 在 ComfyUI Manager 中搜索 `ComfyUI_RH_DreamID-V`
3. 点击安装

### 方法二：手动安装

1. 进入 ComfyUI 的 `custom_nodes` 目录：

```bash
cd ComfyUI/custom_nodes
```

2. 克隆本仓库：

```bash
git clone https://github.com/HM-RunningHub/ComfyUI_RH_DreamID-V.git
```

3. 安装依赖：

```bash
cd ComfyUI_RH_DreamID-V
pip install -r requirements.txt
```

## 📦 模型下载与配置

本插件需要下载以下模型文件（参考 [官方模型准备指南](https://github.com/bytedance/DreamID-V#model-preparation)）：

| 模型 | 下载链接 | 说明 |
|------|----------|------|
| DreamID-V | 🤗 [Huggingface](https://huggingface.co/XuGuo699/DreamID-V) | 支持 480P 和 720P |
| Wan-2.1 | 🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | VAE 和文本编码器 |

### 1. Wan2.1-T2V-1.3B 基础模型

下载地址：🤗 [Huggingface - Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

下载后放置到以下目录：
```
ComfyUI/models/Wan/Wan2.1-T2V-1.3B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/umt5-xxl/  (tokenizer 文件夹)
```

### 2. DreamID-V 模型

下载地址：🤗 [Huggingface - DreamID-V](https://huggingface.co/XuGuo699/DreamID-V)

下载后放置到以下目录：
```
ComfyUI/models/DreamID-V/
└── dreamidv.pth
```

### 模型目录结构总览

```
ComfyUI/
└── models/
    ├── Wan/
    │   └── Wan2.1-T2V-1.3B/
    │       ├── models_t5_umt5-xxl-enc-bf16.pth
    │       ├── Wan2.1_VAE.pth
    │       └── google/
    │           └── umt5-xxl/
    └── DreamID-V/
        └── dreamidv.pth
```

## 🚀 使用方法

1. 在 ComfyUI 中添加 `RunningHub_DreamID-V_Loader` 节点加载模型
2. 添加 `RunningHub_DreamID-V_Sampler` 节点
3. 连接以下输入：
   - **pipeline**：来自 Loader 节点的模型管线
   - **video**：驱动视频（包含动作姿态）
   - **ref_image**：参考人脸图像
4. 配置参数：
   - **size**：输出尺寸（832*480 或 1280*720）
   - **frame_num**：帧数（需为 4n+1，如 81）
   - **sample_steps**：采样步数（默认 20）
   - **fps**：帧率（默认 24）
   - **seed**：随机种子

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| size | 输出视频尺寸 | 832*480 |
| frame_num | 输出帧数 (4n+1) | 81 |
| sample_steps | 扩散采样步数 | 20 |
| fps | 输出视频帧率 | 24 |
| seed | 随机种子 | 42 |

## 💻 系统要求

- **GPU**：建议使用 NVIDIA GPU，显存 >= 16GB
- **Python**：3.8 或更高版本
- **CUDA**：11.7 或更高版本
- **ComfyUI**：最新版本

## 📝 依赖项

- torch >= 2.0.0
- torchvision >= 0.15.0
- easydict
- numpy
- Pillow
- opencv-python
- decord
- tqdm
- mediapipe

## 🙏 致谢

- [DreamID-V](https://github.com/bytedance/DreamID-V) - 字节跳动团队的原始项目
- [Wan Team](https://github.com/Wan-AI) - Wan 视频生成模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 强大的 Stable Diffusion GUI

## 📄 许可证

本项目基于 [Apache-2.0 License](LICENSE) 开源。

## ⚠️ 免责声明

本项目仅供学习和研究使用。请确保在使用时遵守相关法律法规，不要将其用于非法用途或侵犯他人权益的行为。

---

<p align="center">
  如果这个项目对你有帮助，欢迎给个 ⭐ Star！
</p>

