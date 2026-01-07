# ComfyUI_RH_DreamID-V

<p align="center">
  <img src="https://img.shields.io/badge/ComfyUI-Plugin-blue" alt="ComfyUI Plugin">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Python-3.8+-blue" alt="Python">
</p>

A ComfyUI plugin for [DreamID-V](https://github.com/bytedance/DreamID-V), enabling high-fidelity video face swapping powered by Diffusion Transformer technology.

## ✨ Features

- 🎭 **High-Fidelity Face Swapping**: Advanced video face swapping using Diffusion Transformer
- 🎬 **Video-Driven**: Use video as motion/pose driver
- 🖼️ **Reference Image**: Single face image as identity reference
- 🔧 **ComfyUI Integration**: Seamlessly integrated into ComfyUI workflows

## 📋 Nodes

This plugin provides two core nodes:

| Node Name | Description |
|-----------|-------------|
| `RunningHub_DreamID-V_Loader` | Load the DreamID-V model pipeline |
| `RunningHub_DreamID-V_Sampler` | Execute video face swapping sampling |

## 🛠️ Installation

### Method 1: Via ComfyUI Manager (Recommended)

1. Install [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
2. Search for `ComfyUI_RH_DreamID-V` in ComfyUI Manager
3. Click Install

### Method 2: Manual Installation

1. Navigate to ComfyUI's `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:

```bash
git clone https://github.com/HM-RunningHub/ComfyUI_RH_DreamID-V.git
```

3. Install dependencies:

```bash
cd ComfyUI_RH_DreamID-V
pip install -r requirements.txt
```

## 📦 Model Downloads & Configuration

This plugin requires the following model files:

### 1. Wan2.1-T2V-1.3B Base Model

Download from: [Hugging Face - Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)

Place the files in the following directory:
```
ComfyUI/models/Wan/Wan2.1-T2V-1.3B/
├── models_t5_umt5-xxl-enc-bf16.pth
├── Wan2.1_VAE.pth
└── google/umt5-xxl/  (tokenizer folder)
```

### 2. DreamID-V Model

Download from: [Hugging Face - DreamID-V](https://huggingface.co/bytedance/DreamID-V) or [GitHub Release](https://github.com/bytedance/DreamID-V/releases)

Place the file in the following directory:
```
ComfyUI/models/DreamID-V/
└── dreamidv.pth
```

### Complete Model Directory Structure

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

## 🚀 Usage

1. Add the `RunningHub_DreamID-V_Loader` node in ComfyUI to load the model
2. Add the `RunningHub_DreamID-V_Sampler` node
3. Connect the following inputs:
   - **pipeline**: Model pipeline from the Loader node
   - **video**: Driving video (containing motion/pose)
   - **ref_image**: Reference face image
4. Configure parameters:
   - **size**: Output size (832*480 or 1280*720)
   - **frame_num**: Number of frames (must be 4n+1, e.g., 81)
   - **sample_steps**: Sampling steps (default: 20)
   - **fps**: Frame rate (default: 24)
   - **seed**: Random seed

## ⚙️ Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| size | Output video resolution | 832*480 |
| frame_num | Number of output frames (4n+1) | 81 |
| sample_steps | Diffusion sampling steps | 20 |
| fps | Output video frame rate | 24 |
| seed | Random seed | 42 |

## 💻 System Requirements

- **GPU**: NVIDIA GPU with VRAM >= 16GB recommended
- **Python**: 3.8 or higher
- **CUDA**: 11.7 or higher
- **ComfyUI**: Latest version

## 📝 Dependencies

- torch >= 2.0.0
- torchvision >= 0.15.0
- easydict
- numpy
- Pillow
- opencv-python
- decord
- tqdm
- mediapipe

## 🙏 Acknowledgements

- [DreamID-V](https://github.com/bytedance/DreamID-V) - Original project by ByteDance
- [Wan Team](https://github.com/Wan-AI) - Wan video generation model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Powerful Stable Diffusion GUI

## 📄 License

This project is licensed under the [Apache-2.0 License](LICENSE).

## ⚠️ Disclaimer

This project is for educational and research purposes only. Please ensure compliance with relevant laws and regulations when using this tool. Do not use it for illegal purposes or to infringe upon the rights of others.

---

<p align="center">
  If you find this project helpful, please give it a ⭐ Star!
</p>

