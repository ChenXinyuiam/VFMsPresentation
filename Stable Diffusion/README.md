# Stable Diffusion Multi-Task Demo

基于 **Stable Diffusion v1.5** 的图像生成综合演示项目，涵盖文生图、图生图风格迁移、图像局部重绘三个核心场景，通过 Gradio Web 界面进行交互。

## 环境配置

**Python 版本**：3.10

```bash
# 1. 创建虚拟环境
conda create -n SD python=3.10 -y
conda activate SD

# 2. 安装依赖
pip install -r requirements.txt
```

## 启动

```bash
python app.py
```

启动后自动在浏览器打开 `http://127.0.0.1:7861`。


## 模型加载逻辑

首次运行时自动从 HuggingFace 下载 `runwayml/stable-diffusion-v1-5` 模型权重并缓存到本地；此后启动直接读取本地缓存，无需联网。

## 功能概览

### 1. 文生图（Text-to-Image）
根据文本 Prompt 从零生成图像。

| 参数 | 说明 |
|---|---|
| Prompt | 描述想要生成的画面内容 |
| Negative Prompt | 排除不希望出现的内容 |
| Steps | 去噪迭代次数，越大细节越丰富，默认 25 |
| Guidance Scale | Prompt 约束强度，越大越贴近描述，默认 7.5 |

### 2. 图生图（Image-to-Image）
以一张图片为起点，按照 Prompt 描述的风格进行改造。输入图片会被自动缩放至 512×512。

| 参数 | 说明 |
|---|---|
| Source Image | 输入原始图片 |
| Prompt | 目标风格描述 |
| Negative Prompt | 排除不希望的风格或质量问题 |
| Strength | 重绘强度，0=不改变原图，1=完全重新生成，默认 0.6 |
| Steps | 去噪迭代次数，默认 25 |
| Guidance Scale | Prompt 约束强度，默认 7.5 |

### 3. 局部重绘（Inpainting）
上传图片后用画笔涂抹需要修改的区域，模型仅对遮罩区域重新生成，其余区域保持不变。

| 参数 | 说明 |
|---|---|
| Draw Mask on Image | 上传图片并用画笔涂抹要修改的区域（必须涂抹，否则输出不变） |
| Prompt | 描述遮罩区域内应填充的内容 |
| Negative Prompt | 排除不希望的内容 |
| Denoising Strength | 遮罩区域重绘强度，越高改动越彻底，默认 0.75 |
| Steps | 去噪迭代次数，默认 25 |
| Guidance Scale | Prompt 约束强度，默认 7.5 |

