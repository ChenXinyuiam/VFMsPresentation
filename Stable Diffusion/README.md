# Stable Diffusion Multi-Task Demo

A comprehensive image generation demo based on **Stable Diffusion v1.5**, covering three core scenarios: Text-to-Image, Image-to-Image style transfer, and Inpainting, with an interactive Gradio web interface.

## Environment Setup

**Python Version**: 3.10

```bash
# 1. Create virtual environment
conda create -n SD python=3.10 -y
conda activate SD

# 2. Install dependencies
pip install -r requirements.txt
```

## Launch

```bash
python app.py
```

The app will automatically open at `http://127.0.0.1:7861` in your browser.


## Model Loading

On first run, the model weights for `runwayml/stable-diffusion-v1-5` are automatically downloaded from HuggingFace and cached locally. Subsequent launches load directly from the local cache without requiring internet access.

## Features

### 1. Text-to-Image
Generate images from scratch based on a text prompt.

| Parameter | Description |
|---|---|
| Prompt | Describe the content you want to generate |
| Negative Prompt | Specify content to exclude from the output |
| Steps | Number of denoising iterations; higher values yield more detail. Default: 25 |
| Guidance Scale | Prompt adherence strength; higher values follow the prompt more strictly. Default: 7.5 |

### 2. Image-to-Image
Use an existing image as a starting point and transform it according to the prompt. The input image is automatically resized to 512×512.

| Parameter | Description |
|---|---|
| Source Image | The input reference image |
| Prompt | Description of the target style |
| Negative Prompt | Specify unwanted styles or quality issues to exclude |
| Strength | Redraw strength; 0 = keep original, 1 = fully regenerate. Default: 0.6 |
| Steps | Number of denoising iterations. Default: 25 |
| Guidance Scale | Prompt adherence strength. Default: 7.5 |

### 3. Inpainting
Upload an image and paint over the region to be modified. The model regenerates only the masked area while keeping the rest unchanged.

| Parameter | Description |
|---|---|
| Draw Mask on Image | Upload an image and paint the region to modify (mask must be drawn, otherwise output remains unchanged) |
| Prompt | Describe the content to fill in the masked region |
| Negative Prompt | Specify content to exclude from the output |
| Denoising Strength | Redraw intensity for the masked area; higher values produce more drastic changes. Default: 0.75 |
| Steps | Number of denoising iterations. Default: 25 |
| Guidance Scale | Prompt adherence strength. Default: 7.5 |

