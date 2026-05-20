# SAM2 Gradio Demo

基于 SAM2 的演示脚本，用于展示图像自动分割、基于提示的图像分割和视频目标跟踪功能。

## 功能说明

1. 自动分割
   - 上传图片后，自动生成多个候选 mask。
   - 支持调整 `points_per_side` 来控制自动分割的精度。

2. 图像提示分割
   - 支持前景点、背景点和 Box 提示。
   - 点击上传的图片即可添加提示点。
   - 支持输出单个 mask 或多个候选 mask。

3. 视频目标跟踪
   - 支持加载视频预览帧并直接在预览帧上点击目标。
   - 支持点提示和框提示。
   - 会生成带 mask 覆盖的输出视频。

## 依赖环境

- Python 3.8+
- PyTorch
- OpenCV (`opencv-python`)
- Gradio
- Pillow
- NumPy

本地 SAM2 模型权重和配置文件：

- `checkpoints/sam2.1_hiera_small.pt`
- `configs/sam2.1/sam2.1_hiera_s.yaml`