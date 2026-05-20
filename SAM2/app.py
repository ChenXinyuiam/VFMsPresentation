import os
import gc
import cv2
import math
import tempfile
import numpy as np
import torch
import gradio as gr
from PIL import Image

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"
SAM2_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

MAX_IMAGE_SIDE = 1280
MAX_VIDEO_SIDE = 720
MAX_VIDEO_FRAMES = 120

image_predictor = None
video_predictor = None


def clear_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def resize_long_side(image, max_side):
    h, w = image.shape[:2]
    scale = min(max_side / max(h, w), 1.0)

    if scale >= 1.0:
        return image, 1.0

    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def parse_points(text):
    if text is None or text.strip() == "":
        return None

    points = []
    for item in text.split(";"):
        item = item.strip()
        if item == "":
            continue
        x, y = item.split(",")
        points.append([float(x), float(y)])

    if len(points) == 0:
        return None

    return np.array(points, dtype=np.float32)


def parse_box(text):
    if text is None or text.strip() == "":
        return None

    values = [float(v.strip()) for v in text.split(",")]

    if len(values) != 4:
        raise gr.Error("Box 格式应为：x1,y1,x2,y2")

    return np.array(values, dtype=np.float32)


def scale_points(points, scale):
    if points is None:
        return None
    return points * scale


def scale_box(box, scale):
    if box is None:
        return None
    return box * scale


def overlay_masks(image, masks, alpha=0.55):
    if isinstance(masks, np.ndarray):
        if masks.ndim == 2:
            masks = [masks]
        elif masks.ndim == 3:
            masks = list(masks)

    output = image.copy().astype(np.float32)
    rng = np.random.default_rng(42)

    for mask in masks:
        mask = np.squeeze(mask).astype(bool)
        color = rng.integers(0, 255, size=3).astype(np.float32)
        output[mask] = output[mask] * (1 - alpha) + color * alpha

    return np.clip(output, 0, 255).astype(np.uint8)


def draw_prompts(image, pos_points=None, neg_points=None, box=None):
    output = image.copy()

    if pos_points is not None:
        for x, y in pos_points:
            cv2.circle(output, (int(x), int(y)), 8, (0, 255, 0), -1)

    if neg_points is not None:
        for x, y in neg_points:
            cv2.circle(output, (int(x), int(y)), 8, (255, 0, 0), -1)

    if box is not None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 255, 0), 3)

    return output


def add_click_point(image, point_type, pos_text, neg_text, evt: gr.SelectData):
    x, y = evt.index
    point = f"{x},{y}"

    if point_type == "前景点":
        if pos_text is None or pos_text.strip() == "":
            pos_text = point
        else:
            pos_text = pos_text + ";" + point
    else:
        if neg_text is None or neg_text.strip() == "":
            neg_text = point
        else:
            neg_text = neg_text + ";" + point

    return pos_text, neg_text


def add_video_click_point(image, current_text, evt: gr.SelectData):
    x, y = evt.index
    point = f"{x},{y}"

    if current_text is None or current_text.strip() == "":
        return point
    return current_text + ";" + point


def clear_prompt_points():
    return "", ""


def clear_video_points():
    return ""


def get_image_predictor():
    global image_predictor

    if image_predictor is None:
        clear_cuda()
        sam2_model = build_sam2(
            SAM2_MODEL_CFG,
            SAM2_CHECKPOINT,
            device=DEVICE
        )
        image_predictor = SAM2ImagePredictor(sam2_model)

    return image_predictor


def get_video_predictor():
    global video_predictor

    if video_predictor is None:
        clear_cuda()
        video_predictor = build_sam2_video_predictor(
            SAM2_MODEL_CFG,
            SAM2_CHECKPOINT,
            device=DEVICE
        )

    return video_predictor


def run_auto_mask(image, points_per_side):
    if image is None:
        raise gr.Error("请先上传图片")

    image = np.array(image.convert("RGB"))
    image, scale = resize_long_side(image, MAX_IMAGE_SIDE)

    clear_cuda()

    sam2_model = build_sam2(
        SAM2_MODEL_CFG,
        SAM2_CHECKPOINT,
        device=DEVICE,
        apply_postprocessing=False
    )

    generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=int(points_per_side),
        points_per_batch=32,
        pred_iou_thresh=0.80,
        stability_score_thresh=0.90,
        crop_n_layers=0,
        min_mask_region_area=100
    )

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                masks = generator.generate(image)
        else:
            masks = generator.generate(image)

    if len(masks) == 0:
        return Image.fromarray(image), "未生成 mask"

    mask_arrays = [m["segmentation"] for m in masks]
    result = overlay_masks(image, mask_arrays)

    info = []
    info.append(f"生成 mask 数量：{len(masks)}")
    info.append(f"图像缩放比例 scale：{scale:.4f}")
    info.append("")
    info.append("前 10 个 mask 信息：")

    for i, m in enumerate(masks[:10]):
        area = m.get("area", None)
        bbox = m.get("bbox", None)
        pred_iou = m.get("predicted_iou", 0)
        stability = m.get("stability_score", 0)

        info.append(
            f"{i + 1}. area={area}, bbox={bbox}, "
            f"pred_iou={pred_iou:.4f}, stability={stability:.4f}"
        )

    del generator
    del sam2_model
    clear_cuda()

    return Image.fromarray(result), "\n".join(info)


def run_image_prompt(image, positive_points, negative_points, box_text, multimask):
    if image is None:
        raise gr.Error("请先上传图片")

    original_image = np.array(image.convert("RGB"))
    resized_image, scale = resize_long_side(original_image, MAX_IMAGE_SIDE)

    pos_points_original = parse_points(positive_points)
    neg_points_original = parse_points(negative_points)
    box_original = parse_box(box_text)

    pos_points = scale_points(pos_points_original, scale)
    neg_points = scale_points(neg_points_original, scale)
    box = scale_box(box_original, scale)

    point_coords = []
    point_labels = []

    if pos_points is not None:
        point_coords.extend(pos_points.tolist())
        point_labels.extend([1] * len(pos_points))

    if neg_points is not None:
        point_coords.extend(neg_points.tolist())
        point_labels.extend([0] * len(neg_points))

    if len(point_coords) > 0:
        point_coords = np.array(point_coords, dtype=np.float32)
        point_labels = np.array(point_labels, dtype=np.int32)
    else:
        point_coords = None
        point_labels = None

    if point_coords is None and box is None:
        raise gr.Error("至少需要输入前景点、背景点或 Box")

    predictor = get_image_predictor()

    with torch.inference_mode():
        if torch.cuda.is_available():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictor.set_image(resized_image)
                masks, scores, logits = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=multimask
                )
        else:
            predictor.set_image(resized_image)
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask
            )

    order = np.argsort(scores)[::-1]
    masks = masks[order]
    scores = scores[order]

    best_mask = masks[0]
    result = overlay_masks(resized_image, best_mask)
    result = draw_prompts(result, pos_points, neg_points, box)

    info = []
    info.append(f"原图尺寸：{original_image.shape[1]} x {original_image.shape[0]}")
    info.append(f"推理尺寸：{resized_image.shape[1]} x {resized_image.shape[0]}")
    info.append(f"坐标缩放比例 scale：{scale:.4f}")
    info.append("")
    info.append(f"输出 mask 数量：{len(masks)}")
    info.append(f"最高分 score：{float(scores[0]):.4f}")

    for i, score in enumerate(scores):
        info.append(f"mask {i + 1}: score={float(score):.4f}")

    clear_cuda()

    return Image.fromarray(result), "\n".join(info)


def get_video_frame_preview(video_path, frame_idx):
    if video_path is None:
        raise gr.Error("请先上传视频")

    frame_idx = int(frame_idx)
    frame_idx = max(0, frame_idx)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise gr.Error("无法读取视频帧数")

    frame_idx = min(frame_idx, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise gr.Error("无法读取指定帧")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    original_h, original_w = frame.shape[:2]

    resized_frame, scale = resize_long_side(frame, MAX_VIDEO_SIDE)
    resized_h, resized_w = resized_frame.shape[:2]

    info = (
        f"原视频帧尺寸：{original_w} x {original_h}\n"
        f"预览/推理帧尺寸：{resized_w} x {resized_h}\n"
        f"缩放比例 scale：{scale:.4f}\n"
        f"当前提示帧 index：{frame_idx}\n"
        f"请直接点击这张预览帧添加 point，坐标不会偏。"
    )

    return Image.fromarray(resized_frame), info


def extract_video_frames(video_path, max_frames, max_side):
    frame_dir = tempfile.mkdtemp(prefix="sam2_video_frames_")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or math.isnan(fps):
        fps = 24

    idx = 0

    while idx < max_frames:
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame, _ = resize_long_side(frame, max_side)

        frame_path = os.path.join(frame_dir, f"{idx:05d}.jpg")
        Image.fromarray(frame).save(frame_path)

        idx += 1

    cap.release()

    if idx == 0:
        raise gr.Error("未能从视频中读取帧")

    return frame_dir, fps, idx


def write_video_with_masks(frame_dir, video_segments, fps, output_path):
    frame_names = sorted(
        name for name in os.listdir(frame_dir)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    first_frame = np.array(
        Image.open(os.path.join(frame_dir, frame_names[0])).convert("RGB")
    )

    h, w = first_frame.shape[:2]

    temp_avi_path = output_path.replace(".mp4", "_temp.avi")

    writer = cv2.VideoWriter(
        temp_avi_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (w, h)
    )

    for frame_idx, frame_name in enumerate(frame_names):
        frame = np.array(
            Image.open(os.path.join(frame_dir, frame_name)).convert("RGB")
        )

        if frame_idx in video_segments:
            masks = []
            for obj_id, mask in video_segments[frame_idx].items():
                masks.append(np.squeeze(mask))

            frame = overlay_masks(frame, masks)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()

    ffmpeg_cmd = (
        f'ffmpeg -y -i "{temp_avi_path}" '
        f'-vcodec libx264 -pix_fmt yuv420p '
        f'-movflags +faststart '
        f'"{output_path}"'
    )

    os.system(ffmpeg_cmd)

    if os.path.exists(temp_avi_path):
        os.remove(temp_avi_path)


def run_video_tracking(video_path, frame_idx, prompt_type, point_text, box_text):
    if video_path is None:
        raise gr.Error("请先上传视频")

    frame_dir, fps, num_frames = extract_video_frames(
        video_path=video_path,
        max_frames=MAX_VIDEO_FRAMES,
        max_side=MAX_VIDEO_SIDE
    )

    frame_idx = int(frame_idx)
    frame_idx = max(0, min(frame_idx, num_frames - 1))

    predictor = get_video_predictor()

    with torch.inference_mode():
        if torch.cuda.is_available():
            context = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            context = torch.autocast("cpu", enabled=False)

        with context:
            inference_state = predictor.init_state(video_path=frame_dir)
            predictor.reset_state(inference_state)

            obj_id = 1

            if prompt_type == "point":
                points = parse_points(point_text)

                if points is None:
                    raise gr.Error("请先在提示帧预览图上点击目标，或者手动输入 x,y")

                labels = np.ones(len(points), dtype=np.int32)

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels
                )

            elif prompt_type == "box":
                box = parse_box(box_text)

                if box is None:
                    raise gr.Error("Box 格式应为：x1,y1,x2,y2。注意：这里使用预览帧坐标。")

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box
                )

            video_segments = {}

            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {}

                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                    video_segments[out_frame_idx][out_obj_id] = mask

    output_dir = tempfile.mkdtemp(prefix="sam2_video_output_")
    output_path = os.path.join(output_dir, "result.mp4")

    write_video_with_masks(
        frame_dir=frame_dir,
        video_segments=video_segments,
        fps=fps,
        output_path=output_path
    )

    clear_cuda()

    info = (
        f"处理帧数：{num_frames}\n"
        f"提示帧 index：{frame_idx}\n"
        f"视频推理最长边：{MAX_VIDEO_SIDE}\n"
        f"点和 Box 坐标均基于预览帧/推理帧，不再使用原视频坐标。\n"
        f"输出文件：{output_path}"
    )

    return output_path, info


with gr.Blocks(title="SAM2 Gradio Demo") as demo:
    gr.Markdown("# SAM2 三功能展示 Demo")

    gr.Markdown(
        "当前配置：\n\n"
        f"- checkpoint: `{SAM2_CHECKPOINT}`\n"
        f"- config: `{SAM2_MODEL_CFG}`\n"
        f"- device: `{DEVICE}`\n\n"
        "图像和视频都做了坐标修正：坐标与实际推理图像保持一致。"
    )

    with gr.Tab("1. 自动分割"):
        auto_img = gr.Image(type="pil", label="输入图片")

        auto_points = gr.Slider(
            minimum=8,
            maximum=64,
            value=32,
            step=8,
            label="points_per_side"
        )

        auto_btn = gr.Button("运行自动分割")
        auto_out = gr.Image(type="pil", label="自动分割结果")
        auto_info = gr.Textbox(label="mask 信息", lines=12)

        auto_btn.click(
            fn=run_auto_mask,
            inputs=[auto_img, auto_points],
            outputs=[auto_out, auto_info]
        )

    with gr.Tab("2. 图像提示分割"):
        gr.Markdown(
            "上传图片后，选择当前点击类型，然后直接点击图片。"
            "前景点表示目标，背景点表示需要排除的区域。"
        )

        img_input = gr.Image(
            type="pil",
            label="上传图片，然后点击图片添加点提示"
        )

        click_type = gr.Radio(
            choices=["前景点", "背景点"],
            value="前景点",
            label="当前点击类型"
        )

        pos_input = gr.Textbox(
            label="前景点 foreground points",
            placeholder="例如：500,375;620,410"
        )

        neg_input = gr.Textbox(
            label="背景点 background points",
            placeholder="例如：800,400"
        )

        clear_btn = gr.Button("清空前景点和背景点")

        box_input = gr.Textbox(
            label="Box",
            placeholder="例如：425,600,700,875"
        )

        multimask_input = gr.Checkbox(
            value=False,
            label="输出多个候选 mask"
        )

        img_btn = gr.Button("运行提示式分割")
        img_out = gr.Image(type="pil", label="分割结果")
        img_info = gr.Textbox(label="预测信息", lines=12)

        img_input.select(
            fn=add_click_point,
            inputs=[img_input, click_type, pos_input, neg_input],
            outputs=[pos_input, neg_input]
        )

        clear_btn.click(
            fn=clear_prompt_points,
            inputs=[],
            outputs=[pos_input, neg_input]
        )

        img_btn.click(
            fn=run_image_prompt,
            inputs=[
                img_input,
                pos_input,
                neg_input,
                box_input,
                multimask_input
            ],
            outputs=[img_out, img_info]
        )

    with gr.Tab("3. 视频目标跟踪"):
        gr.Markdown(
            "视频追踪中不要直接按原视频播放器坐标输入点。"
            "请先加载提示帧预览，然后直接点击预览帧添加 point。"
        )

        video_input = gr.Video(label="输入视频")

        frame_input = gr.Number(
            value=0,
            precision=0,
            label="提示所在帧 index"
        )

        load_frame_btn = gr.Button("加载提示帧预览")

        video_preview = gr.Image(
            type="pil",
            label="提示帧预览：请直接点击目标"
        )

        video_preview_info = gr.Textbox(
            label="提示帧信息",
            lines=6
        )

        prompt_type = gr.Radio(
            choices=["point", "box"],
            value="point",
            label="提示类型"
        )

        video_point = gr.Textbox(
            label="点提示 point，点击预览帧自动加入",
            placeholder="例如：210,350"
        )

        clear_video_btn = gr.Button("清空视频点提示")

        video_box = gr.Textbox(
            label="Box，使用预览帧坐标",
            placeholder="例如：300,0,500,400"
        )

        video_btn = gr.Button("运行视频分割与跟踪")
        video_out = gr.Video(label="输出视频")
        video_info = gr.Textbox(label="处理信息", lines=6)

        load_frame_btn.click(
            fn=get_video_frame_preview,
            inputs=[video_input, frame_input],
            outputs=[video_preview, video_preview_info]
        )

        video_preview.select(
            fn=add_video_click_point,
            inputs=[video_preview, video_point],
            outputs=video_point
        )

        clear_video_btn.click(
            fn=clear_video_points,
            inputs=[],
            outputs=video_point
        )

        video_btn.click(
            fn=run_video_tracking,
            inputs=[
                video_input,
                frame_input,
                prompt_type,
                video_point,
                video_box
            ],
            outputs=[video_out, video_info]
        )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )