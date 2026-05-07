import gradio as gr
from sd_engine import StableDiffusionEngine

# 初始化引擎
engine = StableDiffusionEngine()

with gr.Blocks(title="Stable Diffusion Multi-Task Demo") as demo:
    gr.Markdown("# 🎨 Stable Diffusion 综合演示 (课程作业)")
    gr.Markdown("涵盖了文生图、图生图风格迁移以及图像局部重绘三个核心场景。")

    with gr.Tabs():
        # --- 场景 1: Text-to-Image ---
        with gr.TabItem("1. 文生图 (Text-to-Image)"):
            with gr.Row():
                with gr.Column():
                    t2i_p = gr.Textbox(label="Prompt", value="A futuristic city in Shaanxi, cyberpunk style, highly detailed")
                    t2i_np = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed")
                    t2i_step = gr.Slider(1, 50, 25, label="Steps")
                    t2i_cfg = gr.Slider(1, 20, 7.5, label="Guidance Scale")
                    t2i_btn = gr.Button("Generate", variant="primary")
                t2i_out = gr.Image(label="Result")
            t2i_btn.click(engine.txt2img, inputs=[t2i_p, t2i_np, t2i_step, t2i_cfg], outputs=t2i_out)

        # --- 场景 2: Image-to-Image ---
        with gr.TabItem("2. 图生图 (Image-to-Image)"):
            with gr.Row():
                with gr.Column():
                    i2i_img = gr.Image(type="pil", label="Source Image")
                    i2i_p = gr.Textbox(label="Prompt (Style to apply)", value="Van Gogh oil painting style")
                    i2i_np = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed")
                    i2i_str = gr.Slider(0, 1, 0.6, label="Strength (重绘强度)")
                    i2i_step = gr.Slider(1, 50, 25, label="Steps")
                    i2i_cfg = gr.Slider(1, 20, 7.5, label="Guidance Scale")
                    i2i_btn = gr.Button("Transform", variant="primary")
                i2i_out = gr.Image(label="Result")
            i2i_btn.click(engine.img2img, inputs=[i2i_p, i2i_np, i2i_img, i2i_str, i2i_step, i2i_cfg], outputs=i2i_out)

        # --- 场景 3: Inpainting ---
        with gr.TabItem("3. 局部重绘 (Inpainting)"):
            gr.Markdown("请上传图片并使用画笔涂抹你想修改/修复的区域。")
            with gr.Row():
                with gr.Column():
                    # 关键：使用 Editor 组件进行涂抹
                    inp_img = gr.ImageMask(type="pil", label="Draw Mask on Image")
                    inp_p = gr.Textbox(label="Prompt (What to put in mask)", value="a cute cat sitting there")
                    inp_np = gr.Textbox(label="Negative Prompt", value="low quality, blurry, deformed")
                    inp_strength = gr.Slider(0.0, 1.0, 0.75, step=0.05, label="Denoising Strength (重绘强度)")
                    inp_step = gr.Slider(1, 50, 25, label="Steps")
                    inp_cfg = gr.Slider(1, 20, 7.5, label="Guidance Scale")
                    inp_btn = gr.Button("Inpaint", variant="primary")
                inp_out = gr.Image(label="Result")
            inp_btn.click(engine.inpaint, inputs=[inp_p, inp_np, inp_img, inp_strength, inp_step, inp_cfg], outputs=inp_out)

if __name__ == "__main__":
    # 添加 inbrowser=True
    demo.launch(
        server_name="127.0.0.1", 
        server_port=7861, 
        inbrowser=True,  # 核心参数：启动后自动在默认浏览器打开
        share=False
    )