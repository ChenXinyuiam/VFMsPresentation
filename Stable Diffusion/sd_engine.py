import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline, 
    StableDiffusionInpaintPipeline
)

class StableDiffusionEngine:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model_id = model_id
        
        # 优先从本地缓存加载，缓存不存在时自动联网下载
        try:
            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=self.dtype, local_files_only=True
            ).to(self.device)
        except Exception:
            print(f"[INFO] 本地缓存未找到，正在从 HuggingFace 下载模型: {model_id}")
            self.txt2img_pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=self.dtype
            ).to(self.device)
        
        # 共享组件以节省显存 (非常重要)
        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            safety_checker=self.txt2img_pipe.safety_checker
        ).to(self.device)
        
        self.inpaint_pipe = StableDiffusionInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            tokenizer=self.txt2img_pipe.tokenizer,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            feature_extractor=self.txt2img_pipe.feature_extractor,
            safety_checker=self.txt2img_pipe.safety_checker
        ).to(self.device)

        if self.device == "cuda":
            self.txt2img_pipe.enable_attention_slicing()

    def txt2img(self, prompt, neg_prompt, steps, scale):
        return self.txt2img_pipe(
            prompt=prompt, negative_prompt=neg_prompt,
            num_inference_steps=steps, guidance_scale=scale
        ).images[0]

    def img2img(self, prompt, neg_prompt, init_image, strength, steps, scale):
        # 转换图像尺寸为 512x512（符合 SD v1.5 训练分布，保证生成质量）
        init_image = init_image.convert("RGB").resize((512, 512))
        return self.img2img_pipe(
            prompt=prompt, negative_prompt=neg_prompt, image=init_image,
            strength=strength, num_inference_steps=steps, guidance_scale=scale
        ).images[0]

    def inpaint(self, prompt, neg_prompt, dict_data, strength, steps, scale):
        image = dict_data['background'].convert("RGB").resize((512, 512))
        # 提取 alpha 通道作为 mask：涂抹处 alpha=255(白=重绘)，未涂抹处 alpha=0(黑=保留)
        mask = dict_data['layers'][0].split()[-1].resize((512, 512))
        
        return self.inpaint_pipe(
            prompt=prompt, 
            negative_prompt=neg_prompt, 
            image=image,
            mask_image=mask, 
            strength=strength, # 添加这个关键参数
            num_inference_steps=steps, 
            guidance_scale=scale
        ).images[0]