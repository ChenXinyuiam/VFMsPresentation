import torch
from diffusers import StableDiffusionPipeline

class SDGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # 加载模型
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        # 如果显存较小，开启优化
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()

    def generate(self, prompt, negative_prompt="", steps=25, scale=7.5):
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=scale
        ).images[0]
        return image