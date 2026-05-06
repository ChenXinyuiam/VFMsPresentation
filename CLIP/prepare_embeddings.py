from pathlib import Path

import clip
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


MODEL_NAME = "ViT-B/32"
EMBEDDING_DIR = Path("embeddings")
EMBEDDING_PATH = EMBEDDING_DIR / "flickr8k_image_embeddings.npy"
PATHS_PATH = EMBEDDING_DIR / "flickr8k_image_paths.txt"
IMAGE_DIR = EMBEDDING_DIR / "flickr8k_images"
BATCH_SIZE = 64


def get_image(example):
    image = example.get("image")
    if image is not None:
        return image.convert("RGB")

    for key in ("image_path", "file_name", "filename"):
        value = example.get(key)
        if value:
            return Image.open(value).convert("RGB")

    raise KeyError(f"Could not find an image field. Available fields: {list(example.keys())}")


def main():
    EMBEDDING_DIR.mkdir(exist_ok=True)
    IMAGE_DIR.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(MODEL_NAME, device=device)
    model.eval()

    dataset = load_dataset("jxie/flickr8k", split="train")
    embeddings = []
    image_paths = []

    with torch.no_grad():
        for start in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Embedding Flickr8k"):
            batch = dataset.select(range(start, min(start + BATCH_SIZE, len(dataset))))
            images = []

            for offset, example in enumerate(batch):
                image = get_image(example)
                image_path = IMAGE_DIR / f"{start + offset:05d}.jpg"
                if not image_path.exists():
                    image.save(image_path, quality=92)
                images.append(image)
                image_paths.append(str(image_path))

            inputs = torch.stack([preprocess(image) for image in images]).to(device)
            image_features = model.encode_image(inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy().astype("float32"))

    np.save(EMBEDDING_PATH, np.concatenate(embeddings, axis=0))
    PATHS_PATH.write_text("\n".join(image_paths), encoding="utf-8")
    print(f"Saved embeddings to {EMBEDDING_PATH}")
    print(f"Saved image paths to {PATHS_PATH}")


if __name__ == "__main__":
    main()
