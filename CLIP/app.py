from pathlib import Path

import clip
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CIFAR100


MODEL_NAME = "ViT-B/32"
EMBEDDING_DIR = Path("embeddings")
FLICKR_EMBEDDINGS = EMBEDDING_DIR / "flickr8k_image_embeddings.npy"
FLICKR_PATHS = EMBEDDING_DIR / "flickr8k_image_paths.txt"

CIFAR100_CLASSES = [
    "apple", "aquarium fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak tree", "orange", "orchid", "otter", "palm tree", "pear",
    "pickup truck", "pine tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea",
    "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow tree", "wolf", "woman", "worm",
]


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load(MODEL_NAME, device=device)
model.eval()


def normalize_image(image):
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return image.convert("RGB")


@torch.no_grad()
def build_cifar_text_features():
    prompts = [f"a photo of a {name}" for name in CIFAR100_CLASSES]
    inputs = clip.tokenize(prompts).to(device)
    text_features = model.encode_text(inputs)
    return text_features / text_features.norm(dim=-1, keepdim=True)


CIFAR_TEXT_FEATURES = build_cifar_text_features()


def zero_shot_predict(image):
    image = normalize_image(image)
    if image is None:
        return {}

    image = image.resize((32, 32), Image.BICUBIC)
    inputs = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (100.0 * image_features @ CIFAR_TEXT_FEATURES.T).softmax(dim=-1)[0]

    top_probs, top_indices = probs.topk(5)
    return {
        CIFAR100_CLASSES[index]: float(prob)
        for prob, index in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist())
    }


def load_flickr_index():
    if not FLICKR_EMBEDDINGS.exists() or not FLICKR_PATHS.exists():
        return None, []

    embeddings = np.load(FLICKR_EMBEDDINGS)
    paths = FLICKR_PATHS.read_text(encoding="utf-8").splitlines()
    return embeddings, paths


FLICKR_INDEX, FLICKR_IMAGE_PATHS = load_flickr_index()


def retrieve_similar_images(image):
    image = normalize_image(image)
    if image is None:
        return []
    if FLICKR_INDEX is None:
        raise gr.Error("Flickr8k embeddings not found. Run: python prepare_embeddings.py")

    inputs = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query = model.encode_image(inputs)
        query = query / query.norm(dim=-1, keepdim=True)

    scores = FLICKR_INDEX @ query.cpu().numpy()[0]
    top_indices = np.argsort(scores)[-6:][::-1]
    return [FLICKR_IMAGE_PATHS[index] for index in top_indices]


def sample_cifar_image():
    dataset = CIFAR100(root="data", train=False, download=True)
    index = np.random.randint(0, len(dataset))
    image, label = dataset[index]
    return image, CIFAR100_CLASSES[label]


with gr.Blocks(title="CLIP Demo") as demo:
    gr.Markdown("# CLIP 应用展示")

    with gr.Tab("Zero-Shot CIFAR100"):
        with gr.Row():
            zero_input = gr.Image(type="pil", label="输入图片")
            zero_output = gr.Label(num_top_classes=5, label="Top-5 预测")
        with gr.Row():
            zero_button = gr.Button("预测", variant="primary")
            sample_button = gr.Button("随机 CIFAR100 样例")
        sample_label = gr.Textbox(label="样例真实类别", interactive=False)
        zero_button.click(zero_shot_predict, zero_input, zero_output)
        sample_button.click(sample_cifar_image, outputs=[zero_input, sample_label])

    with gr.Tab("Flickr8k Image Retrieval"):
        with gr.Row():
            retrieval_input = gr.Image(type="pil", label="输入图片")
            retrieval_output = gr.Gallery(label="最相似的 6 张图片", columns=3, rows=2)
        retrieval_button = gr.Button("检索", variant="primary")
        retrieval_button.click(retrieve_similar_images, retrieval_input, retrieval_output)


if __name__ == "__main__":
    demo.launch(share=True)
