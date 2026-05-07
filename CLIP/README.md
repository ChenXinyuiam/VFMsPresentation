# CLIP Demo

This project demonstrates two CLIP-based visual foundation model tasks:

- Zero-shot image classification on CIFAR100 labels.
- Image retrieval on Flickr8k, returning the 6 most similar images.

## Setup

```powershell
conda activate clip
pip install -r requirements.txt
```

## Prepare Flickr8k Embeddings

```powershell
python prepare_embeddings.py
```

The generated files are saved under `embeddings/`:

- `flickr8k_image_embeddings.npy`
- `flickr8k_image_paths.txt`
- `flickr8k_images/`

## Run

```powershell
python app.py
```

The Gradio app starts locally and also creates a temporary public share link.
