# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83
import json
import random
import clip
import torch
import tqdm
from torch import nn
from PIL import Image
from pathlib import Path
from collections import defaultdict
import argparse

random.seed(10)
torch.manual_seed(10)


def encode_images(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(
        device
    )

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)
    return photos_features.cpu().numpy()


def encode_text(search_query):
    with torch.no_grad():
        text_encoded = model.encode_text(
            clip.tokenize(search_query, truncate=True).to(device)
        )
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded.cpu().numpy()


def find_best_matches(text_features, photo_features):
    similarities = (photo_features @ text_features.T).squeeze(1)
    best_photo_idx = (-similarities).argsort()
    similarities = -similarities
    similarities.sort()
    return best_photo_idx, similarities


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--valid_descr_path", type=str, default="../../data/valid_data.json"
)
parser.add_argument(
    "--train_descr_path", type=str, default="../../data/train_data.json"
)
parser.add_argument(
    "--imgs_path", type=str, default="/network/scratch/b/benno.krojer/dataset/games"
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(
        model
    )  # Actually this line is unnecessary since clip by default already on float16

img_dirs = args.imgs_path
valid_data = json.load(open(args.valid_descr_path, "r"))
valid = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        valid.append((img_dir, int(img_idx), text))

correct = 0
ranks = defaultdict(int)
for img_dir, img_idx, text in tqdm.tqdm(valid):
    img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
    img_files = sorted(
        img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
    )
    img_embs = encode_images(img_files)
    text_emb = encode_text(text.strip())
    ranked_idx, sim = find_best_matches(text_emb, img_embs)
    ranked_files = [str(img_files[rank]).split("/")[-1][:-4] for rank in ranked_idx]
    target = str(img_files[int(img_idx)]).split("/")[-1][:-4]
    if ranked_files[0] == target:
        correct += 1
    ranks[ranked_files.index(target) + 1] += 1
print(correct)
print(len(valid))
print(ranks)
acc = correct / len(valid)
print(f"final_acc {acc}")
