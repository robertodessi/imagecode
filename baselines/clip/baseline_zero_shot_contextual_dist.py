# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from PIL import Image

import clip
import torch
import tqdm

random.seed(10)
torch.manual_seed(10)


def encode_images(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos])
    photos_preprocessed = photos_preprocessed.to(device)

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
parser.add_argument("--valid_descr_path", default="../../data/valid_data.json")
parser.add_argument("--clip_model", default="ViT-B/16")
parser.add_argument("--imgs_path", default="../../data/images/")
parser.add_argument("--output_path", default=None)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")
model, preprocess = clip.load(args.clip_model, device=device)

clip.model.convert_weights(model)

img_dirs = args.imgs_path
valid_data = json.load(open(args.valid_descr_path, "r"))
valid = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        valid.append((img_dir, int(img_idx), text))

correct = 0
ranks = defaultdict(int)
accs, captions, is_video = [], [], []
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

    captions.append(text)
    accs.append(ranked_files[0] == target)
    is_video.append("open-images" not in img_dir)

print(correct)
print(len(valid))
print(ranks)
acc = correct / len(valid)
print(f"final_acc {acc}")


if args.output_path is not None:
    acc_tnsr = torch.Tensor(accs).float()
    interaction = dict(
        sender_input=None,
        receiver_input=None,
        labels=None,
        message=None,
        receiver_output=None,
        message_length=None,
        aux={"acc": acc_tnsr},
        aux_input={"decoded_captions": captions, "decoded_messages": None},
    )

    output_path = Path(args.output_path)
    torch.save(interaction, output_path / "interaction")
