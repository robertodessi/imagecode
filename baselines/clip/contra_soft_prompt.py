# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from PIL import Image

import clip
import torch

try:
    # requires python >= 3.7
    from contextlib import nullcontext
except ImportError:
    # not exactly the same, but will do for our purposes
    from contextlib import suppress as nullcontext

import tqdm

import wandb
from torch import nn, optim

random.seed(10)
torch.manual_seed(10)
wandb.init(project="soft-prompt-clip", settings=wandb.Settings(start_method="fork"))


def get_clip_embeddings(
    max_vocab: int = 5000,
    data_path: str = "/private/home/rdessi/imagecode/data/",
):
    assert max_vocab > 0

    data_path = Path(data_path)

    # not including the test set since it is unlabeled and not used
    with open(data_path / "train_data.json") as fd:
        train = json.load(fd)
    with open(data_path / "valid_data.json") as fd:
        valid = json.load(fd)

    train_and_valid = {**train, **valid}

    token_list = []
    for _, captions in train_and_valid.items():
        for caption in captions.values():
            token_list.extend(clip.tokenize(caption, truncate=True)[0].tolist())

    token_counter = Counter(token_list)

    max_vocab = max_vocab if max_vocab else len(token_counter)
    most_freq_tokens = [
        x[0]
        for x in token_counter.most_common(max_vocab + 3)
        if x[0] not in [49406, 49407, 0]  # eos, sos and pad
    ]

    return model.token_embedding.weight[most_freq_tokens]


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


def get_eot_idx(messages: torch.Tensor) -> torch.Tensor:
    max_k = messages.size(1)
    zero_mask = messages == 0

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths - 2


class SoftEmbedding(nn.Module):
    def __init__(
        self,
        embeddings,
        n_tokens: int = 5,
        init: str = "random",
        init_range: float = 0.5,
    ):
        super(SoftEmbedding, self).__init__()

        self.n_tokens = n_tokens
        init2fn = {
            "random": self._init_random,
            "random_embedding": self._init_sample_random_embedding,
        }
        learned_embeddings = init2fn[init](embeddings, n_tokens, init_range)
        self.learned_embeddings = nn.parameter.Parameter(learned_embeddings)

    def _init_random(self, embeddings, n_tokens, init_range):
        embed_dim = embeddings.size(1)
        learned_embeddings = torch.FloatTensor(n_tokens, embed_dim)
        learned_embeddings.uniform_(-init_range, init_range)
        return learned_embeddings

    def _init_sample_random_embedding(self, embeddings, n_tokens, *args, **kwargs):
        idxs = torch.randperm(embeddings.size(0))[: self.n_tokens]
        return embeddings[idxs].clone().detach()

    def forward(self, embeddings: torch.Tensor):
        learned_embeddings = self.learned_embeddings.repeat(embeddings.size(0), 1, 1)
        new_embeddings = [embeddings[:, :1], learned_embeddings, embeddings[:, 1:]]
        return torch.cat(new_embeddings, dim=1)


config = wandb.config
parser = argparse.ArgumentParser()
parser.add_argument("--batchsize", type=int, default=36)
parser.add_argument("--lr", type=float, default=4e-6)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument(
    "--valid_descr_path", type=str, default="../../data/valid_data.json"
)
parser.add_argument(
    "--train_descr_path", type=str, default="../../data/train_data.json"
)
parser.add_argument(
    "--imgs_path", type=str, default="/private/home/rdessi/imagecode/data/images/"
)
parser.add_argument("--n_tokens", type=int, default=2)
parser.add_argument("--max_clip_vocab", type=int, default=500)
parser.add_argument("--remove_head", action="store_true", default=False)
parser.add_argument(
    "--prompt_init",
    choices=["random", "random_embedding"],
    default="random",
)
parser.add_argument("--pretraining_epochs", type=int, default=2)

args = parser.parse_args()
wandb.config.update(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}", flush=True)
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
wandb.watch(model)
model.float()

# for p in model.parameters():
#     p.requires_grad = False

img_dirs = args.imgs_path

with open(args.valid_descr_path, "r") as fd:
    valid_data = json.load(fd)
with open(args.train_descr_path, "r") as fd:
    train_data = json.load(fd)

train = []
for img_dir, data in train_data.items():
    for img_idx, text in data.items():
        train.append((img_dir, int(img_idx), text))
valid = []
for img_dir, data in valid_data.items():
    for img_idx, text in data.items():
        valid.append((img_dir, int(img_idx), text))

loss_txt = nn.CrossEntropyLoss()


clip_embeddings = get_clip_embeddings(max_vocab=args.max_clip_vocab)

soft_embeddings = SoftEmbedding(
    embeddings=clip_embeddings, n_tokens=config.n_tokens, init=config.prompt_init
).to(device)

optimizer = optim.Adam(
    [dict(params=soft_embeddings.parameters()), dict(params=model.parameters())],
    lr=config.lr,  # fix
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=0.2,
)


def encode_images(images, finetuning=False):
    fwd_context = nullcontext() if finetuning else torch.no_grad()
    with fwd_context:
        photos_features = model.encode_image(images)
        photos_features = photos_features / photos_features.norm(dim=-1, keepdim=True)
    return photos_features


def _encode_text(text, finetuning, add_soft_prompt):
    ctx_len = text.shape[1]
    eot_idx = get_eot_idx(text)

    fwd_context = nullcontext() if finetuning else torch.no_grad()

    with fwd_context:
        # [batch_size, n_ctx, d_model]
        x = model.token_embedding(text).type(model.dtype)
        x = x + model.positional_embedding.type(model.dtype)

    if add_soft_prompt:
        if config.remove_head:
            x = torch.cat([x[:, :1], x[:, config.n_tokens + 1 :]], dim=1)  # noqa
        else:
            x = x[:, : -config.n_tokens]  # noqa

            if ctx_len - config.n_tokens <= eot_idx:
                # 49407 is the  eot token for clip
                eot = model.token_embedding(torch.Tensor([49407]).long().to(device))
                eot = eot + model.positional_embedding[x.size(1) - 1].type(model.dtype)
                x[:, -1] = eot

        x = soft_embeddings(x)

    with fwd_context:
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = model.ln_final(x).type(model.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

    return x


def encode_text(text, finetuning=False, add_soft_prompt=True):
    text_encoded = _encode_text(text, finetuning, add_soft_prompt)
    text_encoded = text_encoded / text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded


def model_forward(image, text, finetuning, add_soft_prompt):
    image_features = encode_images(image, finetuning)
    text_features = encode_text(text, finetuning, add_soft_prompt)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


finetuning = True
best_val = 0
for i in range(args.epochs):
    # EVALUATE
    if i >= args.pretraining_epochs:
        finetuning = False
    if i != 0:
        correct = 0
        ranks = defaultdict(int)
        for img_dir, img_idx, text in tqdm.tqdm(valid):
            img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
            img_files = sorted(
                img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
            )
            images = [Image.open(photo_file) for photo_file in img_files]
            images = torch.stack([preprocess(photo) for photo in images]).to(device)
            with torch.no_grad():
                img_embs = encode_images(images)
                text = clip.tokenize(text.strip(), truncate=True).to(device)
                text_emb = encode_text(text, add_soft_prompt=(not finetuning))
            ranked_idx, sim = find_best_matches(text_emb, img_embs)
            ranked_files = [
                str(img_files[rank]).split("/")[-1][:-4] for rank in ranked_idx
            ]
            target = str(img_files[int(img_idx)]).split("/")[-1][:-4]
            if ranked_files[0] == target:
                correct += 1
            ranks[ranked_files.index(target) + 1] += 1
        print(correct, flush=True)
        print(len(valid), flush=True)
        print(ranks, flush=True)
        acc = correct / len(valid)
        print(f"acc {acc}", flush=True)
        wandb.log({"val_acc": acc})
        if acc > best_val:
            best_val = acc
            string = ""
            for key, val in list(vars(args).items()):
                if "path" not in key:
                    string += f"_{val}"
            torch.save(
                {
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoints/CONTRA_clip_best_{string.replace('/', '')}.pt",
            )
        print("------------------------------", flush=True)

    print(f"EPOCH: {i}", flush=True)
    step = 0
    random.shuffle(train)
    for img_dir, img_idx, text in train:
        step += 1
        text = [text]
        img_idx = int(img_idx)
        img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
        img_files = sorted(
            img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
        )
        images = [Image.open(photo_file) for photo_file in img_files]
        images = torch.stack([preprocess(photo) for photo in images]).to(device)
        text = clip.tokenize(text, truncate=True).to(device)
        logits_per_image, logits_per_text = model_forward(
            images, text, finetuning=finetuning, add_soft_prompt=(not finetuning)
        )
        # the index of the correct one
        ground_truth = torch.tensor([img_idx]).long().to(device)
        loss = loss_txt(logits_per_text, ground_truth)
        loss.backward()
        if step % config.batchsize == 0:  # fix
            print("STEP: " + str(step), flush=True)
            print(f"TOTAL LOSS: {loss}", flush=True)
            wandb.log({"loss": loss})
            if device == "cpu":
                optimizer.step()
            else:
                optimizer.step()
            optimizer.zero_grad()
