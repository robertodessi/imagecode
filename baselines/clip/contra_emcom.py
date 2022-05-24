# inspired from: https://github.com/openai/CLIP/issues/83
# https://github.com/openai/CLIP/issues/83

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image

import clip
import torch
import tqdm

import wandb
from torch import nn, optim

from egg.core.gs_wrappers import RelaxedEmbedding, gumbel_softmax_sample as gs

random.seed(10)
torch.manual_seed(10)
wandb.init(project="finetune-clip", settings=wandb.Settings(start_method="fork"))


def encode_images(photos_batch):
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(
        device
    )

    with torch.no_grad():
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)
    return photos_features  # .cpu().numpy()


def encode_text(model, search_query, emergent_text):
    with torch.no_grad():
        text_encoded = custom_encode_text(
            model, clip.tokenize(search_query, truncate=True).to(device), emergent_text
        )
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
    return text_encoded  # .cpu().numpy()


def get_first_pad_idx(messages: torch.Tensor) -> torch.Tensor:
    max_k = messages.size(1)
    zero_mask = messages == 0

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


def custom_encode_text(model, text, emergent_text):
    x = model.token_embedding(text).type(model.dtype)  # [batch_size, n_ctx, d_model]

    x = x + model.positional_embedding.type(model.dtype)

    # fn returns 1-indexed values, we skip one and use -3 because in idx -2 there's the eot token
    idx = get_first_pad_idx(text) - 3
    x[0, idx] = emergent_text.half()

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = model.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x).type(model.dtype)

    # x.shape = [batch_size, n_ctx, transformer.width]
    # take features from the eot embedding (eot_token is the highest number in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ model.text_projection

    return x


def custom_forward(model, image_features, text, emergent_text):
    text_features = custom_encode_text(model, text, emergent_text)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # cosine similarity as logits
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_image, logits_per_text


def get_clip_embeddings(
    pretrained_embeddings: torch.Tensor,
    freeze_embeddings: bool = False,
    max_vocab: int = None,
    data_path: str = "/private/home/rdessi/imagecode/data/",
):
    data_path = Path(data_path)

    assert max_vocab is None or max_vocab > 0

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

    idx2clip_idx = {idx: clip_idx for idx, clip_idx in enumerate(most_freq_tokens)}
    return (
        RelaxedEmbedding.from_pretrained(
            pretrained_embeddings.weight[most_freq_tokens],
            freeze=freeze_embeddings,
        ),
        idx2clip_idx,
    )


class SymbolSender(nn.Module):
    def __init__(
        self,
        agent: nn.Module,
        embedding: nn.Module,
        temperature: float = 1.0,
        straight_through: bool = False,
        **kwargs,
    ):
        super(SymbolSender, self).__init__()
        self.agent = agent
        self.embedding = embedding

        self.straight_through = straight_through
        self.temperature = temperature

    def forward(self, image_features):
        x = self.agent(image_features)
        message = gs(x, self.temperature, self.training, self.straight_through)
        return self.embedding(message), message.detach()


class DeepSender(nn.Module):
    def __init__(self, input_dim, vocab_size):
        super(DeepSender, self).__init__()
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, image_feats):
        return self.fc(image_feats)


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
    "--imgs_path", type=str, default="/network/scratch/b/benno.krojer/dataset/games"
)
parser.add_argument("--job_id")
parser.add_argument("--vocab_size", type=int, default=2000)
parser.add_argument("--gs_temperature", type=float, default=1.0)
parser.add_argument("--straight_through", action="store_true", default=False)

args = parser.parse_args()
wandb.config.update(args)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"DEVICE USED: {device}")
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
wandb.watch(model)
if device == "cpu":
    model.float()
else:
    # Actually this line is unnecessary since clip by default already on float16
    clip.model.convert_weights(model)

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

tokenizer = clip.simple_tokenizer.SimpleTokenizer()

embedding, idx2clip_idx = get_clip_embeddings(
    model.token_embedding, False, args.vocab_size
)
sender = DeepSender(model.visual.output_dim, args.vocab_size)
sender = SymbolSender(sender, embedding, args.gs_temperature, args.straight_through)
sender.to(device)

optimizer = optim.Adam(
    [{"params": model.parameters()}, {"params": sender.parameters()}],
    lr=args.lr,
    betas=(0.9, 0.98),
    eps=1e-6,
    weight_decay=0.2,
)

best_val = 0
for i in range(args.epochs):
    # EVALUATE
    if i != 0:
        sender.eval()
        correct = 0
        ranks = defaultdict(int)
        final_messages = []
        for img_dir, img_idx, text in tqdm.tqdm(valid):
            img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
            img_files = sorted(
                img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
            )
            img_embs = encode_images(img_files).float()
            with torch.no_grad():
                emergent_text, idxs = sender(img_embs[img_idx])

            clip_idx = idx2clip_idx[idxs[img_idx].argmax(dim=-1).item()]
            emergent_token = tokenizer.decode([clip_idx])
            final_messages.append("/".join([text, emergent_token]))

            text_emb = encode_text(model, text.strip(), emergent_text).float()
            ranked_idx, sim = find_best_matches(text_emb, img_embs)
            ranked_files = [
                str(img_files[rank]).split("/")[-1][:-4] for rank in ranked_idx
            ]
            target = str(img_files[int(img_idx)]).split("/")[-1][:-4]
            if ranked_files[0] == target:
                correct += 1
            ranks[ranked_files.index(target) + 1] += 1
        print(correct)
        print(len(valid))
        print(ranks)
        acc = correct / len(valid)
        wandb.log({"val_acc": acc})
        if acc > best_val:
            best_val = acc
            string = ""
            for key, val in list(vars(args).items()):
                if "path" not in key:
                    string += f"_{val}"
            torch.save(
                {
                    "final_messages": final_messages,
                    "epoch": i,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoints/CONTRA_clip_best_{string.replace('/', '')}.pt",
            )
        print("------------------------------")

    print(f"EPOCH: {i}")
    sender.train()
    step = 0
    random.shuffle(train)
    for img_dir, img_idx, caption in train:
        step += 1
        text = [caption]
        img_idx = int(img_idx)
        img_files = list((Path(img_dirs) / img_dir).glob("*.jpg"))
        img_files = sorted(
            img_files, key=lambda x: int(str(x).split("/")[-1].split(".")[0][3:])
        )
        images = [Image.open(photo_file) for photo_file in img_files]
        images = torch.stack([preprocess(photo) for photo in images]).to(device)
        text = clip.tokenize(text, truncate=True).to(device)

        image_features = model.encode_image(images)

        emergent_text, idxs = sender(image_features.float())

        clip_idx = idx2clip_idx[idxs[img_idx].argmax(dim=-1).item()]
        emergent_token = tokenizer.decode([clip_idx])

        logits_per_image, logits_per_text = custom_forward(
            model, image_features, text, emergent_text[img_idx]
        )
        # the index of the correct one
        ground_truth = torch.tensor([img_idx]).long().to(device)
        loss = loss_txt(logits_per_text, ground_truth)
        loss.backward()
        if step % args.batchsize == 0:
            print("STEP: " + str(step))
            print(f"TOTAL LOSS: {loss}")
            wandb.log({"loss": loss})
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)
            optimizer.zero_grad()
