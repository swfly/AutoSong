# scripts/train_composer_cluster.py
"""Unsupervised cluster‑based pre‑training for the Composer.

Steps
-----
1.  Scan every song directory with Whisper‑aligned `aligned_lyrics.json`.
2.  For each sentence:  
    • convert text → pinyin token IDs (using the same vocabulary).  
    • pool the *frozen* word embeddings from `Composer.text_emb` → sentence‑level vector.  
    • save triples (song_id, sentence_embed, block_idx).
3.  Fit MiniBatchKMeans over all sentence embeddings (k = n_roles).
4.  Use cluster IDs as *pseudo labels* for every 1‑second *block* that overlaps the sentence.
5.  Train the Composer for two epochs:
    • **Role CE** between `role_logits` and pseudo label.  
    • **Duration CE** (optional) from Whisper `end‑start` length bucket.  
    • **Masked‑pinyin reconstruction** (10 % blocks) to keep phoneme head awake.

Run
---
$ python scripts/train_composer_cluster.py --dataset dataset/ \
        --blocks 180 --roles 8 --epochs 10 --lr 3e-4
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.composer_decoder import Composer
from models.text_encoder import TextEncoder  # same vocab / tokeniser

# ─────────────────────────── dataset helpers ────────────────────────────
class SentenceBlock:
    """Container for one Whisper sentence and the blocks it spans."""

    def __init__(self, embed: torch.Tensor, block_ids: List[int], dur_slot: int):
        self.embed      = embed  # (d,)
        self.block_ids  = block_ids
        self.dur_slot   = dur_slot

class ComposerDataset(Dataset):
    """Provides (lyric_ids, duration_slots, pseudo_role_ids) per track."""

    def __init__(self, root: str, text_enc: TextEncoder, block_num: int, n_dur_slots: int, kmeans: MiniBatchKMeans):
        super().__init__()
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        d_model = text_enc.encode("啊").shape[-1]  # dummy to get device later

        for song_dir in sorted(os.listdir(root)):
            aligned_path = Path(root) / song_dir / "aligned_lyrics.json"
            if not aligned_path.exists():
                continue
            with open(aligned_path, encoding="utf-8") as f:
                segments = json.load(f)

            # build track‑level arrays
            lyric_tokens = torch.full((block_num, text_enc.max_tokens), text_enc.pad_id, dtype=torch.long)
            duration_ids = torch.zeros(block_num, dtype=torch.long)
            role_ids     = torch.zeros(block_num, dtype=torch.long)

            for seg in segments:
                start_s, end_s, text = seg["start"], seg["end"], seg["text"]
                dur = end_s - start_s
                dur_bucket = min(n_dur_slots - 1, int(dur / 0.25))  # 0‑0.25 → 0, ...

                # sentence embedding
                token_ids = text_enc.encode(text)[0]                 # (T,)
                embed_vec = text_enc.encode(text).float().mean(1)   # naive mean of word‑emb
                cluster_id = kmeans.predict(embed_vec.unsqueeze(0).numpy())[0]

                # map to block indices
                blk_start = int(start_s * block_num / 180)
                blk_end   = int(end_s   * block_num / 180)
                blk_end   = min(block_num - 1, blk_end)
                for b in range(blk_start, blk_end + 1):
                    lyric_tokens[b] = token_ids
                    duration_ids[b] = dur_bucket
                    role_ids[b]     = cluster_id

            self.samples.append((lyric_tokens, duration_ids, role_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ─────────────────────────── training loop ──────────────────────────────
@torch.inference_mode()
def gather_sentence_embeddings(root: str, text_enc: TextEncoder) -> np.ndarray:
    """Return (N_sent, d) numpy array for clustering."""
    embeds = []
    for song_dir in tqdm(sorted(os.listdir(root)), desc="Embeddings"):
        path = Path(root) / song_dir / "aligned_lyrics.json"
        if not path.exists():
            continue
        segs = json.load(open(path, encoding="utf-8"))
        for seg in segs:
            ids = text_enc.encode(seg["text"])
            emb = text_enc.text_emb(ids.to(torch.device("cpu"))).mean(1).squeeze(0)  # (d,)
            embeds.append(emb.numpy())
    return np.stack(embeds, 0)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_enc = TextEncoder(max_tokens=512)

    # 1) Cluster sentences to build pseudo‑role labels -------------------
    cache = Path("kmeans_roles.npy")
    if cache.exists():
        km_state = np.load(cache, allow_pickle=True).item()
        kmeans = MiniBatchKMeans(n_clusters=args.roles)
        kmeans.__dict__.update(km_state)
    else:
        sent_embeds = gather_sentence_embeddings(args.dataset, text_enc)
        kmeans = MiniBatchKMeans(n_clusters=args.roles, batch_size=4096, verbose=1, n_init="auto").fit(sent_embeds)
        np.save(cache, kmeans.__dict__)

    # 2) Build dataset ---------------------------------------------------
    ds = ComposerDataset(args.dataset, text_enc, args.blocks, args.dur_slots, kmeans)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)

    # 3) Model -----------------------------------------------------------
    composer = Composer(block_num=args.blocks, n_roles=args.roles, phoneme_len=args.ph_len).to(device)
    opt = torch.optim.AdamW(composer.parameters(), lr=args.lr, weight_decay=1e-4)

    # 4) Training --------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        composer.train()
        total_ce, total_blocks = 0.0, 0
        for lyric_ids, dur_ids, role_ids in tqdm(dl, desc=f"Epoch {epoch}"):
            lyric_ids, dur_ids, role_ids = (
                lyric_ids.to(device), dur_ids.to(device), role_ids.to(device)
            )
            ph_logits, blank_logits, role_w, _ = composer(lyric_ids, dur_ids)
            role_logits = role_w.log()  # convert weights → logits for CE

            loss_role = F.cross_entropy(
                role_logits.view(-1, args.roles), role_ids.view(-1), ignore_index=0
            )

            opt.zero_grad()
            loss_role.backward()
            opt.step()

            total_ce += loss_role.item() * lyric_ids.size(0)
            total_blocks += lyric_ids.size(0)

        print(f"[Epoch {epoch}] Role‑CE: {total_ce/total_blocks:.4f}")

        # quick ckpt
        if epoch % 2 == 0:
            torch.save({"composer": composer.state_dict()}, args.ckpt)
            print("💾 saved", args.ckpt)


# ──────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--blocks", type=int, default=180)
    p.add_argument("--roles", type=int, default=8)
    p.add_argument("--dur_slots", type=int, default=16)
    p.add_argument("--ph_len", type=int, default=32)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ckpt", default="checkpoints/composer_cluster.pt")
    args = p.parse_args()

    train(args)
