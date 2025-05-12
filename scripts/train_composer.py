# scripts/train_composer.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from models.composer import Composer
from models.text_encoder import TextEncoder  # same vocab / tokeniser

phoneme_per_sentence = 16
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
text_encoder = TextEncoder(phoneme_per_sentence)

# ─────────────────────────── dataset helpers ────────────────────────────
def find_aligned_lyrics_json_and_txt(base_path):
    aligned_lyrics_txt_paths = []
    
    # Walk through all subdirectories of the given base path
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('aligned_lyrics.json'):
                # Get the directory containing the aligned_lyrics.json file
                dir_path = root
                # Find all .txt files in the same directory
                txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                
                # If .txt files are found, add the .json path and corresponding .txt files
                if txt_files:
                    for txt_file in txt_files:
                        txt_path = os.path.join(dir_path, txt_file)
                        aligned_lyrics_txt_paths.append({
                            "json_path": os.path.join(dir_path, file),
                            "txt_path": txt_path
                        })
    
    return aligned_lyrics_txt_paths

files_pairs = find_aligned_lyrics_json_and_txt("dataset")

class TextBlock:
    """Represents a text block that contains lyrics or is empty."""
    
    def __init__(self, text: str = "", is_empty: bool = True, length: float = 0.0):
        self.text = text
        self.text_tokens = text_encoder.encode(text)
        self.is_empty = is_empty  # If this block is empty or contains text
        self.length = int(length)  # Length of the sentence this block belongs to
        
    def __repr__(self):
        return f"TextBlock(text={self.text}, is_empty={self.is_empty}, length={self.length})"

def prepare_text_blocks(json_path: str, block_num: int, n_dur_slots: int) -> List[TextBlock]:
    """
    Prepare text blocks from aligned lyrics JSON and include the required information.
    
    Args:
        json_path: Path to the aligned lyrics JSON.
        block_num: Number of blocks (e.g., 180).
        n_dur_slots: Number of duration slots to bucket the sentence length into.
        
    Returns:
        A list of TextBlock objects, one for each block in the song.
    """
    with open(json_path, encoding="utf-8") as f:
        segments = json.load(f)

    text_blocks = []  # Initialize all blocks as empty by default

    initial_pos = int(segments[0]["start"])
    start_pos = 0
    while True:
        duration = float(random.randint(6,7))
        duration = min(initial_pos-start_pos, duration)
        text_blocks.append(TextBlock(
            text = "",
            is_empty=False,
            length = duration
        ))
        start_pos += duration
        if start_pos >= initial_pos:
            break


    for seg in segments:
        start_s, end_s, text = seg["start"], seg["end"], seg["text"]
        if '李宗盛' in text or '母带' in text:
            text = ""
        if int(start_s) != start_pos:
            target_time = int(start_s)
            while True:
                duration = float(random.randint(6,7))
                duration = min(target_time-start_pos, duration)
                text_blocks.append(TextBlock(
                    text = "",
                    is_empty=False,
                    length = duration
                ))
                start_pos += duration
                if start_pos >= target_time:
                    break
        
        # Calculate sentence duration and relative position for duration bucketing
        dur = end_s - start_s
        
        # Assign block details
        text_blocks.append(TextBlock(
            text=text,  # Here you can store the actual tokenized text if needed
            is_empty=False,
            length=dur
        ))

    text_blocks.append(TextBlock(
        text="",  # Here you can store the actual tokenized text if needed
        is_empty=True,
        length=1
    ))
    text_blocks.append(TextBlock(
        text="",  # Here you can store the actual tokenized text if needed
        is_empty=True,
        length=1
    ))
    while(len(text_blocks) < 150):
        text_blocks.append(TextBlock(
            text="",  # Here you can store the actual tokenized text if needed
            is_empty=True,
            length=1
        ))
    # for b in text_blocks:
    #     print(b)
    # quit()
    return text_blocks


def train_step(composer: Composer, optimizer: torch.optim.Optimizer, input_text, text_blocks: List[TextBlock], device: torch.device):
    """Performs a single training step."""

    class_weights = torch.ones(4911).to(device)
    class_weights[9] = 1e-2
    class_weights[10] = 1e-2
    class_weights[2] = 1e-2
    ce_ph = nn.CrossEntropyLoss(weight=class_weights)
    ce = nn.CrossEntropyLoss()
    bce = nn.CrossEntropyLoss()

    input = input_text.to(device)
    # Prepare teacher data tensors
    text_ids = []          # Placeholder for input phoneme tokens
    blank_labels = []      # Placeholder for the empty flag (binary)
    role_labels = []       # Placeholder for role labels (random initialization)
    duration_labels = []   # Placeholder for duration (quantized)

    # Create the teacher data
    for block in text_blocks:
        # Prepare text (phoneme tokens)
        text_ids.append(block.text_tokens.to(device).reshape(-1))
        
        # Prepare blank (1 = empty, 0 = not empty)
        blank_labels.append(1 if block.is_empty else 0)
        
        # Randomly assign roles initially (can be modified later)
        role_labels.append(random.randint(0, 7))  # Random role between 0 and 7 (n_roles = 8)
        
        # Quantize the duration into slots (1-16 seconds mapped to [0, 15])
        duration_slot = min(max(int(block.length), 1), 16) - 1  # Slot 0 corresponds to 1 second, slot 15 corresponds to 16 seconds
        duration_labels.append(duration_slot)

    # Convert lists to tensors
    text_ids = torch.stack(text_ids, dim=0).to(device).unsqueeze(0)  # (B, T)
    blank_labels = torch.tensor(blank_labels, dtype=torch.float32).to(device).reshape((-1,1)).unsqueeze(0)  # (B,)
    role_labels = torch.tensor(role_labels, dtype=torch.long).to(device).reshape((-1,1)).unsqueeze(0)  # (B,)
    duration_labels = torch.tensor(duration_labels, dtype=torch.long).to(device).reshape((-1,1)).unsqueeze(0)  # (B,)
    # Forward pass
    ph_logits, empty_logits, length_logits, role_logits = composer(input, text_ids, blank_labels, duration_labels, role_labels)

    # Loss calculations
    ph_loss = ce_ph(ph_logits[:, :-1, :].reshape((-1,ph_logits.shape[-1])), text_ids[:, 1:].reshape((-1)))  # ignore padding index
    blank_loss = F.binary_cross_entropy_with_logits(empty_logits[:, :-1, :].reshape((-1)), blank_labels[:, 1:].reshape((-1)))
    # role_loss = ce(role_weights.view(-1, role_weights.size(-1)), role_labels.view(-1))
    duration_loss = ce(length_logits[:, :-1, :].reshape((-1,length_logits.shape[-1])), duration_labels[:, 1:].reshape(-1))
    role_loss = 0.0
    total_loss = ph_loss + blank_loss * 0.1 + role_loss + duration_loss * 0.1
    # Backward pass and optimization
    optimizer.zero_grad()  # Zero out previous gradients
    total_loss.backward()  # Backpropagate gradients
    optimizer.step()       # Update weights
    # print(torch.argmax(ph_logits[:, -1:], dim=-1))

    return total_loss.item(), ph_loss.item(), blank_loss.item(), role_loss, duration_loss.item()

training_data = {}

def train(composer: Composer, optimizer: torch.optim.Optimizer, device: torch.device, num_epochs: int = 10):
    for epoch in range(1, num_epochs):
        random_pair = random.choice(files_pairs)
        if random_pair["json_path"] in training_data:
            text_blocks = training_data[random_pair["json_path"]][0]
            plain_text_tokens = training_data[random_pair["json_path"]][1]
        else:
            text_blocks = prepare_text_blocks(random_pair["json_path"], 180, 16)
            with open(random_pair["txt_path"], encoding="utf-8") as f:
                plain_text = f.read()
                plain_text_tokens = TextEncoder().encode(plain_text)
            training_data[random_pair["json_path"]] = [text_blocks, plain_text_tokens]

        epoch_loss = 0
        ph_loss_total = 0
        blank_loss_total = 0
        role_loss_total = 0
        duration_loss_total = 0
        
        
        # Slice the text blocks into a batch
        loss, ph_loss, blank_loss, role_loss, duration_loss = \
            train_step(composer, optimizer, plain_text_tokens, text_blocks, device)

        # Accumulate the losses
        epoch_loss += loss
        ph_loss_total += ph_loss
        blank_loss_total += blank_loss
        role_loss_total += role_loss
        duration_loss_total += duration_loss

        # Average loss for the epoch
        print(f"Epoch {epoch + 1}/{num_epochs}, Total Loss: {epoch_loss / len(text_blocks):.4f}")
        print(f"Phoneme Loss: {ph_loss_total / len(text_blocks):.4f}, Blank Loss: {blank_loss_total / len(text_blocks):.4f}")
        print(f"Role Loss: {role_loss_total / len(text_blocks):.4f}, Duration Loss: {duration_loss_total / len(text_blocks):.4f}")
        
        if epoch % 200 == 0:
            checkpoint_path = os.path.join("checkpoints", f"composer.pt")
            checkpoint = {
                'epoch': epoch,
                'iteration': epoch,
                'model_state_dict': composer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")


composer = Composer(max_sentence_len=phoneme_per_sentence).to(device)
composer.train()
# Optimizer
optimizer = torch.optim.AdamW(composer.parameters(), lr=2e-5, weight_decay=1e-4)
# Start training
train(composer, optimizer, device, num_epochs=10000)