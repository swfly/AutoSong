# scripts/test_composer.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
from models.composer import Composer
from models.text_encoder import TextEncoder
import os
import numpy as np

# Configuration
phoneme_per_sentence = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_blocks = 180  # 最大生成块数防止无限循环

# 加载模型
def load_model(checkpoint_path):
    composer = Composer(max_sentence_len=phoneme_per_sentence).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    composer.load_state_dict(checkpoint['model_state_dict'])
    composer.eval()
    return composer

# 自回归生成函数
def autoregressive_generate(composer, text_prompt, text_encoder, device):
    # 初始化空块
    B = 1  # batch size
    current_blocks = {
        'sentences': torch.zeros(B, 1, phoneme_per_sentence), 
        'empty_tags': torch.zeros((B, 1, 1)),
        'lengths': torch.ones((B, 1, 1)), 
        'roles': torch.zeros((B, 1, 1))
    }
    
    # 文本提示编码
    text_tokens = TextEncoder().encode(text_prompt).to(device)
    text_ids = text_tokens
    generated_blocks = []
    
    for _ in range(max_blocks):
        # 准备输入张量
        inputs = (
            text_ids,
            current_blocks['sentences'].long().to(device),
            current_blocks['empty_tags'].float().to(device),
            current_blocks['lengths'].long().to(device),
            current_blocks['roles'].long().to(device)
        )
        
        # 前向传播
        with torch.no_grad():
            token_logits, empty_logits, length_logits, role_logits = composer(*inputs)
        
        # 获取最新块的预测（最后一个位置）
        new_token = torch.argmax(token_logits[:, -1:], dim=-1)  # (B,1,T)
        new_empty = torch.sigmoid(empty_logits[:, -1:]) > 0.5  # (B,1)
        new_length = torch.argmax(length_logits[:, -1:], dim=-1)  # (B,1)
        new_role = torch.argmax(role_logits[:, -1:], dim=-1)  # (B,1)
        
        print(new_token)
        print(new_empty)
        print(new_length)
        print(new_role)
        # 停止条件
        # if new_empty.item():
        #     break
            
        # 更新当前块
        current_blocks['sentences'] = torch.cat([
            current_blocks['sentences'], 
            new_token.cpu()], dim=1)
        current_blocks['empty_tags'] = torch.cat([
            current_blocks['empty_tags'], 
            new_empty.float().cpu()], dim=1)
        current_blocks['lengths'] = torch.cat([
            current_blocks['lengths'], 
            new_length.cpu().unsqueeze(0)], dim=1)
        current_blocks['roles'] = torch.cat([
            current_blocks['roles'], 
            new_role.cpu().unsqueeze(0)], dim=1)
        
        # 解码当前块
        decoded_block = text_encoder.decode(new_token[0,0].cpu().numpy())
        generated_blocks.append(decoded_block)
    
    return generated_blocks

# 测试函数
def test_model(checkpoint_path, lyrics_path):
    # 初始化组件
    composer = load_model(checkpoint_path)
    text_encoder = TextEncoder(phoneme_per_sentence)
    
    # 加载歌词
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        lyrics = f.read()
    
    # 生成文本块
    blocks = autoregressive_generate(composer, lyrics, text_encoder, device)
    
    print("\nGenerated Structure:")
    for i, blk in enumerate(blocks):
        print(f"Block {i+1}: {blk}")

# 运行测试
test_model(checkpoint_path="checkpoints/composer.pt", lyrics_path="dataset/song_001/song_001.txt")