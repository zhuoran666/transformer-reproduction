

import torch
import torch.nn as nn
import math
from src.dataset import PAD_IDX

#位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # 增加batch维度
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

#相对位置
class RelativePositionalBias(nn.Module):
    def __init__(self, num_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
#核心逻辑：将相对距离分桶，近距离精确分桶，远距离对数分桶
    def _relative_position_bucket(self, relative_position):
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        
        n = -relative_position
        n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact
        

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(n, num_buckets - 1))
        
        return torch.where(is_small, n, val_if_large)

    def forward(self, seq_len, device):
        q_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        k_pos = torch.arange(seq_len, dtype=torch.long, device=device)
        relative_position = k_pos[None, :] - q_pos[:, None] #[seq_len, seq_len]
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket) 
        values = values.permute(2, 0, 1).unsqueeze(0) # [1, num_heads, seq_len, seq_len]
        return values

#多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,dropout: float = 0.1,use_rpe=False,max_len=512):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.use_rpe = use_rpe
        if self.use_rpe:
            self.relative_positional_bias = RelativePositionalBias(num_heads, max_distance=max_len)
        self.register_buffer('scale', torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32)))

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        seq_len = query.shape[1]
        #投影分割
        Q = self.fc_q(query).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.fc_k(key).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.fc_v(value).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if self.use_rpe:
            relative_position_bias = self.relative_positional_bias(seq_len, query.device)
            energy += relative_position_bias
        #掩码
        if mask is not None:
            dtype = energy.dtype
            finfo = torch.finfo(dtype)
            energy = energy.masked_fill(mask == True, finfo.min)

        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.d_model)
        x = self.fc_o(x)
        
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(p=dropout), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.net(x)
#编码层
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,use_residual=True, use_rpe=False, max_len=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rpe, max_len)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.use_residual = use_residual
    def forward(self, src, src_mask):
        attn_output = self.self_attn(src, src, src, src_mask)
        if self.use_residual:
            src = self.norm1(src + self.dropout(attn_output))
        else:
            src = self.norm1(self.dropout(attn_output)) # 无残差
        
        ff_output = self.feed_forward(src)
        if self.use_residual:
            src = self.norm2(src + self.dropout(ff_output))
        else:
            src = self.norm2(self.dropout(ff_output)) # 无残差
        return src
#解码层
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1,use_residual=True, use_rpe=False, max_len=512):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rpe, max_len)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout, use_rpe=False)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.use_residual = use_residual
    def forward(self, tgt, enc_src, tgt_mask, src_mask):

        self_attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)

        if self.use_residual:
            tgt = self.norm1(tgt + self.dropout(self_attn_output))
        else:
            tgt = self.norm1(self.dropout(self_attn_output))
        cross_attn_output = self.cross_attn(tgt, enc_src, enc_src, src_mask)
        if self.use_residual:
            tgt = self.norm2(tgt + self.dropout(cross_attn_output))
        else:
            tgt = self.norm2(self.dropout(cross_attn_output))
        ff_output = self.feed_forward(tgt)
        if self.use_residual:
            tgt = self.norm3(tgt + self.dropout(ff_output))
        else:
            tgt = self.norm3(self.dropout(ff_output))
        return tgt

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, d_model, nhead,
                 src_vocab_size, tgt_vocab_size, dim_feedforward, dropout,use_positional_encoding=True,use_residual=True,use_rpe=False, max_len=512):
        super().__init__()
        self.d_model = d_model
        
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.use_positional_encoding = use_positional_encoding # 2. 保存状态
        if self.use_positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model, dropout)

        #传递参数
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, dim_feedforward, dropout,use_residual,use_rpe, max_len) for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout,use_residual,use_rpe, max_len) for _ in range(num_decoder_layers)
        ])

        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self._initialize_weights()
#独立的encode和decode
    def encode(self, src, src_key_padding_mask):

        src_mask = self.make_src_mask(src_key_padding_mask)
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        
        if self.use_positional_encoding:
            src_emb = self.positional_encoding(src_emb)
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
        return enc_output

    def decode(self, tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask):
        dec_tgt_mask = self.make_tgt_mask(tgt_mask, tgt_key_padding_mask)
        dec_memory_mask = self.make_src_mask(memory_key_padding_mask)
        
        #tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        if self.use_positional_encoding:
            tgt_emb = self.positional_encoding(tgt_emb)
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, memory, dec_tgt_mask, dec_memory_mask)
        return dec_output



    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    #构建掩码
    def make_src_mask(self, src_key_padding_mask):
        return src_key_padding_mask.unsqueeze(1).unsqueeze(2) if src_key_padding_mask is not None else None
    def make_tgt_mask(self, tgt_causal_mask, tgt_key_padding_mask):
        if tgt_key_padding_mask is not None:
            tgt_padding_mask = tgt_key_padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            tgt_padding_mask = None
        if tgt_causal_mask is None:
            return tgt_padding_mask
        if tgt_padding_mask is None:
            return tgt_causal_mask.unsqueeze(0)
        return tgt_causal_mask.unsqueeze(0) | tgt_padding_mask
    
    def forward(self, src, tgt, src_mask, tgt_mask,
                src_key_padding_mask, tgt_key_padding_mask,
                memory_key_padding_mask):
        #构建掩码
        enc_src_mask = self.make_src_mask(src_key_padding_mask)
        dec_tgt_mask = self.make_tgt_mask(tgt_mask, tgt_key_padding_mask)
        dec_memory_mask = self.make_src_mask(memory_key_padding_mask)
        #嵌入和位置编码
        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        if self.use_positional_encoding:
            src_emb = self.positional_encoding(src_emb)
            tgt_emb = self.positional_encoding(tgt_emb)
        #编码解码过程
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, enc_src_mask)
        dec_output = tgt_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, dec_tgt_mask, dec_memory_mask)
        return self.generator(dec_output)
#掩码生成函数
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1) == 1
    return mask
def create_mask(src, tgt, device):
    tgt_causal_mask = generate_square_subsequent_mask(tgt.shape[1], device)
    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return None, tgt_causal_mask, src_padding_mask, tgt_padding_mask