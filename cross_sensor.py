import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        attn_output, _ = self.cross_attention(query, key, value)
        return attn_output

class CrossSensorPredictor(nn.Module):
    def __init__(self, embed_dim = 768, num_heads = 12):
        super(CrossSensorPredictor, self).__init__()
        self.cross_attention_s1_to_s2 = CrossAttention(embed_dim, num_heads)
        self.cross_attention_s2_to_s1 = CrossAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, s1, s2):
        # Apply cross attention: s1 as query, s2 as key and value
        s1_to_s2 = self.cross_attention_s1_to_s2(s1, s2, s2)
        # print('s1 shape: ', s1.shape)
        # print('s2 shape: ', s1.shape)
        s1_to_s2 = self.mlp(s1_to_s2)
        # print('s1_to_s2 shape: ', s1_to_s2.shape)

        # Apply cross attention: s2 as query, s1 as key and value
        s2_to_s1 = self.cross_attention_s2_to_s1(s2, s1, s1)
        s2_to_s1 = self.mlp(s2_to_s1)
        # print('s2_to_s1 shape: ', s2_to_s1.shape)

        return s1_to_s2, s2_to_s1
