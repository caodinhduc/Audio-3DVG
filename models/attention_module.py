import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class LanguageConditionedCrossAttention(nn.Module):
    def __init__(self, obj_dim, lang_dim, hidden_dim, num_heads=4):
        super(LanguageConditionedCrossAttention, self).__init__()
        self.obj_dim = obj_dim
        self.lang_dim = lang_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Object projections
        self.q_proj = nn.Linear(obj_dim, hidden_dim)
        self.k_proj = nn.Linear(obj_dim, hidden_dim)
        self.v_proj = nn.Linear(obj_dim, hidden_dim)

        # Language-conditioned bias terms
        self.q_lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.k_lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.v_lang_proj = nn.Linear(lang_dim, hidden_dim)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, obj_dim)

    def forward(self, obj_feats, lang_embed):
        """
        obj_feats: (B, N, D)       - object features (batch, num_objects, obj_dim)
        lang_embed: (B, L)         - sentence embedding (batch, lang_dim)
        """
        B, N, _ = obj_feats.shape

        # Project language to same dim as hidden_dim
        q_lang = self.q_lang_proj(lang_embed).unsqueeze(1)  # (B, 1, H)
        k_lang = self.k_lang_proj(lang_embed).unsqueeze(1)
        v_lang = self.v_lang_proj(lang_embed).unsqueeze(1)

        # Project objects
        Q = self.q_proj(obj_feats) + q_lang   # (B, N, H)
        K = self.k_proj(obj_feats) + k_lang
        V = self.v_proj(obj_feats) + v_lang

        # Reshape for multi-head attention
        def reshape(x):
            return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            # (B, num_heads, N, head_dim)

        Q = reshape(Q)
        K = reshape(K)
        V = reshape(V)

        # Scaled dot-product attention
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)  # (B, heads, N, N)

        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Combine heads
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # (B, N, H)

        # Project back to object feature dim
        out = self.out_proj(attended)  # (B, N, obj_dim)

        return out  # language-modulated object features


class TargetToRelationalCrossAttention(nn.Module):
    def __init__(self, target_dim, relational_dim, lang_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Projections
        self.q_proj = nn.Linear(target_dim, hidden_dim)
        self.k_proj = nn.Linear(relational_dim, hidden_dim)
        self.v_proj = nn.Linear(relational_dim, hidden_dim)

        # Language modulation
        self.q_lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.k_lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.v_lang_proj = nn.Linear(lang_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, target_dim)

    def forward(self, targets, relationals, lang_feat):
        """
        targets:     (B, N, d_t) — N target candidate features
        relationals: (B, N, d_r) — N relational object features
        lang_feat:   (B, d_l)    — language sentence embedding
        Returns:
            updated_targets: (B, N, d_t)
        """
        B, N, _ = targets.shape

        # Project language
        q_lang = self.q_lang_proj(lang_feat).unsqueeze(1)  # (B, 1, H)
        k_lang = self.k_lang_proj(lang_feat).unsqueeze(1)
        v_lang = self.v_lang_proj(lang_feat).unsqueeze(1)

        # Project inputs
        Q = self.q_proj(targets) + q_lang  # (B, N, H)
        K = self.k_proj(relationals) + k_lang
        V = self.v_proj(relationals) + v_lang

        # Reshape for multi-head
        def reshape(x):
            return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q = reshape(Q)  # (B, heads, N, head_dim)
        K = reshape(K)
        V = reshape(V)

        # Attention: Q from targets, K/V from relational objects
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_weights = F.softmax(attn_logits, dim=-1)
        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)

        # Merge heads
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # (B, N, H)

        # Final projection
        updated_targets = self.out_proj(attended)  # (B, N, d_t)

        return updated_targets


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
                    nn.Linear(512, 1)
                    )
        self.target_projector = nn.Sequential(
                    nn.Linear(562, 512)
                    )
        
        self.relation_projector = nn.Sequential(
                    nn.Linear(562, 512)
                    )
        self.attn_layer = LanguageConditionedCrossAttention(obj_dim=512, lang_dim=1024, hidden_dim=512)
        self.cross_attn = TargetToRelationalCrossAttention(512, 512, 1024, hidden_dim=512)
    def forward(self, data_dict):
        
        target_representation = data_dict["target_representation"] # B x 16 x 1074
        relation_representation = data_dict["relation_representation"] # B x 16 x 1074
        bts_audio_feature = data_dict["bts_audio_feature"].squeeze(1) # B x 758

        target_representation = self.target_projector(target_representation)
        relation_representation = self.target_projector(relation_representation)
        # bts_candidate_mask = data_dict["bts_candidate_mask"] # B x 16 

        attns = self.attn_layer(target_representation, bts_audio_feature)
        cross_attns = self.cross_attn(target_representation, relation_representation, bts_audio_feature)

        scores = self.fc(target_representation + attns + cross_attns)
        data_dict['score'] = scores
        # concatenate
        return data_dict