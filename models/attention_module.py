import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedAttention(nn.Module):
    def __init__(self, dim_a, dim_b, latent_dim=256, heads=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.heads = heads
        self.scale = (latent_dim // heads) ** -0.5

        self.to_q_aa = nn.Linear(dim_a, latent_dim)
        self.to_k_aa = nn.Linear(dim_a, latent_dim)
        
        self.to_v_a = nn.Linear(dim_a, latent_dim)

        self.to_k_ab = nn.Linear(dim_a, latent_dim)
        self.to_q_bb = nn.Linear(dim_b, latent_dim)

        self.out = nn.Linear(latent_dim, latent_dim)

    def forward(self, A, B):
        Bsz, N, _ = B.shape
        H = self.heads
        d_head = self.latent_dim // H

        # Linear projections
        V = self.to_v_a(A)
        Q_self = self.to_q_aa(A)  # [B, N, D]
        K_self = self.to_k_aa(A)
        

        Q_cross = self.to_k_ab(A)  # Cross-attention
        K_cross = self.to_q_bb(B)

        # Reshape for multi-head attention
        def split_heads(x):  # [B, seq, D] -> [B, H, seq, d_head]
            return x.view(Bsz, -1, H, d_head).transpose(1, 2)

        V = split_heads(V)
        Q_self = split_heads(Q_self)
        K_self = split_heads(K_self)
        Q_cross = split_heads(Q_cross)
        K_cross = split_heads(K_cross)

        # Attention scores
        attn_self = torch.matmul(Q_self, K_self.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn_cross = torch.matmul(Q_cross, K_cross.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Pad attn_cross to shape [B, H, N, N] if needed
        # if M != N:
        #     diff = N - M
        #     pad = (0, diff)  # pad last dim
        #     attn_cross = F.pad(attn_cross, pad, "constant", 0)

        # Final attention: element-wise sum
        attn = attn_self + attn_cross

        # Softmax
        attn = F.softmax(attn, dim=-1)
        # print(attn.shape)

        # Again pad V_cross if needed
        # if M != N:
        #     V_cross = F.pad(V_cross, (0, 0, 0, diff), "constant", 0)

        # Combine values
        out_attn = torch.matmul(attn, V)  # [B, H, N, d_head]

        # Merge heads
        out = out_attn.transpose(1, 2).contiguous().view(Bsz, N, self.latent_dim)

        return self.out(out)  # Final projection


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        # self.fc = nn.Sequential(nn.Linear(2098, 1024),
        #                     nn.BatchNorm1d(8),
        #                     nn.ReLU(),
        #                     nn.Linear(1024, 128),
        #                     # nn.BatchNorm1d(128),
        #                     nn.ReLU(),
        #                     nn.Linear(128, 1),
        #                     )
        self.fc = nn.Sequential(nn.Linear(256, 64),
                    nn.BatchNorm1d(8),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    # nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                    )
        self.attn = CombinedAttention(dim_a=2098, dim_b=1074)
        self.MAX_NUM_OBJECT = 8
    def forward(self, data_dict):
        
        target_representation = data_dict["target_representation"] # B x 16 x 1074
        relation_representation = data_dict["relation_representation"] # B x 16 x 1074
        bts_audio_feature = data_dict["bts_audio_feature"] # B x 1 x 1024
        
        # bts_candidate_obbs = data_dict["bts_candidate_obbs"]  # B x 16 x 6
        # bts_relation_obbs = data_dict["bts_relation_obbs"] # B x 16 x 6
        # bts_candidate_mask = data_dict["bts_candidate_mask"] # B x 16 
        # bts_relation_mask = data_dict["bts_relation_mask"] # B x 16

        repeated_bts_audio = bts_audio_feature.repeat(1, self.MAX_NUM_OBJECT, 1)
        final_representation = torch.cat((target_representation, repeated_bts_audio), dim=2)


        ###########
        attn = self.attn(final_representation, relation_representation)
        print(attn.shape)
        ###########


        scores = self.fc(attn)
        data_dict['score'] = scores
        # concatenate
        return data_dict