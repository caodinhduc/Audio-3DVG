import torch
import torch.nn as nn
import json
import ast
import numpy as np
from models.obj_encoder import PcdObjEncoder


class AttentionModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(nn.Linear(2098, 1024),
                            nn.BatchNorm1d(16),
                            nn.ReLU(),
                            nn.Linear(1024, 128),
                            # nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Linear(128, 1),
                            )

    def forward(self, data_dict):
        
        target_representation = data_dict["target_representation"] # B x 16 x 1074
        relation_representation = data_dict["relation_representation"] # B x 16 x 1074
        bts_audio_feature = data_dict["bts_audio_feature"] # B x 1 x 1024
        bts_candidate_obbs = data_dict["bts_candidate_obbs"]  # B x 16 x 6
        bts_relation_obbs = data_dict["bts_relation_obbs"] # B x 16 x 6
        bts_candidate_mask = data_dict["bts_candidate_mask"] # B x 16 
        bts_relation_mask = data_dict["bts_relation_mask"] # B x 16

        repeated_bts_audio = bts_audio_feature.repeat(1, 16, 1)
        final_representation = torch.cat((target_representation, repeated_bts_audio), dim=2)
        scores = self.fc(final_representation)
        # concatenate

        return data_dict