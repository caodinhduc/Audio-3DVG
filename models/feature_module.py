import torch
import torch.nn as nn
import json
import ast
import numpy as np
from models.obj_encoder import PcdObjEncoder


class FeatureModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_encoder = PcdObjEncoder()
        with open("label2vect.json", "r") as f:
            self.text_encoder = json.load(f)
        self.MAX_NUM_OBJECT = 8
    def forward(self, data_dict):
        batch_size = len(data_dict['instance_points'])
        bts_candidate_point = []
        bts_candidate_obbs = []
        bts_candidate_mask = []
        
        bts_relation_point = []
        bts_relation_obbs = []
        bts_relation_mask = []
        for i in range(batch_size):
            candidate_point = []
            candidate_obbs = []
            candidate_mask = []
        
            relation_point = []
            relation_obbs = []
            relation_mask = []
            instance_point = data_dict['instance_points'][i]
            instance_obb = data_dict['instance_obbs'][i]
            instance_class = data_dict['instance_class'][i]
            # num_obj = len(instance_point)

            audio_feature = data_dict['audio_feature'][i]
            audio_class = data_dict['audio_class'][i]
            nel_label = data_dict['nel_label'][i]
            nel_label = ast.literal_eval(nel_label)

            for idx, i_class in enumerate(instance_class):
                # i_class = str(i_class)
                if i_class in nel_label:
                    if i_class == audio_class:
                        candidate_point.append(instance_point[idx][:, :6].tolist())
                        candidate_obbs.append(instance_obb[idx][:6].tolist())
                        candidate_mask.append(1)
                    else:
                        relation_point.append(instance_point[idx][:, :6].tolist())
                        relation_obbs.append(instance_obb[idx][:6].tolist())
                        relation_mask.append(1)
            # filtering to MAX OBJECT
            while len(candidate_point) < 8:
                candidate_point.append(np.zeros((1024, 6)).tolist())
                candidate_obbs.append(np.zeros(6).tolist())
                candidate_mask.append(0)
            while len(relation_point) < 8:
                relation_point.append(np.zeros((1024, 6)).tolist())
                relation_obbs.append(np.zeros(6).tolist())
                relation_mask.append(0)
            print('hih')

        return data_dict