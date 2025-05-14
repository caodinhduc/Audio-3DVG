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
        self.TEXT_EMBEDDING_DIM = 300


    def forward(self, data_dict):
        batch_size = len(data_dict['instance_points'])
        bts_candidate_point = []
        bts_candidate_obbs = []
        bts_candidate_mask = []
        
        bts_relation_point = []
        bts_relation_obbs = []
        bts_relation_mask = []
        
        bts_candidate_label_embedding = []
        bts_relation_label_embedding = []
        bts_audio_feature = []
        
        for i in range(batch_size):
            candidate_point = []
            candidate_obbs = []
            candidate_mask = []
            candidate_label_embedding = []
        
            relation_point = []
            relation_obbs = []
            relation_mask = []
            relation_label_embedding = []
            instance_point = data_dict['instance_points'][i]
            instance_obb = data_dict['instance_obbs'][i]
            instance_class = data_dict['instance_class'][i]
            # num_obj = len(instance_point)

            audio_feature = data_dict['embedded_audio'][i]
            bts_audio_feature.append(audio_feature)
            
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
                        candidate_label_embedding.append(self.text_encoder[str(i_class)])
                    else:
                        relation_point.append(instance_point[idx][:, :6].tolist())
                        relation_obbs.append(instance_obb[idx][:6].tolist())
                        relation_mask.append(1)
                        relation_label_embedding.append(self.text_encoder[str(i_class)])
            # filtering to MAX OBJECT
            if len(candidate_point) > self.MAX_NUM_OBJECT:
                candidate_point = candidate_point[:self.MAX_NUM_OBJECT]
                candidate_obbs = candidate_obbs[:self.MAX_NUM_OBJECT]
                candidate_mask = candidate_mask[:self.MAX_NUM_OBJECT]
                candidate_label_embedding = candidate_label_embedding[:self.MAX_NUM_OBJECT]
            if len(relation_point) > self.MAX_NUM_OBJECT:
                relation_point = relation_point[:self.MAX_NUM_OBJECT]
                relation_obbs = relation_obbs[:self.MAX_NUM_OBJECT]
                relation_mask = relation_mask[:self.MAX_NUM_OBJECT]
                relation_label_embedding = relation_label_embedding[:self.MAX_NUM_OBJECT]
            # filtering to MAX OBJECT
            while len(candidate_point) < self.MAX_NUM_OBJECT:
                candidate_point.append(np.zeros((1024, 6)).tolist())
                candidate_obbs.append(np.zeros(6).tolist())
                candidate_mask.append(0)
                candidate_label_embedding.append(np.zeros(300).tolist())
            while len(relation_point) < self.MAX_NUM_OBJECT:
                relation_point.append(np.zeros((1024, 6)).tolist())
                relation_obbs.append(np.zeros(6).tolist())
                relation_mask.append(0)
                relation_label_embedding.append(np.zeros(300).tolist())

            bts_candidate_point.append(candidate_point)
            bts_candidate_obbs.append(candidate_obbs)
            bts_candidate_mask.append(candidate_mask)
            bts_candidate_label_embedding.append(candidate_label_embedding)

            bts_relation_point.append(relation_point)
            bts_relation_obbs.append(relation_obbs)
            bts_relation_mask.append(relation_mask)
            bts_relation_label_embedding.append(relation_label_embedding)

        
        bts_candidate_point = torch.tensor(bts_candidate_point).cuda()
        bts_relation_point = torch.tensor(bts_relation_point).cuda()

        bts_candidate_obbs = torch.tensor(bts_candidate_obbs).cuda() # B x self.MAX_NUM_OBJECT x 6
        bts_relation_obbs = torch.tensor(bts_relation_obbs).cuda()  # B x self.MAX_NUM_OBJECT x 6

        bts_candidate_mask = torch.tensor(bts_candidate_mask).cuda() # B x self.MAX_NUM_OBJECT
        bts_relation_mask = torch.tensor(bts_relation_mask).cuda() # B x self.MAX_NUM_OBJECT

        bts_candidate_label_embedding = torch.tensor(bts_candidate_label_embedding).cuda() # B x self.MAX_NUM_OBJECT x 300
        bts_relation_label_embedding = torch.tensor(bts_relation_label_embedding).cuda() # B x self.MAX_NUM_OBJECT x 300

        bts_audio_feature = torch.stack(bts_audio_feature).cuda() # B x 1 x 1024

        # input_pointnet = torch.cat([bts_candidate_point, bts_relation_point], dim=1).cuda()
        object_encoding = self.object_encoder(torch.cat([bts_candidate_point, bts_relation_point], dim=1)) # B x 32 x 768

        target_representation = torch.cat((bts_candidate_obbs, bts_candidate_label_embedding, object_encoding[:, :self.MAX_NUM_OBJECT, :]), dim=2)
        relation_representation = torch.cat((bts_relation_obbs, bts_relation_label_embedding, object_encoding[:, self.MAX_NUM_OBJECT:, :]), dim=2)
        data_dict["target_representation"] = target_representation # B x self.MAX_NUM_OBJECT x 1074
        data_dict["relation_representation"] = relation_representation # B x self.MAX_NUM_OBJECT x 1074
        data_dict["bts_audio_feature"] = bts_audio_feature # B x self.MAX_NUM_OBJECT x 1074
        data_dict["bts_candidate_obbs"] = bts_candidate_obbs # B x self.MAX_NUM_OBJECT x 6
        data_dict["bts_relation_obbs"] = bts_relation_obbs
        data_dict["bts_candidate_mask"] = bts_candidate_mask
        data_dict["bts_relation_mask"] = bts_relation_mask
        return data_dict