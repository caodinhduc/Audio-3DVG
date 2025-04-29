import copy

import torch
import torch.nn as nn
# import einops

from models.backbone.point_net_pp import PointNetPP

# class GTObjEncoder(nn.Module):
#     def __init__(self, config, hidden_size):
#         super().__init__()
#         self.config = copy.deepcopy(config)
#         self.config.hidden_size = hidden_size

#         if self.config.onehot_ft:
#             self.ft_linear = [nn.Embedding(self.config.num_obj_classes, self.config.hidden_size)]
#         else:
#             self.ft_linear = [nn.Linear(self.config.dim_ft, self.config.hidden_size)]
#         self.ft_linear.append(nn.LayerNorm(self.config.hidden_size))
#         self.ft_linear = nn.Sequential(*self.ft_linear)

#         self.dropout = nn.Dropout(self.config.dropout)

#     def forward(self, obj_fts):
#         '''
#         Args:
#             obj_fts: LongTensor (batch, num_objs), or, FloatTensor (batch, num_objs, dim_ft)
#             obj_locs: FloatTensor (batch, num_objs, dim_loc)
#         '''
#         obj_embeds = self.ft_linear(obj_fts)
#         obj_embeds = self.dropout(obj_embeds)
#         return obj_embeds

class PcdObjEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa_n_points =  [32, 16, None]
        self.sa_n_samples = [32, 32, None]
        self.sa_radii = [0.2, 0.4, None]
        self.sa_mlps = [[3, 64, 64, 128], [128, 128, 128, 256], [256, 256, 512, 768]]

        self.pcd_net = PointNetPP(
            sa_n_points=self.sa_n_points,
            sa_n_samples=self.sa_n_samples,
            sa_radii=self.sa_radii,
            sa_mlps=self.sa_mlps,
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, obj_pcds):
        batch_size, num_objs, _, _ = obj_pcds.size()
        # obj_embeds = self.pcd_net(
        #     einops.rearrange(obj_pcds, 'b o p d -> (b o) p d')
        # )
        # obj_embeds = einops.rearrange(obj_embeds, '(b o) d -> b o d', b=batch_size)

        # TODO: due to the implementation of PointNetPP, this way consumes less GPU memory
        obj_embeds = []
        for i in range(batch_size):
            obj_embeds.append(self.pcd_net(obj_pcds[i]))
        obj_embeds = torch.stack(obj_embeds, 0)

        # obj_embeds = []
        # for i in range(num_objs):
        #     obj_embeds.append(self.pcd_net(obj_pcds[:, i]))
        # obj_embeds = torch.stack(obj_embeds, 1)

        obj_embeds = self.dropout(obj_embeds)
        return obj_embeds

        