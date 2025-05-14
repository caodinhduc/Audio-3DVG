import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.getcwd(), "lib"))  # HACK add the lib folder
from utils.box_util import get_3d_box_batch, box3d_iou_batch

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3  # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2, 0.8]  # put larger weights on positive objectness
MAX_NUM_OBJECT = 8


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2, gamma=5, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.gamma = gamma
        self.reduction = reduction
        self.soft_plus = nn.Softplus()

    def forward(self, score, label):
        score *= self.gamma
        sim = (score*label).sum()
        neg_sim = score*label.logical_not()
        neg_sim = torch.logsumexp(neg_sim, dim=0) # soft max
        loss = torch.clamp(neg_sim - sim + self.margin, min=0).sum()
        return loss

class TargetClassificationLoss(nn.Module):
    def __init__(self):
        super(TargetClassificationLoss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, gt, pred):
        return self.criterion(gt, pred)


class SegLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(SegLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt)**self.gamma) * self.alpha * logpt
        return loss


def compute_scene_mask_loss(data_dict):
    pred = data_dict['seg_scores']
    ref_center_label = data_dict["ref_center_label"].cuda()

    point_min = data_dict['point_min']
    point_max = data_dict['point_max']
    batch_size = point_min.shape[0]
    ones = torch.ones(batch_size, dtype=torch.long, device='cuda')

    # label for 9 area
    first_point = point_min + (point_max - point_min) / 3
    second_point = point_min + (point_max - point_min) / 3 * 2
    result_first = torch.le(ref_center_label, first_point)
    result_second = torch.le(ref_center_label, second_point)

    label = torch.where(result_first[:, 0] & result_first[:, 1], ones * 0, ones*4)
    label = torch.where(result_first[:, 0].logical_not() & result_second[:, 0] & result_first[:, 1], ones, label)
    label = torch.where(result_second[:, 0].logical_not() & result_first[:, 1], ones*2, label)
    label = torch.where(result_first[:, 0] & result_first[:, 1].logical_not() & result_second[:, 0], ones*3, label)
    label = torch.where(result_second[:, 0].logical_not() & result_first[:, 1].logical_not() & result_second[:, 1], ones*5, label)
    label = torch.where(result_first[:, 0] & result_second[:, 1].logical_not(), ones*6, label)
    label = torch.where(result_first[:, 0].logical_not() & result_second[:, 0] & result_second[:, 1].logical_not(), ones*7, label)
    label = torch.where(result_second[:, 0].logical_not() & result_second[:, 1].logical_not(), ones*8, label)

    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred, label)

    pred = torch.argmax(pred, 1)
    corrects = (pred == label)
    acc = corrects.sum() / float(label.numel())
    return loss, acc


def compute_box_loss(data_dict, box_mask):
    """ Compute 3D bounding box loss.
    Args:
        data_dict: dict (read-only)
    Returns:
        center_loss
        size_reg_loss
    """

    # Compute center loss
    pred_center = data_dict['center']
    pred_size_residual = data_dict['size_residual']

    gt_center = data_dict['ref_center_label']
    gt_size_residual = data_dict['ref_size_residual_label']

    creterion = nn.SmoothL1Loss(reduction='none')
    center_loss = creterion(pred_center, gt_center)
    center_loss = (center_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)
    size_loss = creterion(pred_size_residual, gt_size_residual)
    size_loss = (size_loss * box_mask.unsqueeze(1)).sum() / (box_mask.sum() + 1e-6)

    return center_loss, size_loss


def compute_lang_classification_loss(data_dict):
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(data_dict["lang_scores"], data_dict["object_cat"])

    return loss


def get_loss(data_dict, config):
    """ Loss functions
    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """
    lang_loss = compute_lang_classification_loss(data_dict)
    object_loss = TargetClassificationLoss().cuda()
    data_dict["lang_loss"] = lang_loss
    seg_loss, seg_acc = compute_scene_mask_loss(data_dict)

    # get ref gt
    ref_center_label = data_dict["ref_center_label"].detach().cpu().numpy()
    ref_heading_class_label = data_dict["ref_heading_class_label"].detach().cpu().numpy()
    ref_heading_residual_label = data_dict["ref_heading_residual_label"].detach().cpu().numpy()
    ref_size_class_label = data_dict["ref_size_class_label"].detach().cpu().numpy()
    ref_size_residual_label = data_dict["ref_size_residual_label"].detach().cpu().numpy()

    ref_gt_obb = config.param2obb_batch(ref_center_label, ref_heading_class_label, ref_heading_residual_label,
                                        ref_size_class_label, ref_size_residual_label)
    ref_gt_bbox = get_3d_box_batch(ref_gt_obb[:, 3:6], ref_gt_obb[:, 6], ref_gt_obb[:, 0:3])

    attribute_scores = data_dict['attribute_scores']
    relation_scores = data_dict['relation_scores']
    scene_scores = data_dict['scene_scores']

    pred_obb_batch = data_dict['pred_obb_batch']
    batch_size = len(pred_obb_batch)
    cluster_label = []
    box_mask = torch.zeros(batch_size).cuda()

    criterion = ContrastiveLoss(margin=0.2, gamma=5)
    ref_loss = torch.zeros(1).cuda().requires_grad_(True)
    start_idx = 0

    # LOSS FOR MY IMPLEMENTATION
    class_loss = torch.zeros(1).cuda().requires_grad_(True)
    bts_candidate_obbs = data_dict["bts_candidate_obbs"]
    scores = data_dict["score"] # B x 8 x 1
    batch_pred_scores = []
    batch_pred_label = []
    batch_size = bts_candidate_obbs.shape[0]
    label = []
    for ii in range(batch_size):  
        candidate_obbs = bts_candidate_obbs[ii].cpu().numpy() # MAX_NUM_OBJECT x 6
        pred_score = scores[ii].reshape(-1) # MAX_NUM_OBJECT x 1

        # just for eval
        x = pred_score.detach().cpu().numpy()
        batch_pred_scores.append(int(np.argmax(x)))

        pred_bbox = get_3d_box_batch(candidate_obbs[:, 3:6], np.zeros(MAX_NUM_OBJECT), candidate_obbs[:, 0:3])
        ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[ii], (MAX_NUM_OBJECT, 1, 1)))
        # print('max iou: ', ious.max())
        label.append(ious.argmax())  # MAX_NUM_OBJECT
        # batch_pred_label.append(label.tolist())
    label = np.array(label)
    class_loss = class_loss + object_loss(scores.reshape(batch_size, 8), torch.from_numpy(label).long().cuda())


    accuracy = np.mean(batch_pred_scores == label)
    print(f"Accuracy: {accuracy:.2f}")
    print('target classification loss: ', class_loss)

    for i in range(batch_size):
        pred_obb = pred_obb_batch[i]  # (num, 7)
        num_filtered_obj = pred_obb.shape[0]
        if num_filtered_obj == 0:
            cluster_label.append([])
            box_mask[i] = 1
            continue

        label = np.zeros(num_filtered_obj)
        pred_bbox = get_3d_box_batch(pred_obb[:, 3:6], pred_obb[:, 6], pred_obb[:, 0:3])
        ious = box3d_iou_batch(pred_bbox, np.tile(ref_gt_bbox[i], (num_filtered_obj, 1, 1)))
        label[ious.argmax()] = 1  # treat the bbox with highest iou score as the gt

        label = torch.FloatTensor(label).cuda()
        cluster_label.append(label)
        if num_filtered_obj == 1: continue

        attribute_score = attribute_scores[start_idx:start_idx + num_filtered_obj]
        relation_score = relation_scores[start_idx:start_idx + num_filtered_obj]
        scene_score = scene_scores[start_idx:start_idx + num_filtered_obj]
        score = attribute_score + relation_score + scene_score

        start_idx += num_filtered_obj
        # print('max iou check: ', ious.max())
        if ious.max() < 0.2: continue

        ref_loss = ref_loss + criterion(score, label)

    ref_loss = ref_loss / batch_size
    data_dict['ref_loss'] = ref_loss

    data_dict['loss'] = ref_loss + lang_loss + seg_loss + 10 * class_loss
    data_dict["seg_loss"] = seg_loss
    data_dict['seg_acc'] = seg_acc
    data_dict['seg_loss'] = seg_loss
    data_dict['class_loss'] = class_loss
    data_dict['cluster_label'] = cluster_label

    return data_dict
