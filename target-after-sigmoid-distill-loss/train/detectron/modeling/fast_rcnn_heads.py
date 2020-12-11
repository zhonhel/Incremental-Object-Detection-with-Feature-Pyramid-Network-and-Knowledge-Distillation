from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.utils.blob as blob_utils
from caffe2.python import core

def add_fast_rcnn_outputs(model, blob_in, dim):
    
    
    model.FC(
        'fc7_newC',
        'cls_score_toothbrush',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        'fc7_newC',
        'bbox_pred_toothbrush',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )    
    
    
    model.FC(
        'fc7_oldC',
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes
    )
    model.FC(
        'fc7_oldC',
        'bbox_pred',
        dim,
        num_bbox_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )
    
            

def add_fast_rcnn_losses(model):

    cls_prob_toothbrush, loss_cls_toothbrush = model.net.SoftmaxWithLoss(
        ['cls_score_toothbrush', 'labels_int32'], ['cls_prob_toothbrush', 'loss_cls_toothbrush'],
        scale=1*2/3*model.GetLossScale()
    )
    loss_bbox_toothbrush = model.net.SmoothL1Loss(
        [
            'bbox_pred_toothbrush', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox_toothbrush',
        scale=1*2/3*model.GetLossScale()
    )
    
    
    model.get_weightsFastrcnn()
    model.StopGradient('freeze_cls_score', 'freeze_cls_score')
    model.distillLoss(['cls_score', 'freeze_cls_score'],['distillLoss'],temperature=2)       
    fastrcnn_reg_loss = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'freeze_bbox_pred', 'fastrcnn_reg_loss_inside_weights',
            'fastrcnn_reg_loss_outside_weights'
        ],
        'fastrcnn_reg_loss',
        scale=2*2/3*model.GetLossScale()
    )    
    
          
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls_toothbrush, loss_bbox_toothbrush,'distillLoss',fastrcnn_reg_loss])

    model.AddLosses(['loss_cls_toothbrush', 'loss_bbox_toothbrush','distillLoss','fastrcnn_reg_loss'])

    return loss_gradients



def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):

    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
            
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')

    model.Split(
        ['fc7',core.BlobReference('split_tensor')],
        ['fc7_newC','fc7_oldC'],
        axis=0
    )
        
    return 'fc7', hidden_dim