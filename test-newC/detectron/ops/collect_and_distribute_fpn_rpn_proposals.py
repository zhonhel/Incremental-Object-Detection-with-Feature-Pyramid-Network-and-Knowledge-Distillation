# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from detectron.core.config import cfg
from detectron.datasets import json_dataset
from detectron.datasets import roidb as roidb_utils
import detectron.modeling.FPN as fpn
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.utils.blob as blob_utils


class CollectAndDistributeFpnRpnProposalsOp(object):
    def __init__(self, train):
        self._train = train

    def forward(self, inputs, outputs):
        """See modeling.detector.CollectAndDistributeFpnRpnProposals for
        inputs/outputs documentation.
        """
        # inputs is
        # [rpn_rois_fpn2, ..., rpn_rois_fpn6,
        #  rpn_roi_probs_fpn2, ..., rpn_roi_probs_fpn6]
        # If training with Faster R-CNN, then inputs will additionally include
        #  + [roidb, im_info]
        rois = collect(inputs, outputs)


        distribute(rois, None, outputs, self._train)


def collect(inputs, outputs):
    post_nms_topN = cfg['TEST'].RPN_POST_NMS_TOP_N
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
    outputs[-1].reshape((2,))
    
    roi_inputs = inputs[:num_lvls]
    score_inputs = inputs[num_lvls*2:num_lvls*2+num_lvls]
    # rois are in [[batch_idx, x0, y0, x1, y2], ...] format
    # Combine predictions across all levels and retain the top scoring
    rois = np.concatenate([blob.data for blob in roi_inputs])
    scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]  
    scores=scores[inds]
    outputs[-1].data[0]=rois.shape[0]
    
    
    
    roi_inputs_tooth = inputs[num_lvls:num_lvls*2]
    score_inputs_tooth = inputs[num_lvls*2+num_lvls:num_lvls*2+num_lvls*2]
    
    rois_tooth = np.concatenate([blob.data for blob in roi_inputs_tooth])
    scores_tooth = np.concatenate([blob.data for blob in score_inputs_tooth]).squeeze()
    inds_tooth = np.argsort(-scores_tooth)[:200]
    rois_tooth = rois_tooth[inds_tooth, :]
    scores_tooth=scores_tooth[inds_tooth]
    outputs[-1].data[1]=rois_tooth.shape[0]


    rois=np.concatenate((rois,rois_tooth),axis=0) 

          
    return rois


def distribute(rois, label_blobs, outputs, train):
    """To understand the output blob order see return value of
    detectron.roi_data.fast_rcnn.get_fast_rcnn_blob_names(is_training=False)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(rois[:, 1:5], lvl_min, lvl_max)

    outputs[0].reshape(rois.shape)
    outputs[0].data[...] = rois

    # Create new roi blobs for each FPN level
    # (See: modeling.FPN.add_multilevel_roi_blobs which is similar but annoying
    # to generalize to support this particular case.)
    rois_idx_order = np.empty((0, ))
    for output_idx, lvl in enumerate(range(lvl_min, lvl_max + 1)):
        idx_lvl = np.where(lvls == lvl)[0]
        blob_roi_level = rois[idx_lvl, :]
        outputs[output_idx + 1].reshape(blob_roi_level.shape)
        outputs[output_idx + 1].data[...] = blob_roi_level
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
    rois_idx_restore = np.argsort(rois_idx_order)
    blob_utils.py_op_copy_blob(rois_idx_restore.astype(np.int32), outputs[-2])
