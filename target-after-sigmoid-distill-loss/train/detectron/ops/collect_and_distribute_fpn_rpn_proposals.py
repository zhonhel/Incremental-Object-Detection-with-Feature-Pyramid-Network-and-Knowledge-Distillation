from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from detectron.core.config import cfg
from detectron.datasets import json_dataset
from detectron.datasets import roidb as roidb_utils
import detectron.roi_data.fast_rcnn as fast_rcnn_roi_data
import detectron.utils.blob as blob_utils
import logging
import pylibmc

logger = logging.getLogger(__name__)

class CollectAndDistributeFpnRpnProposalsOp(object):
    def __init__(self, train):
        self._train = train
        self._mc = pylibmc.Client(["127.0.0.1:11212"], binary=True,
                     behaviors={"tcp_nodelay": True,
                                "ketama": True})        

    def forward(self, inputs, outputs):

        rois = collect(inputs, self._train)
        


        im_info = inputs[-1].data
        im_scales = im_info[:, 2]
        roidb = blob_utils.deserialize(inputs[-2].data)

        json_dataset.add_proposals(roidb, rois, im_scales, crowd_thresh=0)
        roidb_utils.add_bbox_regression_targets(roidb)

        output_blob_names = fast_rcnn_roi_data.get_fast_rcnn_blob_names()
        blobs = {k: [] for k in output_blob_names}
        fast_rcnn_roi_data.add_fast_rcnn_blobs(blobs, im_scales, roidb, self._mc)
        for i, k in enumerate(output_blob_names):
            blob_utils.py_op_copy_blob(blobs[k], outputs[i])
            



def collect(inputs, is_training):

    post_nms_topN = cfg['TRAIN'].RPN_POST_NMS_TOP_N
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    num_lvls = k_max - k_min + 1
       
    roi_inputs = inputs[0:num_lvls]
    score_inputs = inputs[num_lvls:num_lvls*2]

    rois = np.concatenate([blob.data for blob in roi_inputs])
    scores = np.concatenate([blob.data for blob in score_inputs]).squeeze()
    inds = np.argsort(-scores)[:post_nms_topN]
    rois = rois[inds, :]
    
    return rois