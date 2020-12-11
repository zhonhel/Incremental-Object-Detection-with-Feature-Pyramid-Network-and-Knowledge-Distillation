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

import logging
logger = logging.getLogger(__name__)

class mySplitOp(object):
    def __init__(self, train):
        self._train = train

    def forward(self, inputs, outputs):
        
        len0=int(inputs[1].data[0])
        len1=int(inputs[1].data[1])
        
        channelsNum=(inputs[0].data.shape)[1]
        
        print('channelsNum ',channelsNum)

        outputs[0].reshape((len0,channelsNum))
        outputs[1].reshape((len1,channelsNum))
        
        outputs[0].data[...]=inputs[0].data[:len0,:]
        outputs[1].data[...]=inputs[0].data[len0:,:]
        
