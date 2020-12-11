from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import pylibmc
import math

import logging

logger = logging.getLogger(__name__)


class distillLossOp(object):
    def __init__(self, train,temperature):
        self._train = train
        self._mc = pylibmc.Client(["127.0.0.1:11212"], binary=True,
                     behaviors={"tcp_nodelay": True,
                                "ketama": True})
        self._temperature=temperature

    def forward(self, inputs, outputs):
        
        outputs[0].reshape((1,))
        outputs[0].data[0]=0.5
            
    def backwardd(self,inputs,outputs):
        
        #inputs[0] distilled logits
        #inputs[1] label
        #inputs[2] loss
        #inputs[3] loss gradient
        
        #outputs[0] distilled logits gradient
        #outputs[1] label logits gradient
                
        outputs[0].reshape(inputs[0].data.shape)

        probDistilled=inputs[0].data
        probLabel=inputs[1].data
        
        probDistilled_row_max=np.max(probDistilled,axis=1)
        probLabel_row_max=np.max(probLabel,axis=1)

        probDistilled = probDistilled - probDistilled_row_max[:, np.newaxis]
        probLabel=probLabel - probLabel_row_max[:, np.newaxis]

        probDistilled=probDistilled/self._temperature
        probLabel=probLabel/self._temperature


        logger.info(np.max(probDistilled))
        
        probDistilled=math.e**probDistilled
        probLabel=math.e**probLabel
        
        probDistilled_row_sum=np.sum(probDistilled,axis=1)
        probLabel_row_sum=np.sum(probLabel,axis=1)
        
        probDistilled = probDistilled / probDistilled_row_sum[:, np.newaxis]
        probLabel=probLabel/probLabel_row_sum[:, np.newaxis]

        outputs[0].data[...]=2*2/3*(1./self._temperature)*(self._temperature**2)*(1./512.)*(probDistilled-probLabel)

