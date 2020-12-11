from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pylibmc

class get_weightsOpRPN(object):
    def __init__(self, train,slvl,tag):
        self._train = train
        self._slvl=slvl
        self._tag=tag
        self._mc = pylibmc.Client(["127.0.0.1:11212"], binary=True,
                     behaviors={"tcp_nodelay": True,
                                "ketama": True})        

    def forward(self, inputs, outputs):
        
        if self._tag=='cls_probs':
            while True:
                if(self._mc.get('rpn_cls_probs_fpn'+self._slvl+'_s')=='weidu'):
                    break              
            tmp=self._mc.get('rpn_cls_probs_fpn'+self._slvl)
            self._mc.replace('rpn_cls_probs_fpn'+self._slvl+'_s','yidu')
                
        if self._tag=='bbox_pred':
            while True:
                if(self._mc.get('rpn_bbox_pred_fpn'+self._slvl+'_s')=='weidu'):
                    break              
            tmp=self._mc.get('rpn_bbox_pred_fpn'+self._slvl)
            self._mc.replace('rpn_bbox_pred_fpn'+self._slvl+'_s','yidu')
            
        outputs[0].reshape(tmp.shape)
        outputs[0].data[...]=tmp[...]
        
        outputs[1].reshape(tmp.shape)
        outputs[1].data[...]=1.0
        
        outputs[2].reshape(tmp.shape)
        outputs[2].data[...]=1.0/(tmp.shape[1]*tmp.shape[2]*tmp.shape[3]) 
        
 
 
        
        