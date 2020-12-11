from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import pylibmc

class get_weightsOpFastrcnn(object):
    def __init__(self, train):
        self._train = train
        self._mc = pylibmc.Client(["127.0.0.1:11212"], binary=True,
                     behaviors={"tcp_nodelay": True,
                                "ketama": True})        

    def forward(self, inputs, outputs):
        
        while True:
            if(self._mc.get('freeze_fastrcnn_label_s')=='weidu'):
                break       
        freeze_fastrcnn_label=self._mc.get('freeze_fastrcnn_label')
        self._mc.replace('freeze_fastrcnn_label_s','yidu')
        
        for i in range(2):

            outputs[3*i].reshape(freeze_fastrcnn_label[i].shape)
            outputs[3*i].data[...]=freeze_fastrcnn_label[i][...]
            
            outputs[3*i+1].reshape(freeze_fastrcnn_label[i].shape)
            outputs[3*i+1].data[...]=1.0
            
            outputs[3*i+2].reshape(freeze_fastrcnn_label[i].shape)
            outputs[3*i+2].data[...]=1.0/(freeze_fastrcnn_label[i].shape[1])    
 
 
        
        