from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import importlib
import logging

from caffe2.python import core
from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.modeling.detector import DetectionModelHelper
from detectron.roi_data.loader import RoIDataLoader
import detectron.modeling.fast_rcnn_heads as fast_rcnn_heads

import detectron.modeling.name_compat as name_compat
import detectron.modeling.optimizer as optim

import detectron.modeling.rpn_heads as rpn_heads
import detectron.roi_data.minibatch as roi_data_minibatch
import detectron.utils.c2 as c2_utils

logger = logging.getLogger(__name__)

def generalized_rcnn(model):
    """This model type handles:
      - Fast R-CNN
      - RPN only (not integrated with Fast R-CNN)
      - Faster R-CNN (stagewise training from NIPS paper)
      - Faster R-CNN (end-to-end joint training)
      - Mask R-CNN (stagewise training from NIPS paper)
      - Mask R-CNN (end-to-end joint training)
    """
    return build_generic_detection_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        add_roi_box_head_func=get_func(cfg.FAST_RCNN.ROI_BOX_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.ROI_MASK_HEAD),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD),
        freeze_conv_body=cfg.TRAIN.FREEZE_CONV_BODY
    )

# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #

def create(model_type_func, train=False, gpu_id=0):
    """Generic model creation function that dispatches to specific model
    building functions.

    By default, this function will generate a data parallel model configured to
    run on cfg.NUM_GPUS devices. However, you can restrict it to build a model
    targeted to a specific GPU by specifying gpu_id. This is used by
    optimizer.build_data_parallel_model() during test time.
    """
    model = DetectionModelHelper(
        name=model_type_func,
        train=train,
        num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=train
    )
    model.only_build_forward_pass = False
    model.target_gpu_id = gpu_id
    return get_func(model_type_func)(model)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    new_func_name = name_compat.get_new_name(func_name)
    if new_func_name != func_name:
        logger.warn(
            'Remapping old function name: {} -> {}'.
            format(func_name, new_func_name)
        )
        func_name = new_func_name
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'detectron.modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: {}'.format(func_name))
        raise


def build_generic_detection_model(
    model,
    add_conv_body_func,
    add_roi_box_head_func=None,
    add_roi_mask_head_func=None,
    add_roi_keypoint_head_func=None,
    freeze_conv_body=False
):
    def _single_gpu_build_func(model):
        """Build the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model.
        """
        # Add the conv body (called "backbone architecture" in papers)
        # E.g., ResNet-50, ResNet-50-FPN, ResNeXt-101-FPN, etc.
        blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)

        head_loss_gradients = {
            'rpn': None,
            'box': None,
            'mask': None,
            'keypoints': None,
        }

        if cfg.RPN.RPN_ON:
            # Add the RPN head
            head_loss_gradients['rpn'] = rpn_heads.add_generic_rpn_outputs(
                model, blob_conv, dim_conv, spatial_scale_conv
            )

        if cfg.FPN.FPN_ON:
            # After adding the RPN head, restrict FPN blobs and scales to
            # those used in the RoI heads
            blob_conv, spatial_scale_conv = _narrow_to_fpn_roi_levels(
                blob_conv, spatial_scale_conv
            )

        if not cfg.MODEL.RPN_ONLY:
            # Add the Fast R-CNN head
            head_loss_gradients['box'] = _add_fast_rcnn_head(
                model, add_roi_box_head_func, blob_conv, dim_conv,
                spatial_scale_conv
            )


        if model.train:
            loss_gradients = {}
            for lg in head_loss_gradients.values():
                if lg is not None:
                    loss_gradients.update(lg)
            return loss_gradients


    optim.build_data_parallel_model(model, _single_gpu_build_func)
    return model


def _narrow_to_fpn_roi_levels(blobs, spatial_scales):
    """Return only the blobs and spatial scales that will be used for RoI heads.
    Inputs `blobs` and `spatial_scales` may include extra blobs and scales that
    are used for RPN proposals, but not for RoI heads.
    """
    # Code only supports case when RPN and ROI min levels are the same
    assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
    # RPN max level can be >= to ROI max level
    assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
    # FPN RPN max level might be > FPN ROI max level in which case we
    # need to discard some leading conv blobs (blobs are ordered from
    # max/coarsest level to min/finest level)
    num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
    return blobs[-num_roi_levels:], spatial_scales[-num_roi_levels:]


def _add_fast_rcnn_head(
    model, add_roi_box_head_func, blob_in, dim_in, spatial_scale_in
):
    """Add a Fast R-CNN head to the model."""
    blob_frcn, dim_frcn = add_roi_box_head_func(
        model, blob_in, dim_in, spatial_scale_in
    )
    fast_rcnn_heads.add_fast_rcnn_outputs(model, blob_frcn, dim_frcn)
    
    if model.train:
        loss_gradients = fast_rcnn_heads.add_fast_rcnn_losses(model)

    return loss_gradients


# ---------------------------------------------------------------------------- #
# Network inputs
# ---------------------------------------------------------------------------- #

def add_training_inputs(model, roidb=None):
    """Create network input ops and blobs used for training. To be called
    *after* model_builder.create().
    """
    # Implementation notes:
    #   Typically, one would create the input ops and then the rest of the net.
    #   However, creating the input ops depends on loading the dataset, which
    #   can take a few minutes for COCO.
    #   We prefer to avoid waiting so debugging can fail fast.
    #   Thus, we create the net *without input ops* prior to loading the
    #   dataset, and then add the input ops after loading the dataset.
    #   Since we defer input op creation, we need to do a little bit of surgery
    #   to place the input ops at the start of the network op list.
    assert model.train, 'Training inputs can only be added to a trainable model'
    if roidb is not None:
        # To make debugging easier you can set cfg.DATA_LOADER.NUM_THREADS = 1
        model.roi_data_loader = RoIDataLoader(
            roidb,
            num_loaders=cfg.DATA_LOADER.NUM_THREADS,
            minibatch_queue_size=cfg.DATA_LOADER.MINIBATCH_QUEUE_SIZE,
            blobs_queue_capacity=cfg.DATA_LOADER.BLOBS_QUEUE_CAPACITY
        )
    orig_num_op = len(model.net._net.op)
    blob_names = roi_data_minibatch.get_minibatch_blob_names(is_training=True)
    for gpu_id in range(cfg.NUM_GPUS):
        with c2_utils.NamedCudaScope(gpu_id):
            for blob_name in blob_names:
                workspace.CreateBlob(core.ScopedName(blob_name))
            model.net.DequeueBlobs(
                model.roi_data_loader._blobs_queue_name, blob_names
            )
    # A little op surgery to move input ops to the start of the net
    diff = len(model.net._net.op) - orig_num_op
    new_op = model.net._net.op[-diff:] + model.net._net.op[:-diff]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def add_inference_inputs(model):
    """Create network input blobs used for inference."""

    def create_input_blobs_for_net(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    create_input_blobs_for_net(model.net.Proto())



