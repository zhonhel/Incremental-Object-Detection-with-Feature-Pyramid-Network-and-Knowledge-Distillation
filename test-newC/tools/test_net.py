#!/usr/bin/env python

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

"""Perform inference on one or more datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import os
import pprint
import sys
import time

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.core.config import merge_cfg_from_list
from detectron.core.test_engine import run_inference
from detectron.utils.logging import setup_logging
import detectron.utils.c2 as c2_utils
import shutil

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg',
        dest='cfg_file',
        help='optional config file',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wait',
        dest='wait',
        help='wait until net file exists',
        default=True,
        type=bool
    )
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true'
    )
    parser.add_argument(
        '--multi-gpu-testing',
        dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference',
        action='store_true'
    )
    parser.add_argument(
        '--range',
        dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None,
        type=int,
        nargs=2
    )
    parser.add_argument(
        'opts',
        help='See detectron/core/config.py for all options',
        default=None,
        nargs=argparse.REMAINDER
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    logger = setup_logging(__name__)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        merge_cfg_from_file(args.cfg_file)
    if args.opts is not None:
        merge_cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for \'{}\' to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    if os.path.exists('./detectron/datasets/data/coco'):
        shutil.rmtree('./detectron/datasets/data/coco')
    os.makedirs('./detectron/datasets/data/coco')

    if 'dior_2nd' in cfg.TEST.WEIGHTS:
        os.system('ln -s /home/wsh/dior/coco/coco_train2014 ./detectron/datasets/data/coco/coco_train2014')
        os.system('ln -s /home/wsh/dior/coco/coco_val2014 ./detectron/datasets/data/coco/coco_val2014 ')
        os.system('ln -s /home/wsh/dior/coco/annotationsN_2nd ./detectron/datasets/data/coco/annotations')
    elif 'dior_3rd' in cfg.TEST.WEIGHTS:
        os.system('ln -s /home/wsh/dior/coco/coco_train2014 ./detectron/datasets/data/coco/coco_train2014')
        os.system('ln -s /home/wsh/dior/coco/coco_val2014 ./detectron/datasets/data/coco/coco_val2014 ')
        os.system('ln -s /home/wsh/dior/coco/annotationsN_3rd ./detectron/datasets/data/coco/annotations')
    elif 'dior_4th' in cfg.TEST.WEIGHTS:
        os.system('ln -s /home/wsh/dior/coco/coco_train2014 ./detectron/datasets/data/coco/coco_train2014')
        os.system('ln -s /home/wsh/dior/coco/coco_val2014 ./detectron/datasets/data/coco/coco_val2014 ')
        os.system('ln -s /home/wsh/dior/coco/annotationsN_4th ./detectron/datasets/data/coco/annotations')
    elif 'dior_5th' in cfg.TEST.WEIGHTS:
        os.system('ln -s /home/wsh/dior/coco/coco_train2014 ./detectron/datasets/data/coco/coco_train2014')
        os.system('ln -s /home/wsh/dior/coco/coco_val2014 ./detectron/datasets/data/coco/coco_val2014 ')
        os.system('ln -s /home/wsh/dior/coco/annotationsN_5th ./detectron/datasets/data/coco/annotations')
    elif '2020.10.6' in cfg.TEST.WEIGHTS:
        os.system('ln -s /home/wsh/dior/coco/coco_train2014 ./detectron/datasets/data/coco/coco_train2014')
        os.system('ln -s /home/wsh/dior/coco/coco_val2014 ./detectron/datasets/data/coco/coco_val2014 ')
        os.system('ln -s /home/wsh/dior/coco/annotationsN ./detectron/datasets/data/coco/annotations')
    else:
        raise Exception


    run_inference(
        cfg.TEST.WEIGHTS,
        ind_range=args.range,
        multi_gpu_testing=args.multi_gpu_testing,
        check_expected_results=True,
    )
