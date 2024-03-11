"""
given a imagenet vid imdb and COCO as PKL, compute mAP
This script is a subset of REPPM.py
When post-processing is done and PKL files are generated.
To compute mAP motion wise again, it is cumbersome to run the whole
script that takes more than 10 minutes.
"""

import pickle

import numpy
import numpy as np
from scipy import signal, ndimage
import json

from repp_utils import get_video_frame_iterator, get_iou, get_pair_features
import motion_utils

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Apply REPP to a saved predictions file')
    parser.add_argument('--repp_cfg', help='repp cfg filename', type=str)
    parser.add_argument('--post',dest='post',default=False,help='involve post-processing or not', action="store_true")
    parser.add_argument('--process_num', default = 6, type=int)

    parser.add_argument('--predictions_file', help='predictions filename', type=str)
    parser.add_argument('--from_python_2', help='predictions filename', action='store_true')
    parser.add_argument('--evaluate', help='evaluate motion mAP', action='store_true')
    parser.add_argument('--annotations_filename', help='ILSVRC annotations. Needed for ILSVRC evaluation',
                        required=False, type=str)
    parser.add_argument('--path_dataset', help='path of the Imagenet VID dataset. Needed for ILSVRC evaluation',
                        required=False, type=str)
    parser.add_argument('--coco', help='store processed predictions in coco format', required=False, type=str)
    parser.add_argument('--imdb', help='store processed predictions in imdb format', required=False, type=str)
    args = parser.parse_args()

    assert not (
                args.evaluate and args.annotations_filename is None), 'Annotations filename is required for ILSVRC evaluation'
    assert not (args.evaluate and args.path_dataset is None), 'Dataset path is required for ILSVRC evaluation'

    # print(' * Loading REPP cfg')
    # repp_params = json.load(open(args.repp_cfg, 'r'))
    # print(repp_params)
    #
    # predictions_file_out = args.predictions_file.replace('.pckl', '_repp')
    #
    # repp = REPP(**repp_params, annotations_filename=args.annotations_filename,
    #             store_coco=args.store_coco, store_imdb=args.store_imdb or args.evaluate,post=args.post)
    #
    # from tqdm import tqdm
    # from tqdm.contrib.concurrent import process_map
    # import sys
    #
    # print(' * Applying repp')
    # total_preds_coco, total_preds_imdb = [], []
    #
    # video_dic = []
    # for vid, video_preds in get_video_frame_iterator(args.predictions_file, from_python_2=args.from_python_2):
    #     video_preds = dict(sorted(video_preds.items(), key=lambda x: x[0]))
    #     video_dic.append(video_preds)
    # res = process_map(repp, video_dic, max_workers=args.process_num)
    # for ele in res:
    #     total_preds_imdb += ele[1]
    #     total_preds_coco += ele[0]
    #
    # if args.store_imdb:
    #     print(' * Dumping predictions with the IMDB format:', predictions_file_out + '_imdb.txt')
    #     with open(predictions_file_out + '_imdb.txt', 'w') as f:
    #         for p in total_preds_imdb: f.write(p + '\n')
    #
    # if args.store_coco:
    #     print(' * Dumping predictions with the COCO format:', predictions_file_out + '_coco.json')
    #     json.dump(total_preds_coco, open(predictions_file_out + '_coco.json', 'w'))

    predictions_file_out = args.imdb
    if args.evaluate:

        print(' * Evaluating REPP predictions')

        import sys

        sys.path.append('ObjectDetection_mAP_by_motion')
        import motion_utils
        from imagenet_vid_eval_motion import get_motion_mAP
        import os

        stats_file_motion = predictions_file_out.replace('preds', 'stats').replace('.txt', '.json')
        motion_iou_file_orig = './tools/imagenet_vid_groundtruth_motion_iou.mat'
        imageset_filename_orig = os.path.join(args.path_dataset, 'ImageSets/VID_val_frames.txt')

        predictions_file_out_txt = predictions_file_out
        if os.path.isfile(stats_file_motion): os.remove(stats_file_motion)
        stats = get_motion_mAP(args.annotations_filename, args.path_dataset,
                               predictions_file_out_txt, stats_file_motion,
                               motion_iou_file_orig, imageset_filename_orig)

        print(stats)
        print(' * Stats stored:', stats_file_motion)