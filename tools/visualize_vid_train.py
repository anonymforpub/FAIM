#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import sys, os
import warnings

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from tqdm import tqdm

from yolox.core import launch
from yolox.data.datasets import vid, vid_classes
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("Parser for visualize training data script")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default='',
        type=str,
        help="plz input your expriment description file",
    )
    parser.add_argument(
        "-p",
        "--file_path",
        default='../../dataset_annotations/imagenetVID_2015/YOLOV_annotations/train_seq.npy',
        type=str,
        help="numpy file path to load DET data for ImageNetVID dataset",
    )
    parser.add_argument(
        "-out",
        "--out_dir",
        default='../../object_det_videos/VOWSAM_outputs/visualizations_gt_mask',
        type=str,
        help="numpy file path to load DET data for ImageNetVID dataset",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--gframe', default=16, help='global frame num')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers')

    return parser

def visualize_samples(info_imgs, labels, masks, confidence_scores, imgs, out_dir):
    '''
    This function is modified and now compatible with newly created instance masks.
    Args:
        info_imgs:  Image paths, list of image paths
        labels: Tensor (N X 5) BBox and class info
        masks: Mask containing boolean mask [B,N,H,W] instead of polygon points, H and W is same as image
        confidence_scores: confidence score predicted by SAM for each mask
        imgs: batch of images as tensor

    Returns:
        Nothing, visualize the whole batch of samples with their corresponding
        masks, bbox, and class name
    '''
    #Transforming image batch to numpy
    imgs = imgs.cpu().numpy().astype(np.uint8)
    masks = masks.cpu().detach().numpy()

    for ind, img in enumerate(imgs):

        #Fetching bounding box, masks, and class labels
        bbox_preds = labels[ind][:, :4].cpu().detach().numpy()  # [num_boxes_per_frame, 4]
        class_id = labels[ind][:, 4].cpu().detach().numpy()  # [num_boxes_per_frame, 4]
        instance_masks = masks[ind]  # [N, 28, 28]
        # Create a copy of the original image to work on

        image_copy = img.copy()

        #filtering bbox_preds to non-zero values to reduce for loop iterations
        bbox_preds = bbox_preds[np.any(bbox_preds != 0, axis=1)]

        # print("instance_masks shape ", instance_masks.shape, "bbox_preds shape ", bbox_preds.shape)

        # Convert bbox coordinates to integers and Draw bounding box
        for i, (bbox_pred, instance_mask) in enumerate(zip(bbox_preds, instance_masks)):

            bbox = [int(coordinate) for coordinate in bbox_pred] # converting into int to be used later

            # Create an overlay image
            # overlay = image_copy[bbox_pred[1]:bbox_pred[3], bbox_pred[0]:bbox_pred[2]]
            # overlay[instance_mask > 0.5] = [0, 0, 255]  # Assuming mask values are in [0, 1]
            # cv2.addWeighted(overlay, 0.5, image_copy[bbox_pred[1]:bbox_pred[3], bbox_pred[0]:bbox_pred[2]], 0.5, 0,
            #                 image_copy[bbox_pred[1]:bbox_pred[3], bbox_pred[0]:bbox_pred[2]])

            #CURRENTLY CONTOUR METHOD IS USED BUT IT CAN BE CHANGED with overlay to control transparency
            instance_mask = (instance_mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(instance_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create an overlay of zeros that is the same size as the image
            overlay = np.zeros_like(image_copy)

            # Fill the detected object (defined by the mask) with a color, e.g., red
            cv2.fillPoly(overlay, contours, (0, 0, 255))

            # Create a mask for blending
            blend_mask = np.zeros_like(image_copy[:, :, 0])  # Assuming the image is in shape (H, W, C)
            cv2.fillPoly(blend_mask, contours, 1)

            # Blend only where the mask is present
            alpha = 0.5  # Adjust the alpha value between 0 (transparent) and 1 (opaque) for the overlay
            image_copy[blend_mask == 1] = cv2.addWeighted(overlay, alpha, image_copy, 1 - alpha, 0)[blend_mask == 1]

            # Draw bounding box
            cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 4)
            # Write Confidence Score of SAM for the predicted mask
            # Calculate the position for the text
            text_x = bbox[0]
            text_y = bbox[1] + 10  # Adjust the value to change the vertical position of the text
            # Add the text on the image
            confidence_score = confidence_scores[ind].cpu().detach().numpy()[i]
            class_name = vid_classes.VID_classes[int(class_id[i])]
            cv2.putText(image_copy, class_name+" : {:.3f}".format(confidence_score), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 0, 0), 2, cv2.LINE_AA)

        # Extract the subdirectories and filename from the original path
        rel_path = info_imgs[ind].split("/train/")[-1]
        file_write_dir = os.path.join(out_dir, os.path.dirname(rel_path))
        # Create directory if it doesn't exist
        os.makedirs(file_write_dir, exist_ok=True)
        # Construct the full path to save the image
        save_path = os.path.join(file_write_dir, os.path.basename(info_imgs[ind]))
        # print(save_path)
        # Save the processed image
        cv2.imwrite(save_path, image_copy)


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True
    lframe = int(args.lframe)
    gframe = int(args.gframe)

    print(f"PRINTING NUM WORKERS {args.num_workers} AND BATCH SIZE {args.batch_size}  AND GFRAME {gframe} AND LFRAME {lframe}")

    dataset = vid.SAMVIDMaskDataset(img_size=exp.input_size, lframe=lframe, gframe=gframe,
                                 dataset_pth=exp.data_dir, file_path=args.file_path)

    #same collate function is used for both train VOWSAM and visualiazation
    dataset_loader = vid.get_trans_loader(batch_size=lframe+gframe, data_num_workers=args.num_workers, dataset=dataset, mask=True)

    for i, (imgs, labels, _, _, masks, confidence_scores, paths, _) in tqdm(enumerate(dataset_loader), total=len(dataset_loader)):
        #calling this function will also write images in the ars.out_dir directory
        visualize_samples(paths, labels, masks, confidence_scores, imgs, out_dir=args.out_dir)

    print(f"{len(dataset)} FILES ARE WRITTEN ON THE DISK")


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    exp.merge(args.opts)
    # exp.test_size = (args.tsize, args.tsize)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    args.machine_rank = 1
    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        1,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
