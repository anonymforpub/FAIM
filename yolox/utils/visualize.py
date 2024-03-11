#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os

import cv2
import numpy as np

__all__ = ["vis", "vis_with_mask", "visualize_mask_pred_gt_during_loss"]

def visualize_mask_pred_gt_during_loss(mask_img_pred, mask_img_targets, out_dir="../VOWSAM_outputs/mask_gt_vis", idx=0, image=None):
    os.makedirs(out_dir, exist_ok=True)

    ground_truth_mask = mask_img_targets.cpu().numpy()
    predicted_mask = mask_img_pred.cpu().detach().numpy()

    # Resize the masks
    resize_shape = (512,512)
    ground_truth_mask = cv2.resize(ground_truth_mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    predicted_mask = cv2.resize(predicted_mask, resize_shape, interpolation=cv2.INTER_NEAREST)

    ground_truth_path = os.path.join(out_dir, f'ground_truth_{idx}.png')
    predicted_path = os.path.join(out_dir, f'predicted_{idx}.png')

    if image is not None:
        image = image.cpu().numpy()
        image_path = os.path.join(out_dir, f'image_path_{idx}.png')
        cv2.imwrite(image_path, image)

    cv2.imwrite(ground_truth_path, ground_truth_mask * 255)
    cv2.imwrite(predicted_path, predicted_mask * 255)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None,t_size = 0.4):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id%len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id%len(_COLORS)]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, t_size, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id%len(_COLORS)] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, t_size, txt_color, thickness=1)

    return img

def vis_with_mask(img, boxes, scores, cls_ids, conf=0.5, class_names=None,t_size = 0.4, masks=None):

    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        # Get the corresponding color for this class
        color = (_COLORS[cls_id%len(_COLORS)] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id%len(_COLORS)]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, t_size, 1)[0]
        # Draw the bounding box
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id%len(_COLORS)] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, t_size, txt_color, thickness=1)

        #ADDING MASK CODE TO HERE VIUSALIZE INSTANCE MASK ON THE IMAGE
        # Apply mask only if this condition satisfies
        if i <len(masks) - 1:
            mask = masks[i]  # Assuming masks is a list of arrays where each array is [28, 28]


            # Resize mask to fit the bounding box
            mask_resized = cv2.resize(mask, (x1 - x0, y1 - y0))
            # print("resized mask shape ", mask_resized.shape)

            mask_resized = (mask_resized > 0.5).astype(np.uint8)
            # print("resized mask vale ", mask_resized)
            contours, _ = cv2.findContours(mask_resized, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # print("contours ", contours)
            cv2.fillPoly(img[y0:y1, x0:x1], contours, color)

    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
