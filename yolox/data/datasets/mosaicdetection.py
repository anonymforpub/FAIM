#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random

import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank

from ..data_augment import random_affine
from ..data_augment import box_candidates, random_perspective, random_perspective_mask
from .datasets_wrapper import Dataset
import os


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()

    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            # 3 additional image indices
            indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_affine(
                mosaic_img,
                mosaic_labels,
                target_size=(input_w, input_h),
                degrees=self.degrees,
                translate=self.translate,
                scales=self.scale,
                shear=self.shear,
            )

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            # -----------------------------------------------------------------
            # img_info and img_id are not used for training.
            # They are also hard to be specified on a mosaic image.
            # -----------------------------------------------------------------
            return mix_img, padded_labels, img_info, img_id

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, img_id = self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1)
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels


class MosaicDetection_VID(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0,perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, dataset_path = ''
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.res = dataset.res
        self.file_num = 0
        self.dataset_path = dataset_path
    def __len__(self):
        return len(self._dataset)

    def get_mosic_idx(self,path):

        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        anno_path = path_dir.replace("Data","Annotations")
        frame_num = len(os.listdir(anno_path))
        self.file_num = frame_num
        #print(frame_num)
        rand_idx = [random.randint(0,frame_num-1) for _ in range(3)]
        raw = '000000'
        res = []
        res.append(path)
        for idx in rand_idx:
            str_idx = str(idx)
            frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
            res.append(frame_idx)
        return res

    #@Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:

            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            indices = self.get_mosic_idx(idx)
            # 3 additional image indices
            #indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                img, _labels, _, img_id = self._dataset.pull_item(index)
                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels = random_perspective(
                mosaic_img,
                mosaic_labels,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):
                mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim,idx)
            mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim)
            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, img_info, idx#np.array([idx])

        else:
            self._dataset._input_dim = self.input_dim
            img, label, img_info, idx= self._dataset.pull_item(idx)
            img, label = self.preproc(img, label, self.input_dim)
            return img, label, img_info, idx#np.array([idx])

    def get_mixup_idx(self,path):
        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        frame_num = self.file_num
        rand_idx = random.randint(0,frame_num-1)
        str_idx = str(rand_idx)
        raw = '000000'
        frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
        return frame_idx

    def mixup(self, origin_img, origin_labels, input_dim,path):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []

        # while len(cp_labels) == 0:
        #     cp_index = random.randint(0, self.__len__() - 1)
        #     cp_labels = self._dataset.load_anno(cp_index)

        cp_index = self.get_mixup_idx(path)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels

class MosaicDetection_VID_SAMIns(Dataset):
    """ VOWSamIns Version of Detection dataset wrapper that performs mixup for normal dataset.
    - modification in __getitem__ function since now Mask and Confidence score will also be propagated
    - Different SAMSeg, where we use singe mask as a segmentation map for each image.
    - Here, we have separte boolean mask for each instance in an image.
    - Also modification in Mixup fucntion
    """

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0,perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, dataset_path = ''
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.res = dataset.res
        self.file_num = 0
        self.dataset_path = dataset_path
    def __len__(self):
        return len(self._dataset)
    def get_mosic_idx(self,path):

        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        anno_path = path_dir.replace("Data","Annotations")
        frame_num = len(os.listdir(anno_path))
        self.file_num = frame_num
        #print(frame_num)
        rand_idx = [random.randint(0,frame_num-1) for _ in range(3)]
        raw = '000000'
        res = []
        res.append(path)
        for idx in rand_idx:
            str_idx = str(idx)
            frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
            res.append(frame_idx)
        return res

    #@Dataset.mosaic_getitem
    def __getitem__(self, idx):

        self._dataset._input_dim = self.input_dim
        #Modification here
        img, label, masks, confidence_scores, img_info, idx= self._dataset.pull_item(idx)
        # print(f"mask BEFORE prepocessin Mosaic, {label} {masks}")
        img, label, masks = self.preproc(img, label, masks, self.input_dim)
        # print(f"mask after prepocessin Mosaic, {label} { masks}")

        return img, label, masks, confidence_scores, img_info, idx#np.array([idx])

    def get_mixup_idx(self,path):
        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        frame_num = self.file_num
        rand_idx = random.randint(0,frame_num-1)
        str_idx = str(rand_idx)
        raw = '000000'
        frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
        return frame_idx

    def mixup(self, origin_img, origin_labels, origin_mask, input_dim, path):
        '''
        Overriding mixup function from the base class
        to handle mask along with image and targets
        
        Args:
            origin_img:
            origin_mask:
            origin_labels:
            input_dim:
            path:

        Returns:
            image
            labels
            mask
        '''
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        # padded_crop_masks = []

        # while len(cp_labels) == 0:
        #     cp_index = random.randint(0, self.__len__() - 1)
        #     cp_labels = self._dataset.load_anno(cp_index)

        cp_index = self.get_mixup_idx(path)
        img, cp_labels, masks, _, _, _ = self._dataset.pull_item(cp_index)
        # print(f"IN MIXUP FUNCTION mask shape : {masks.shape} and image shape : {img.shape} and len labels : {len(cp_labels)}")

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
            # cp_mask = np.zeros(input_dim, dtype=np.uint8)  # Mask also has a similar starting point

        # Compute scale ratio and resize the copy image and mask
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        # Assign resized image and mask to the copy image and mask
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        # Resize again based on jit factor
        # Compute new dimensions based on jit factor
        new_w = int(cp_img.shape[1] * jit_factor)
        new_h = int(cp_img.shape[0] * jit_factor)
        # Resize cp_img and cp_mask based on the new dimensions
        cp_img = cv2.resize(cp_img, (new_w, new_h))
        # Perform horizontal flip if required
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
        # Create padded image and mask with maximum of original and copy sizes
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8)

        # Random crop
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_img[:origin_h, :origin_w] = cp_img
        padded_cropped_img = padded_img[
                             y_offset: y_offset + target_h, x_offset: x_offset + target_w
                             ]

        # Loop neaded to iterate over N masks in each image
        # for mask_ind in range(len(masks)):
        #
        #     cp_mask = np.zeros(input_dim, dtype=np.uint8)  # Mask also has a similar starting point
        #     # Resize masks just like the image
        #     resized_mask = cv2.resize(masks[mask_ind], (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)), interpolation=cv2.INTER_NEAREST)
        #
        #     # print(f"resized_mask shape : {resized_mask.shape} and resized_img shape : {resized_img.shape} and cp_mask shape : {cp_mask.shape} and input dim : {input_dim}")
        #     cp_mask[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        #     ] = resized_mask  # Mask is also updated
        #
        #     #resizing mask based on new dimension with jit_factor
        #     cp_mask = cv2.resize(cp_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Mask is also resized
        #
        #     # Perform horizontal flip if required
        #     if FLIP:
        #         cp_mask = cp_mask[:, ::-1]  # Mask is also flipped
        #
        #     padded_mask = np.zeros((max(origin_h, target_h), max(origin_w, target_w)),
        #                            dtype=np.uint8)  # Similar padding for mask
        #
        #     padded_mask[:origin_h, :origin_w] = cp_mask  # Mask also padded similarly
        #
        #     # Random crop
        #     padded_cropped_mask = padded_mask[
        #                           y_offset: y_offset + target_h, x_offset: x_offset + target_w
        #                           ]  # Mask is also cropped similarly
        #
        #     padded_crop_masks.append(padded_cropped_mask)

        cp_scale_ratio *= jit_factor

        # Adjust bounding boxes
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        # Box candidates
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            # padded_crop_masks = np.array(padded_crop_masks)
            # print(f"Padded Crop mask shape : {padded_crop_masks.shape} and padded_cropped_img shape : {padded_cropped_img.shape} and origin_mask shape : {origin_mask.shape}")
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            # origin_mask = origin_mask.astype(np.float32) #typcasting as float for multiplication
            # origin_mask = 0.5 * origin_mask + 0.5 * padded_crop_masks.astype(np.float32)  # Similar blending for mask

        return origin_img.astype(np.uint8), origin_labels, origin_mask.astype(np.uint8)
class MosaicDetection_VID_SAMSeg(Dataset):
    """ VOWSam Version of Detection dataset wrapper that performs mixup for normal dataset.
    - modification in __getitem__ function since now Mask and Confidence score will also be propagated
    """

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0,perspective=0.0,
        enable_mixup=True, mosaic_prob=1.0, mixup_prob=1.0, dataset_path = ''
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            perspective (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.perspective = perspective
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.res = dataset.res
        self.file_num = 0
        self.dataset_path = dataset_path
    def __len__(self):
        return len(self._dataset)
    def get_mosic_idx(self,path):

        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        anno_path = path_dir.replace("Data","Annotations")
        frame_num = len(os.listdir(anno_path))
        self.file_num = frame_num
        #print(frame_num)
        rand_idx = [random.randint(0,frame_num-1) for _ in range(3)]
        raw = '000000'
        res = []
        res.append(path)
        for idx in rand_idx:
            str_idx = str(idx)
            frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
            res.append(frame_idx)
        return res

    #@Dataset.mosaic_getitem
    def __getitem__(self, idx):
        if self.enable_mosaic and random.random() < self.mosaic_prob:

            # print(f"ENABLE MOSAIC WORKING HERE ")
            mosaic_labels = []
            input_dim = self._dataset.input_dim
            input_h, input_w = input_dim[0], input_dim[1]

            # yc, xc = s, s  # mosaic center x, y
            yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
            xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

            indices = self.get_mosic_idx(idx)
            # 3 additional image indices
            #indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

            for i_mosaic, index in enumerate(indices):
                # img, _labels, _, img_id = self._dataset.pull_item(index)
                img, _labels, _masks, confidence_scores, _, _ = self._dataset.pull_item(index)

                h0, w0 = img.shape[:2]  # orig hw
                scale = min(1. * input_h / h0, 1. * input_w / w0)
                img = cv2.resize(
                    img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                )
                _masks = cv2.resize(
                    _masks, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_NEAREST
                )

                # generate output mosaic image
                (h, w, c) = img.shape[:3]
                if i_mosaic == 0:
                    mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
                    mosaic_mask = np.zeros((input_h * 2, input_w * 2), dtype=np.uint8)

                # suffix l means large image, while s means small image in mosaic aug.
                (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                    mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                )

                mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                mosaic_mask[l_y1:l_y2, l_x1:l_x2] = _masks[s_y1:s_y2, s_x1:s_x2]
                padw, padh = l_x1 - s_x1, l_y1 - s_y1

                labels = _labels.copy()
                # Normalized xywh to pixel xyxy format
                if _labels.size > 0:
                    labels[:, 0] = scale * _labels[:, 0] + padw
                    labels[:, 1] = scale * _labels[:, 1] + padh
                    labels[:, 2] = scale * _labels[:, 2] + padw
                    labels[:, 3] = scale * _labels[:, 3] + padh
                mosaic_labels.append(labels)

            if len(mosaic_labels):
                mosaic_labels = np.concatenate(mosaic_labels, 0)
                np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

            mosaic_img, mosaic_labels, mosaic_mask = random_perspective_mask(
                mosaic_img,
                mosaic_labels,
                mosaic_mask,
                degrees=self.degrees,
                translate=self.translate,
                scale=self.scale,
                shear=self.shear,
                perspective=self.perspective,
                border=[-input_h // 2, -input_w // 2],
            )  # border to remove

            # mosaic_img, mosaic_labels = random_perspective(
            #     mosaic_img,
            #     mosaic_labels,
            #     degrees=self.degrees,
            #     translate=self.translate,
            #     scale=self.scale,
            #     shear=self.shear,
            #     perspective=self.perspective,
            #     border=[-input_h // 2, -input_w // 2],
            # )  # border to remove

            # -----------------------------------------------------------------
            # CopyPaste: https://arxiv.org/abs/2012.07177
            # -----------------------------------------------------------------
            if (
                self.enable_mixup
                and not len(mosaic_labels) == 0
                and random.random() < self.mixup_prob
            ):

                mosaic_img, mosaic_labels, mosaic_mask = self.mixup(mosaic_img, mosaic_labels, mosaic_mask,
                                                                     self.input_dim, idx)
                # mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim,idx)

            mix_img, padded_labels, padded_masks = self.preproc(mosaic_img, mosaic_labels, mosaic_mask, self.input_dim)

            img_info = (mix_img.shape[1], mix_img.shape[0])

            return mix_img, padded_labels, padded_masks, confidence_scores, img_info, idx#np.array([idx])

        else:
            self._dataset._input_dim = self.input_dim
            #Modification here
            img, label, masks, confidence_scores, img_info, idx= self._dataset.pull_item(idx)
            # print(f"mask BEFORE prepocessin Mosaic, {label} {masks}")
            img, label, masks = self.preproc(img, label, masks, self.input_dim)
            # print(f"mask after prepocessin Mosaic, {label} { masks}")

            return img, label, masks, confidence_scores, img_info, idx#np.array([idx])

    def get_mixup_idx(self,path):
        path = os.path.join(self.dataset_path,path)
        path_dir = path[:path.rfind('/')+1]
        frame_num = self.file_num
        rand_idx = random.randint(0,frame_num-1)
        str_idx = str(rand_idx)
        raw = '000000'
        frame_idx = path_dir + raw[0:-len(str_idx)] + str_idx + '.JPEG'
        return frame_idx

    def mixup(self, origin_img, origin_labels, origin_mask, input_dim, path):
        '''
        Overriding mixup function from the base class
        to handle mask along with image and targets
        Args:
            origin_img:
            origin_mask:
            origin_labels:
            input_dim:
            path:

        Returns:
            image
            labels
            mask
        '''
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []

        # while len(cp_labels) == 0:
        #     cp_index = random.randint(0, self.__len__() - 1)
        #     cp_labels = self._dataset.load_anno(cp_index)

        cp_index = self.get_mixup_idx(path)
        img, cp_labels, mask, _, _, _ = self._dataset.pull_item(cp_index)


        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
            cp_mask = np.zeros(input_dim, dtype=np.uint8)  # Mask also has a similar starting point
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114
            cp_mask = np.zeros(input_dim, dtype=np.uint8)  # Mask also has a similar starting point

        # Compute scale ratio and resize the copy image and mask
        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )
        resized_mask = cv2.resize(mask, (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
                                  interpolation=cv2.INTER_NEAREST)  # Mask is also resized

        # Assign resized image and mask to the copy image and mask
        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img
        cp_mask[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_mask  # Mask is also updated

        # Resize again based on jit factor
        # Compute new dimensions based on jit factor
        new_w = int(cp_img.shape[1] * jit_factor)
        new_h = int(cp_img.shape[0] * jit_factor)

        # Resize cp_img and cp_mask based on the new dimensions
        cp_img = cv2.resize(cp_img, (new_w, new_h))
        cp_mask = cv2.resize(cp_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Mask is also resized
        cp_scale_ratio *= jit_factor

        # Perform horizontal flip if required
        if FLIP:
            cp_img = cp_img[:, ::-1, :]
            cp_mask = cp_mask[:, ::-1]  # Mask is also flipped

        # Create padded image and mask with maximum of original and copy sizes
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros((max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8)
        padded_mask = np.zeros((max(origin_h, target_h), max(origin_w, target_w)),
                               dtype=np.uint8)  # Similar padding for mask

        padded_img[:origin_h, :origin_w] = cp_img
        padded_mask[:origin_h, :origin_w] = cp_mask  # Mask also padded similarly

        # Random crop
        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]
        padded_cropped_mask = padded_mask[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]  # Mask is also cropped similarly

        # Adjust bounding boxes
        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        # Box candidates
        keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)
            origin_mask = origin_mask.astype(np.float32) #typcasting as float for multiplication
            origin_mask = 0.5 * origin_mask + 0.5 * padded_cropped_mask.astype(np.float32)  # Similar blending for mask

        return origin_img.astype(np.uint8), origin_labels, origin_mask.astype(np.uint8)