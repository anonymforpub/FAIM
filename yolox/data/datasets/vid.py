#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import copy
import os
import random

import numpy
from loguru import logger

import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as torchDataset
from torch.utils.data.sampler import Sampler,BatchSampler,SequentialSampler
from torch.nn.utils.rnn import pad_sequence
from xml.dom import minidom
import math
from yolox.utils import xyxy2cxcywh

#SAM Mask generation Imports
import json
from pycocotools.coco import COCO
# from coco_video_parser import CocoVID

#SAM IMPORTS
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
import cv2
import os
from skimage import measure
from xml.dom.minidom import Document
import re
from tqdm import tqdm




IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png",".JPEG"]
XML_EXT = [".xml"]
name_list = ['n02691156','n02419796','n02131653','n02834778','n01503061','n02924116','n02958343','n02402425','n02084071','n02121808','n02503517','n02118333','n02510455','n02342885','n02374451','n02129165','n01674464','n02484322','n03790512','n02324045','n02509815','n02411705','n01726692','n02355227','n02129604','n04468005','n01662784','n04530566','n02062744','n02391049']
numlist = range(30)
name_num = dict(zip(name_list,numlist))

class VIDDataset(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        file_path="../../dataset_annotations/imagenetVID_2015/YOLOV_annotations/train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 18,
        gframe = 6,
        val = False,
        mode='random',
        dataset_pth = '../../dataset_annotations/imagenetVID_2015',
        tnum = 1000,
        generate_mask=False,
        zip_data=False

    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.tnum = tnum
        self.input_dim = img_size
        self.file_path = file_path
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.res = self.photo_to_sequence(self.file_path,lframe,gframe)
        self.dataset_pth = dataset_pth
        self.generate_mask=generate_mask
        self.zip_data=zip_data




    def __len__(self):
        return len(self.res)


    def photo_to_sequence(self,dataset_path,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frames in a video dir
        Returns:
            split result
        '''
        res = []
        dataset = np.load(dataset_path,allow_pickle=True).tolist()
        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)
        # test = []
        # for ele in res:
        #     test.extend(ele)
        # random.shuffle(test)
        # i = 0
        # for ele in res:
        #     for j in range(gframe):
        #         ele[j] = test[i]
        #         i += 1

        if self.val:
            random.seed(42)
            random.shuffle(res)
            if self.tnum == -1:
                return res
            else:
                return res[:self.tnum]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res[:15000]

    def get_annotation(self,path,test_size):
        path = path.replace("Data","Annotations").replace("JPEG","xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        for xmls in files:
            photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:
                    #tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
                    tempnode.append(( x1, y1, x2, y2,name_num[nameNode],))
            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]
            # print(f"BBox before trans : {res}")
            res[:, :-1] *= r
            anno_res.append(res)
        return anno_res

    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        #ONLY USE THIS LINE WHEN ACCESSING DATA THROUGH ZIPS
        if self.zip_data:
            path = path.replace("ILSVRC2015/Data", "ILSVRC2015_cluster_mount/Data")

        path = os.path.join(self.dataset_pth,path)

        annos = self.get_annotation(path, self.img_size)[0]

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos, img_info, path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path


class SAMVIDDataset(torchDataset):
    """
    - SAMVID sequence class created to load dataset
        to generate mask from SAM.
    - Modifications to VIDDAtaset class in several places:
        - photosequence becomes complete becaue it fetches all
        data from train_seq.npy file
        - two functions (get_annotation and pull_item)
        to avoid preprocessing and data comes in raw GT form such that
        masks can be predicted and stored into corresponding object location
        in each XML.
        - ALSO DIFFERENT DATA LOADER AND COLLATE FUNCTION IS USED TO discard
        samples containing no BBox information in XML files.
    """

    def __init__(
        self,
        file_path="../../dataset_annotations/imagenetVID_2015/YOLOV_annotations/train_seq.npy",
        img_size=(416, 416),
        preproc=None,
        lframe = 18,
        gframe = 6,
        val = False,
        mode='random',
        dataset_pth = '../../dataset_annotations/imagenetVID_2015',
        tnum = 1000,
        generate_mask=False

    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        self.tnum = tnum
        self.input_dim = img_size
        self.file_path = file_path
        self.mode = mode  # random, continous, uniform
        self.img_size = img_size
        self.preproc = preproc
        self.val = val
        self.res = self.photo_to_sequence_complete(self.file_path,lframe,gframe)
        self.dataset_pth = dataset_pth
        self.generate_mask=generate_mask

    def __len__(self):
        return len(self.res)

    def photo_to_sequence_complete(self,dataset_path,lframe,gframe):
        '''
            This function is overriden from the parent class.
            Since we want to generate mask in two ways:
            1- Complete dataset therfore no random sampling,
            so that we have complete dataset.
            2- Specific 10% random sampling will be done later

        Args:
            dataset_path: list,every element is a list contain all frames in a video dir
        Returns:
            complete result.
        '''
        res = []

        dataset = np.load(dataset_path,allow_pickle=True).tolist()

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)
        #Passing as wrapped in another list other wise
        # __get_item__ function does not work
        return res

    def get_annotation(self,path,test_size):
        xmls = path.replace("Data","Annotations").replace("JPEG","xml")

        anno_res = []
        photoname = xmls.replace("Annotations","Data").replace("xml","JPEG")
        file = minidom.parse(xmls)
        root = file.documentElement
        objs = root.getElementsByTagName("object")
        width = int(root.getElementsByTagName('width')[0].firstChild.data)
        height = int(root.getElementsByTagName('height')[0].firstChild.data)
        tempnode = []
        for obj in objs:
            nameNode = obj.getElementsByTagName("name")[0].firstChild.data
            xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
            xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
            ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
            ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
            x1 = np.max((0,xmin))
            y1 = np.max((0,ymin))
            x2 = np.min((width,xmax))
            y2 = np.min((height,ymax))
            if x2 >= x1 and y2 >= y1:
                #tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
                tempnode.append(( x1, y1, x2, y2,name_num[nameNode],))
        num_objs = len(tempnode)
        res = np.zeros((num_objs, 5))
        # r = min(test_size[0] / height, test_size[1] / width)
        for ix, obj in enumerate(tempnode):
            res[ix, 0:5] = obj[0:5]

        # Discarding Sample that does not have BBOX info in XML
        if not res.size:
            return None
        #     masks = self.genrate_mask(res, photoname, visualize=False)
        #     self.write_annotation(xmls, path, masks, res)
        anno_res.append(res)
            # res[:, :-1] *= r
        return anno_res

    def write_annotation(self, xml, path, masks, res):

        file = minidom.parse(xml)
        root = file.documentElement
        objs = root.getElementsByTagName("object")

        # Create a new XML document.
        new_doc = Document()

        # Clone the root node from the old doc to the new one.
        new_root = new_doc.importNode(root, deep=False)
        new_doc.appendChild(new_root)

        # Clone the folder, filename, source, and size nodes from the old doc to the new one.
        for node_name in ["folder", "filename", "source", "size"]:
            old_node = root.getElementsByTagName(node_name)[0]
            new_node = new_doc.importNode(old_node, deep=True)
            new_root.appendChild(new_node)

        # Iterate over the objects.
        for i, obj in enumerate(objs):
            # Clone each object node from the old doc to the new one.
            new_obj = new_doc.importNode(obj, deep=True)
            new_root.appendChild(new_obj)

            # Get the corresponding mask.
            mask = masks[i]

            # Create a new 'mask' element and add it to the object node.
            mask_element = new_doc.createElement("mask")

            # Iterate over the mask points and add them to the 'mask' element.
            for point in mask:
                point_element = new_doc.createElement("point")

                # Combine x and y into a single text node.
                #Condition since point can be a list of list or simple list
                if len(point)==2:
                    # print(point)
                    point_text = new_doc.createTextNode(f"{point[0]},{point[1]}")
                else:
                    # print(point)
                    point_text = new_doc.createTextNode(f"{point[0][0]},{point[0][1]}")

                point_element.appendChild(point_text)
                mask_element.appendChild(point_element)

            new_obj.appendChild(mask_element)

        # Write the new XML content to the new file.
        xml_str = new_doc.toprettyxml(indent="  ")

        # Replace lines containing only whitespaces with empty lines.
        xml_str = re.sub(r'\n\s*\n', '\n', xml_str)

        # Remove the first line, which is the xml version declaration.
        xml_str = '\n'.join(xml_str.split('\n')[1:])

        # Write the new XML content to the new file by using same function.
        self.write_file(xml_str, xml, "xml")

    def genrate_mask(self, anno_res, image_path, visualize=False):

        contours_thresholhold = 500
        segmentation = []
        # converting image tensor to SAM required input
        image = cv2.imread(image_path)
        # Scale the values to the range [0, 255]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)

        #Initalizing SAM

        predictor = SamPredictor(self.SAM)

        #Setting image for SAM
        predictor.set_image(image)

        # Create a copy of the original image to work on
        image_copy = image.copy()

        for ind in range(len(anno_res)):
            mask, confidence_score, _ = predictor.predict(point_coords=None, point_labels=None, box=anno_res[ind][:4],
                                           multimask_output=False)
            # print(f"preicted mask shape : {mask.shape} and confidence_score: {confidence_score} and image_path: {image_path}")
            mask = (mask.squeeze() * 255).astype(np.uint8)
            #finding contours through Skimage function
            contours = measure.find_contours(mask, 0.5)

            #Taking the biggest of the contour since we know single mask is required for each bbox
            contour = max(contours, key=len)

            #Flipping so x axis have first value
            contour = np.float32(np.flip(contour, axis=1))
            # Approximate contour if it has more than contours_thresholhold points
            if len(contour) > contours_thresholhold:
                # print(f"LENGHT OF BIGGER CONTOUR : {len(contour)} and IMAGE PATH : {image_path} ")
                # Set initial epsilon as 0.05 * arcLength
                epsilon = 0.0005 * cv2.arcLength(contour, True)
                # print(f"EPSILON VALUE : {epsilon} AND CURRENT LENGTH OF CONTOUR : {len(contour)}\n\n")
                contour = cv2.approxPolyDP(contour, epsilon, True)
                if len(contour) < 50:
                    print(f"TRANSFORMED LENGHT OF BIGGEST CONTOUR : {len(contour)} IMAGE PATH : {image_path} ")

            polygon_np = np.array(contour, dtype=np.int32)
            # print(f"transformed POLYGON SHAPE {polygon_np.shape} LENGHT OF CONTOURS {len(contour)}")
            #Draw bbox and Polygon mask on the image
            if visualize:
                #draw Polygon mask
                cv2.fillPoly(image_copy, [polygon_np], (0, 0, 255))
                # Convert bbox coordinates to integers and Draw bounding box
                bbox = [int(coordinate) for coordinate in anno_res[ind][:4].tolist()]
                cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                #Write Confidence Score of SAM for the predicted mask
                # Calculate the position for the text
                text_x = bbox[0]
                text_y = bbox[1] + 10  # Adjust the value to change the vertical position of the text
                # Add the text on the image
                cv2.putText(image_copy, str(confidence_score[0]), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

            #Converting to List for writing in XML
            segmentation.append(polygon_np.tolist())
        # Replace the original image with the modified one
        image = image_copy

        #Save image following same path by modifying folder name split_with_mask
        self.write_file(image, image_path, "image") if visualize else None

        return segmentation

    def file_exist(self, file_path):
        '''
        Verify if file (XML file exists then do not do the whole process)
        This function is specifically created during Mask generation
        Args:
            file_path:
        Returns:
            True or False
        '''
        if "train" in file_path:
            file_path = file_path.replace("/train/", "/train_with_masks/")
        else:
            file_path = file_path.replace("/val/", "/val_with_masks/")

        return os.path.exists(file_path)


    def write_file(self, file, file_path, file_type):
        '''
        A generate function to write file into same path with adding mask folder
        Can write both XML and JPEG file, based on file_type argument
        Args:
            file: File object can be numpy array or XMLstring object
            file_path: Data path which will be modied
            file_type: ATM can be "image" or "xml" only

        Returns:

        '''

        if "train" in file_path:
            prefix, suffix = file_path.split("/train/")
            file_write_dir = os.path.join(prefix, "train_with_masks", suffix.split(os.path.basename(file_path))[0])
        else:
            prefix, suffix = file_path.split("/val/")
            file_write_dir = os.path.join(prefix, "val_with_masks", suffix.split(os.path.basename(file_path))[0])

        if not os.path.exists(file_write_dir):
            os.makedirs(file_write_dir)

        if file_type =="image":
            cv2.imwrite(os.path.join(file_write_dir,os.path.basename(file_path)), file)

        elif file_type =="xml":
            # Write the new XML content to the new file.
            # print(f"file path : {os.path.join(file_write_dir,os.path.basename(file_path))}")
            with open(os.path.join(file_write_dir,os.path.basename(file_path)), "w") as f:
                f.write(file)
        else:
            sys.exit("File type not supported.")

    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = os.path.join(self.dataset_pth,path)

        annos = self.get_annotation(path, self.img_size)
        annos = annos[0] if annos is not None else None


        img = cv2.imread(path)
        height, width = img.shape[:2]

        img_info = (height, width)
        # r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        # img = cv2.resize(
        #     img,
        #     (int(img.shape[1] * r), int(img.shape[0] * r)),
        #     interpolation=cv2.INTER_LINEAR,
        # ).astype(np.uint8)
        return img.astype(np.uint8), annos, img_info, path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path

class SAMVIDMaskDataset(VIDDataset):
    """
    - SAMVIDMask sequence class created to load dataset
        to load mask as well beside other GT for visualization.
    - This Class inherits VIDDataset class. becase it is
    similar to normal VIDdataset class but it includes mask
    - Modifications in one function (get_annotation_with_mask)
        to fetch Mask polygon points for each bounding box.
        if present in the XML file.
    - Later, it can be merged with VIDDataset Class
    """

    def __init__(self,
                 dataset_pth='../../dataset_annotations/imagenetVID_2015',
                 zip_data=False,
                 roi_mask_size=28, *args, **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.dataset_pth = dataset_pth
        self.zip_data = zip_data
        self.roi_mask_size = (roi_mask_size,roi_mask_size)

    def get_annotation_with_mask(self,path,test_size):
        '''
        - This function parses the mask from the data
        - Also preprocess and resize it accordingly
        - Rest is same as bbox coordinates
        Args:
            path:
            test_size:

        Returns:
            anno_res : a list of annotation for each image also containing mask
                    as polygon points
        '''
        # #TEMPORARY REPLACE LINE TO PARSE MASKVITH DATA
        # path = path.replace("train/", "VIT_H_train_with_masks/")
        # print("file_path", path)

        path = path.replace("Data","Annotations").replace("JPEG","xml")
        if os.path.isdir(path):
            files = get_xml_list(path)
        else:
            files = [path]
        files.sort()
        anno_res = []
        segmentation_masks=[]
        confidence_scores = []
        for xmls in files:

            file = minidom.parse(xmls)
            root = file.documentElement
            objs = root.getElementsByTagName("object")
            width = int(root.getElementsByTagName('width')[0].firstChild.data)
            height = int(root.getElementsByTagName('height')[0].firstChild.data)
            tempnode = []
            for obj in objs:
                nameNode = obj.getElementsByTagName("name")[0].firstChild.data
                xmax = int(obj.getElementsByTagName("xmax")[0].firstChild.data)
                xmin = int(obj.getElementsByTagName("xmin")[0].firstChild.data)
                ymax = int(obj.getElementsByTagName("ymax")[0].firstChild.data)
                ymin = int(obj.getElementsByTagName("ymin")[0].firstChild.data)
                x1 = np.max((0,xmin))
                y1 = np.max((0,ymin))
                x2 = np.min((width,xmax))
                y2 = np.min((height,ymax))
                if x2 >= x1 and y2 >= y1:

                    # tempnode.append((name_num[nameNode],x1,y1,x2,y2,))
                    tempnode.append((x1, y1, x2, y2, name_num[nameNode], ))

                    # Check for mask information here
                    mask_elements = obj.getElementsByTagName("mask")
                    if mask_elements:
                        # added code for extracting mask polygon points
                        mask = mask_elements[0].firstChild.data
                        mask_points = [int(coordinate) for point in mask.split(';') for coordinate in point.split(',')]

                        # added code for extracting confidence
                        confidence = float(obj.getElementsByTagName("mask")[0].getAttribute("confidence"))

                    #When mask is not present, we create it by following BoxMask strategy
                    #Condfidence is just random value which should be verified later
                    else:
                        mask_points = [x1,y1, x2,y1, x2,y2, x1,y2, x1,y1]
                        print("MASK not AVAILABLE, creating our own with bboxes", mask_points)
                        confidence = float(0.9981505)

                    # append mask points and confidence to mask_data
                    segmentation_masks.append(mask_points)
                    confidence_scores.append(confidence)

            num_objs = len(tempnode)
            res = np.zeros((num_objs, 5))
            r = min(test_size[0] / height, test_size[1] / width)
            for ix, obj in enumerate(tempnode):
                res[ix, 0:5] = obj[0:5]

            res[:, :-1] *= r
            anno_res.append(res)

        return anno_res, segmentation_masks, np.array(confidence_scores)
    def pull_item_separate_binary_masks_each_image_size(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.
                This function is needed since we are returning more variables now including mask,
                confidence and so on.
                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        #ONLY USE THIS LINE WHEN ACCESSING DATA THROUGH ZIPS
        # if self.zip_data:
        #     path = path.replace("ILSVRC2015/Data", "ILSVRC2015_cluster_mount/Data")

        path = os.path.join(self.dataset_pth, path)
        annos, masks, confidence_scores = self.get_annotation_with_mask(path, self.img_size)

        annos = annos[0]
        # print(masks)
        if len(masks) < 1 and len(annos) > 0:
            print(f"Problem in this File ; {path}")
            print(f"masks in this File ; {masks} and annos : {annos}")

        # print(f"Loading... File ; {path}")
        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)



        #Now resize both image and segmentation mask (gt_mask)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        # Resize each mask individually

        #initializing GT Masks with resizes image shape
        gt_masks = np.zeros((len(annos), img.shape[0], img.shape[1]), dtype=np.uint8)

        # FIRST Converting GT polygon points to label encoded mask
        # this allows consistency between mask and image
        # then resize them and do it in a sinlge loop
        for i, (annotation, segmentation_mask) in enumerate(zip(annos, masks)):
            #initializing with older image resolution for consistency with annotations
            gt_mask = np.zeros((height, width), dtype=np.uint8)
            polygon = np.array(segmentation_mask).reshape(-1, 2).astype(int)
            cv2.fillPoly(gt_mask, [polygon], 1)  # assuming the last value in annotation is the class label
            resized_mask = cv2.resize(
                gt_mask,
                (int(gt_mask.shape[1] * r), int(gt_mask.shape[0] * r)),
                interpolation=cv2.INTER_NEAREST,  # Use INTER_NEAREST for mask images
            ).astype(np.uint8)
            gt_masks[i] = resized_mask

        # print(f"SHAPE AFTER PULL ITEM : {gt_masks.shape} {img.shape} {len(annos)} ")
        return img, annos, gt_masks, confidence_scores, img_info, path
    def pull_item(self,path):
        """
        This pull item variant is designed to generate masks with ROI defined resolution
        - The gt_mask for each instance will be equal to the actual image size
        - It will be rescaled in the same fashion
        - Rest is same as other pull item function with mask
        Args:
            path:

        Returns:
            img, annos, gt_masks, confidence_scores, img_info, path
        """
        #ONLY USE THIS LINE WHEN ACCESSING DATA THROUGH ZIPS
        if self.zip_data:
            path = path.replace("ILSVRC2015/Data", "ILSVRC2015_cluster_mount/Data")
            # print("new path data, ", path)

        path = os.path.join(self.dataset_pth, path)
        annos, masks, confidence_scores = self.get_annotation_with_mask(path, self.img_size)

        annos = annos[0]
        # print(masks)
        if len(masks) != len(annos):
            print(f"Problem in this File ; {path}")
            print(f"masks in this File ; {masks} and annos : {annos}")

        # print(f"Loading... File ; {path}")
        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)



        #Now resize both image and segmentation mask (gt_mask)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        # Resize each mask individually

        #initializing GT Masks with resizes image shape
        # gt_masks = np.zeros((len(annos), *self.roi_mask_size), dtype=np.uint8)
        gt_masks = np.zeros((len(annos), img.shape[0], img.shape[1]), dtype=np.uint8)

        # FIRST Converting GT polygon points to label encoded mask
        # this allows consistency between mask and image
        # then resize them and do it in a sinlge loop
        for i, (annotation, segmentation_mask) in enumerate(zip(annos, masks)):
            #initializing with newer image resolution because we multiply polygon with r before creating mask
            gt_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            polygon = (np.array(segmentation_mask).reshape(-1, 2) * r).astype(int)
            cv2.fillPoly(gt_mask, [polygon], 1)  # assuming the last value in annotation is the class label

            #FOR CHECKING, EXTRA CODE REMOVE LATER, ensure mask is created
            if np.sum(gt_mask) == 0:
                print(f"Problem in this File ; {path}")

            gt_masks[i] = gt_mask

        # print(f"SHAPE AFTER PULL ITEM : {gt_masks.shape} {img.shape} {len(annos)} ")
        return img, annos, gt_masks, confidence_scores, img_info, path
    def pull_item_semantic_seg(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.
                This function is needed since we are returning more variables now including mask,
                confidence and so on.
                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = os.path.join(self.dataset_pth,path)
        annos, masks, confidence_scores = self.get_annotation_with_mask(path, self.img_size)

        annos = annos[0]
        # print(masks)
        if len(masks) < 1 and len(annos) > 0:
            print(f"Problem in this File ; {path}")
            print(f"masks in this File ; {masks} and annos : {annos}")

        # print(f"Loading... File ; {path}")
        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)

        # FIRST Converting GT polygon points to label encoded mask
        # this allows consistency between mask and image
        gt_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for annotation, segmentation_mask in zip(annos, masks):
            polygon = np.array(segmentation_mask).reshape(-1, 2).astype(int)
            cv2.fillPoly(gt_mask, [polygon], int(annotation[-1]))  # assuming the last value in annotation is the class label

        #Now resize both image and segmentation mask (gt_mask)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        gt_mask = cv2.resize(gt_mask, (int(width * r), int(height * r)), interpolation=cv2.INTER_NEAREST).astype(
            np.uint8)

        return img, annos, gt_mask, confidence_scores, img_info, path
    def pull_item_to_visualize_mask(self,path):
        """
            Special Function, only required when visualizing GT Mask.
            - One image / label pair for the given index is picked up and pre-processed.
            - This function is needed since we are returning more variables now including mask,
            confidence without any pre-processing.
            Args:
                index (int): data index
            Returns:
                img (numpy.ndarray): pre-processed image
                padded_labels (torch.Tensor): pre-processed label data.
                    The shape is :math:`[max_labels, 5]`.
                    each label consists of [class, xc, yc, w, h]:
                        class (float): class index.
                        xc, yc (float) : center of bbox whose values range from 0 to 1.
                        w, h (float) : size of bbox whose values range from 0 to 1.
                info_img : tuple of h, w.
                    h, w (int): original shape of the image
                img_id (int): same as the input index. Used for evaluation.
            """
        path = os.path.join(self.dataset_pth,path)

        annos, masks, confidence_scores = self.get_annotation_with_mask(path, self.img_size)

        annos = annos[0]
        # print(masks)
        if len(masks) < 1:
            print(f"Problem in this File ; {path}")
            print(f"masks in this File ; {masks}")
            # masks = None

        img = cv2.imread(path)
        height, width = img.shape[:2]
        img_info = (height, width)

        return img.astype(np.uint8), annos, masks, confidence_scores, img_info, path
    def __getitem__(self, path):
        '''
        This function is needed since we are returning more variables now including mask,
                confidence and so on.
        Args:
            path:
        Returns:

        '''
        img, target, masks, confidence_scores, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            # img, target = self.preproc(img, target, self.input_dim)

            img, target, masks = self.preproc(img, target, masks, self.input_dim)

        # print(f"SHAPE AFTER GET ITEM : {masks.shape} {img.shape} {len(target)} ")
        return img, target, masks, confidence_scores, img_info,path


class Arg_VID(torchDataset):
    """
    VID sequence
    """

    def __init__(
        self,
        data_dir='/media/tuf/ssd/Argoverse-1.1/',
        img_size=(416, 640),
        preproc=None,
        lframe = 0,
        gframe = 16,
        val = False,
        mode='random',
        COCO_anno = '',
        name = "tracking",
    ):
        """
        COCO dataset initialization. Annotation data are read into memory by COCO API.
        Args:
            data_dir (str): dataset root directory
            json_file (str): COCO json file name
            name (str): COCO data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__()
        self.input_dim = img_size
        self.name = name
        self.val = val
        self.data_dir = data_dir
        self.img_size = img_size
        self.coco_anno_path = COCO_anno
        self.name_id_dic = self.get_NameId_dic()
        self.coco = COCO(COCO_anno)
        # remove_useless_info(self.coco)
        self.ids = sorted(self.coco.getImgIds())
        self.class_ids = sorted(self.coco.getCatIds())
        cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in cats])
        self.annotations = self._load_coco_annotations()
        self.mode = mode  # random, continous, uniform
        self.preproc = preproc

        self.res = self.photo_to_sequence(lframe,gframe)

    def get_NameId_dic(self):
        img_dic = {}
        with open(self.coco_anno_path,'r') as train_anno_content:
            train_anno_content = json.load(train_anno_content)
            for im in train_anno_content['images']:
                img_dic[im['name']] = im['id']
        return img_dic

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def __len__(self):
        return len(self.res)

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )
        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe, seq_len = 192):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['sequences']))]
            for im in anno['images']:
                dataset[im['sid']].append(self.coco.dataset['seq_dirs'][im['sid']] + '/' + im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    # split_num = int(ele_len / (gframe))
                    # random.shuffle(element)
                    # for i in range(split_num):
                    #     res.append(element[i * gframe:(i + 1) * gframe])
                    # if self.val and element[(i + 1) * gframe:] != []:
                    #     res.append(element[(i + 1) * gframe:])

                    seq_split_num = int(len(element) / seq_len)
                    for k in range(seq_split_num + 1):
                        tmp = element[k * seq_len:(k + 1) * seq_len]
                        if tmp == []:continue
                        random.shuffle(tmp)
                        split_num = int(len(tmp) / (gframe))
                        for i in range(split_num):
                            res.append(tmp[i * gframe:(i + 1) * gframe])
                        if self.val and tmp[(i + 1) * gframe:] != []:
                            res.append(tmp[(i + 1) * gframe:])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            # random.seed(42)
            # random.shuffle(res)
            return res#[:1000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:1000]#[:15000]


    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        path = path.split('/')[-1]
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir, self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path

    def __getitem__(self, path):

        img, target, img_info, path = self.pull_item(path)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info,path


class OVIS(Arg_VID):
    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        #im_ann['name'] = self.coco.dataset['seq_dirs'][im_ann['sid']] + '/' + im_ann['name']
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)
        objs = []
        for obj in annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))
            if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                obj["clean_bbox"] = [x1, y1, x2, y2]
                objs.append(obj)

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["name"]
            if "name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        return (res, img_info, resized_info, file_name)

    def photo_to_sequence(self,lframe,gframe):
        '''

        Args:
            dataset_path: list,every element is a list contain all frame in a video dir
        Returns:
            split result
        '''
        res = []

        with open(self.coco_anno_path, 'r') as anno:
            anno = json.load(anno)
            dataset = [[] for i in range(len(anno['videos']))]
            for im in anno['images']:
                dataset[im['sid']].append(im['name'])
            for ele in dataset:
                sorted(ele)

        for element in dataset:
            ele_len = len(element)
            if ele_len<lframe+gframe:
                #TODO fix the unsolved part
                #res.append(element)
                continue
            else:
                if self.mode == 'random':
                    split_num = int(ele_len / (gframe))
                    random.shuffle(element)
                    for i in range(split_num):
                        res.append(element[i * gframe:(i + 1) * gframe])
                elif self.mode == 'uniform':
                    split_num = int(ele_len / (gframe))
                    all_uniform_frame = element[:split_num * gframe]
                    for i in range(split_num):
                        res.append(all_uniform_frame[i::split_num])
                elif self.mode == 'gl':
                    split_num = int(ele_len / (lframe))
                    all_local_frame = element[:split_num * lframe]
                    for i in range(split_num):
                        g_frame = random.sample(element[:i * lframe] + element[(i + 1) * lframe:], gframe)
                        res.append(all_local_frame[i * lframe:(i + 1) * lframe] + g_frame)
                else:
                    print('unsupport mode, exit')
                    exit(0)

        if self.val:
            random.seed(42)
            random.shuffle(res)
            return res#[2000:3000]#[1000:1250]#[2852:2865]
        else:
            random.shuffle(res)
            return res#[:15000]

    def pull_item(self,path):
        """
                One image / label pair for the given index is picked up and pre-processed.

                Args:
                    index (int): data index

                Returns:
                    img (numpy.ndarray): pre-processed image
                    padded_labels (torch.Tensor): pre-processed label data.
                        The shape is :math:`[max_labels, 5]`.
                        each label consists of [class, xc, yc, w, h]:
                            class (float): class index.
                            xc, yc (float) : center of bbox whose values range from 0 to 1.
                            w, h (float) : size of bbox whose values range from 0 to 1.
                    info_img : tuple of h, w.
                        h, w (int): original shape of the image
                    img_id (int): same as the input index. Used for evaluation.
                """
        idx = self.name_id_dic[path]
        annos, img_info, resized_info, img_path = self.annotations[idx]
        abs_path = os.path.join(self.data_dir,self.name, img_path)
        img = cv2.imread(abs_path)

        height, width = img.shape[:2]
        img_info = (height, width)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return img, annos.copy(), img_info, img_path

def get_xml_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in XML_EXT:
                image_names.append(apath)

    return image_names

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def make_path(train_dir,save_path):
    res = []
    for root,dirs,files in os.walk(train_dir):
        temp = []
        for filename in files:
            apath = os.path.join(root, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                temp.append(apath)
        if(len(temp)):
            temp.sort()
            res.append(temp)
    res_np = np.array(res,dtype=object)
    np.save(save_path,res_np)


class TestSampler(SequentialSampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class TrainSampler(Sampler):
    def __init__(self,data_source):
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.data_source.res)
        return iter(self.data_source.res)

    def __len__(self):
        return len(self.data_source)

class VIDBatchSampler(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            for filename in ele:
                batch.append(filename)
                if (len(batch)) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch)>0 and not self.drop_last:
            yield batch
    def __len__(self):
        return len(self.sampler)

class VIDBatchSampler_Test(BatchSampler):
    def __iter__(self):
        batch = []
        for ele in self.sampler:
            yield ele
            # for filename in ele:
            #     batch.append(filename)
            #     if (len(batch)) == self.batch_size:
            #         yield batch
            #         batch = []
            # if len(batch)>0 and not self.drop_last:
            #     yield batch
    def __len__(self):
        return len(self.sampler)

def sam_collate_fn(batch):
    '''
    This function is only used when data (GT mask) needs to be
    generated from actual data using SAM
    - used only for generating mask from SAM
    - Removing additional arguments
    - Checking if no bbox annotation then remove
    that data sample
    Args:
        batch:

    Returns:

    '''
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    for sample in batch:
        if any(v is None for v in sample):
            continue
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        ims_info.append(sample[2])
        path.append(sample[3])
        #path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    # path_sequence= torch.tensor(path_sequence)
    # time_embedding = get_timing_signal_1d(path_sequence,256)

    if len(imgs) == 1:
        return imgs[0], ims_info, tar_ori, path
    elif not imgs:
        return None
    return torch.stack(imgs),ims_info,tar_ori,path

def collate_fn_SAMIns(batch):
    '''
    -This is the new main Collate function for training with main VOWSAM
    - Here ,we have separate mask for each instance in an image.
    - But the total batch will be of size [batch_size, max_instances_per_image, mask_height, mask_width]
    - This enables us to not to worry about matching pred_instance mask with the right GT.
    - We will have an identical array with max. no. of instances in an image in that batch

    -Rest is similar to previous collate function

    Args:
        batch: Containing:
        img, target, masks, img_info,path

    Returns:
    '''
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    all_masks = []
    confidence_scores = []

    max_instances_per_image = max(len(sample[2]) for sample in batch)

    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[4])
        path.append(sample[5])

        #Here process masks and confidence scores
        # Padding masks for this image to have shape [max_instances_per_image, mask_height, mask_width]
        masks_for_this_image = torch.tensor(sample[2])
        padded_masks = torch.zeros([max_instances_per_image, *masks_for_this_image.shape[1:]])
        padded_masks[:masks_for_this_image.shape[0]] = masks_for_this_image
        all_masks.append(padded_masks)

        confidence_scores.append(torch.tensor(sample[3]))

    batched_masks = torch.stack(all_masks)

    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,batched_masks,confidence_scores, path, None

def collate_fn_SAM_SEMSEG_train(batch):
    '''
    -This is the main Collate function for training with SAM when we are doing Semantic segmentation
    -Special collate function that includes mask in each batch sample
    - Just adding mask and confidence as two additional samples in each batch
    Args:
        batch: Containing:
        img, target, masks, img_info,path

    Returns:
    '''
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    masks = []

    confidence_scores = []
    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        masks.append(torch.tensor(sample[2]))
        confidence_scores.append(torch.tensor(sample[3]))
        ims_info.append(sample[4])
        path.append(sample[5])

    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,torch.stack(masks),confidence_scores, path, None
def collate_fn_SAMIns_mask_visualize(batch):
    '''
    -This Collate function is there just to visualize data
    - Special collate function that includes mask in each batch sample
    - For now removing tar_tensor since not required
    later, can be added during actual training
    Args:
        batch: Containing:
        img, target, masks, img_info,path

    Returns:
    '''
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    #Sending mask_info as dict because polygons can be of arbitatry lenghts
    #- two keys, one stating polygons with equal points by padding 0s
    #- bool_mask stores info. for each polygon actual length which
    #is extracted late to avoid 0 values in polygon points
    masks_info = {
        "polygons" : [],
        "bool_mask": []
    }
    confidence_scores = []
    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        # PARSING POLYGON POINTS (masks) WITH DIFFERENT LENGTHS
        # Find the maximum length and make the size equal to
        # pass them as a single batch
        if len(sample[2]) > 1:

            polygons = sample[2]
            max_len = max(len(p) for p in polygons)
            # Create a tensor to hold the padded polygons
            padded_polygons = torch.zeros(len(polygons), max_len, dtype=torch.int32)
            # Create a tensor to hold the masks
            mask_bool = torch.zeros(len(polygons), max_len, dtype=torch.bool)

            for i, polygon in enumerate(polygons):
                padded_polygons[i, :len(polygon)] = torch.Tensor(polygon)  # Fill with polygon points
                mask_bool[i, :len(polygon)] = True  # Indicate where the actual data is
            masks_info["polygons"].append(padded_polygons)
            masks_info["bool_mask"].append(mask_bool)

        else:
            masks_info["polygons"].append(torch.tensor(np.array(sample[2])))

        confidence_scores.append(torch.tensor(sample[3]))
        ims_info.append(sample[4])
        path.append(sample[5])

    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,masks_info,confidence_scores, path, None
def collate_fn(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([120,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(sample[1]))
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        #path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    # path_sequence= torch.tensor(path_sequence)
    # time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,None

def get_vid_loader(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader


def get_trans_loader_sam(batch_size,data_num_workers,dataset):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn':sam_collate_fn
    }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader

def vid_val_loader(batch_size,data_num_workers,dataset,):
    sampler = VIDBatchSampler_Test(TestSampler(dataset),batch_size,drop_last=False)
    dataloader_kwargs = {
        "num_workers": data_num_workers,
        "pin_memory": True,
        "batch_sampler": sampler,
        'collate_fn': collate_fn
    }
    loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return loader

def collate_fn_trans(batch):
    tar = []
    imgs = []
    ims_info = []
    tar_ori = []
    path = []
    path_sequence = []
    for sample in batch:
        tar_tensor = torch.zeros([100,5])
        imgs.append(torch.tensor(sample[0]))
        tar_ori.append(torch.tensor(copy.deepcopy(sample[1])))
        sample[1][:,1:]=xyxy2cxcywh(sample[1][:,1:])
        tar_tensor[:sample[1].shape[0]] = torch.tensor(sample[1])
        tar.append(tar_tensor)
        ims_info.append(sample[2])
        path.append(sample[3])
        path_sequence.append(int(sample[3][sample[3].rfind('/')+1:sample[3].rfind('.')]))
    path_sequence= torch.tensor(path_sequence)
    time_embedding = get_timing_signal_1d(path_sequence,256)
    return torch.stack(imgs),torch.stack(tar),ims_info,tar_ori,path,time_embedding

def get_trans_loader(batch_size,data_num_workers,dataset, mask=False, visualize=False):
    sampler = VIDBatchSampler(TrainSampler(dataset), batch_size, drop_last=False)
    # if mask and visualize:
    #     dataloader_kwargs = {
    #         "num_workers": data_num_workers,
    #         "pin_memory": True,
    #         "batch_sampler": sampler,
    #         'collate_fn': collate_fn_with_mask
    #     }
    if mask:
        dataloader_kwargs = {
            "num_workers": data_num_workers,
            "pin_memory": True,
            "batch_sampler": sampler,
            'collate_fn': collate_fn_SAMIns
        }
    else:
        dataloader_kwargs = {
            "num_workers": data_num_workers,
            "pin_memory": True,
            "batch_sampler": sampler,
            'collate_fn':collate_fn
        }
    vid_loader = torch.utils.data.DataLoader(dataset, **dataloader_kwargs)
    return vid_loader


class DataPrefetcherSAM:
    """
    Copy of DataPrefetcher class
    - Modifications in different functions
    - Adding mask as target_mask everyhwere
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.max_iter = len(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcherSAM._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            output = next(self.loader)
            # self.next_input, self.next_target,_,_,_,self.time_ebdding = next(self.loader)
            if len(output) == 6:
                self.next_input, self.next_target, _, _, _, self.time_ebdding = output
            else:
                self.next_input, self.next_target, _, _, self.next_target_mask, self.next_mask_confidence_score, _, self.time_ebdding = output

        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.time_ebdding = None
            self.next_target_mask = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_target_mask = self.next_target_mask.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        target_mask = self.next_target_mask
        time_ebdding = self.time_ebdding
        mask_confidence_score= self.next_mask_confidence_score
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
            target_mask.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target,target_mask, mask_confidence_score

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.max_iter = len(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            output = next(self.loader)
            # self.next_input, self.next_target,_,_,_,self.time_ebdding = next(self.loader)
            if len(output) == 6:
                self.next_input, self.next_target, _, _, _, self.time_ebdding = output
            else:
                self.next_input, self.next_target, _, _, _, _, _, self.time_ebdding = output

        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.time_ebdding = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        time_ebdding = self.time_ebdding
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target,time_ebdding

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())

def get_timing_signal_1d(index_squence,channels,min_timescale=1.0, max_timescale=1.0e4,):
    num_timescales = channels // 2

    log_time_incre = torch.tensor(math.log(max_timescale/min_timescale)/(num_timescales-1))
    inv_timescale = min_timescale*torch.exp(torch.arange(0,num_timescales)*-log_time_incre)

    scaled_time = torch.unsqueeze(index_squence,1)*torch.unsqueeze(inv_timescale,0) #(index_len,1)*(1,channel_num)
    sig = torch.cat([torch.sin(scaled_time),torch.cos(scaled_time)],dim=1)
    return sig
