#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    This script generates mask by using the required SAM model for DET train set
- checkpoint default is SAM_B however, you can modfiy with argparse
- All functions are written in this file Basides the data loader stuff which comes from vid
- This script is different since it adds the segmentation coordinates in the JSON unlike separate XML files in VID.

'''
import argparse
import random
import warnings
from loguru import logger
import os
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn

from yolox.core import launch

from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
from yolox.data.data_augment import Vid_Val_Transform
from yolox.data.datasets import coco

#SAM Related Imports
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import measure
from xml.dom.minidom import Document
from xml.dom import minidom
import xml.etree.ElementTree as ET
import numpy as np
import sys
import cv2
import re

import json

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--tsize", default=576, type=int, help="test img size")
    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-urlT",
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
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default='../../checkpoints/sam_vit_b_01ec64.pth', type=str, help="checkpoint file")
    parser.add_argument(
        '-data_dir',
        default='',
        type=str,
        help="path to your dataset",

    )
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-v",
        "--val_split",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parser.add_argument('--lframe', default=0, help='local frame num')
    parser.add_argument('--gframe', default=1, help='global frame num')
    parser.add_argument('--mode', default='random', help='frame sample mode')
    parser.add_argument('--tnum', default=-1, help='vid test sequences')
    parser.add_argument('-w', '--num_workers', type=int, default=4, help='number of workers')
    return parser


def update_json_from_SAM(json_file, masks, confidence_scores, file_path, image_id):
    """
    Update the parsed JSON data with segmentation from SAM.

    Args:
    - json_file: Parsed JSON data in memory.
    - masks: List of predicted masks represented as polygon points.
    - confidence_scores: List of confidence scores for the masks.
    - file_path: File path of the image being processed.
    - image_id: ID of the image in the COCO dataset.

    Returns:
    - Updated JSON data.
    """

    # Find the matching annotation in the loaded JSON data and update it
    # Update each corresponding annotation in the loaded JSON data

    obj_index = 0
    for index in range(len(json_file['annotations'])):
        if json_file['annotations'][index]['image_id'] == image_id:
            try:
                mask = masks[obj_index]
                flat_mask = [coord for point in mask for coord in point]
                # print("flat_mask  IN loop", flat_mask)
                # print("BOUNDING BOX FOR THE SAME OBJECT", json_file['annotations'][index]['bbox'])
                json_file['annotations'][index]['segmentations'] = flat_mask
            except IndexError:
                print(f"No prediction mask for image_id: {image_id}. Using bbox as mask.")
                bbox = json_file['annotations'][index]['bbox']
                xmin, ymin, width, height = bbox
                # Create a rectangular mask using the bounding box
                mask = [xmin, ymin, xmin + width, ymin, xmin + width, ymin + height, xmin, ymin + height]
                json_file['annotations'][index]['segmentations'] = [mask]
            obj_index += 1
            if obj_index >= len(masks):  # break if all masks have been added
                break

    return json_file

def masks_to_polygons(predictions, image_path, labels, visualize=False):

    contours_thresholhold = 500
    cannot_visualize_no_mask = False
    segmentation = []
    confidence_scores=[]

    if visualize:
        # Read Image from path
        image = cv2.imread(image_path)
        # Scale the values to the range [0, 255]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(np.uint8)
        # Create a copy of the original image to work on
        image_copy = image.copy()

    for ind, (mask, confidence_score) in enumerate(zip(predictions["masks"],predictions["iou_predictions"])):

        # print(f"\n\n preicted mask shape : {mask.shape} and confidence_score: {confidence_score} and image_path: {image_path}")
        mask = (mask.cpu().numpy().squeeze() * 255).astype(np.uint8)

        #finding contours through Skimage function
        contours = measure.find_contours(mask, 0.5)

        if not np.any(mask == 255):
            cannot_visualize_no_mask = True
            print(f"Image with No Mask Generated from SAM : {image_path} ")
            continue
        #Confidence score to CPU
        confidence_score = confidence_score.cpu().numpy().squeeze()
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
        #This need some modification now, verify bboxes coming from labels
        if visualize and not cannot_visualize_no_mask:

            #draw Polygon mask
            cv2.fillPoly(image_copy, [polygon_np], (0, 0, 255))
            # Convert bbox coordinates to integers and Draw bounding box
            bbox = [int(coordinate) for coordinate in labels[ind][:4].tolist()]
            cv2.rectangle(image_copy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            #Write Confidence Score of SAM for the predicted mask
            # Calculate the position for the text
            text_x = bbox[0]
            text_y = bbox[1] + 10  # Adjust the value to change the vertical position of the text
            # Add the text on the image

            cv2.putText(image_copy, "{:.3f}".format(confidence_score), (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Converting to List for writing in XML

        # Check if the mask has the extra nesting level by examining the first point.
        # it is needed to parse polygons to insert into json file
        if len(polygon_np.shape) == 3:
            polygon_np= polygon_np[:, 0, :]

        segmentation.append(polygon_np.tolist())
        confidence_scores.append(confidence_score)

    if visualize and not cannot_visualize_no_mask:
        # Replace the original image with the modified one
        image = image_copy
        # Save image following same path by modifying folder name split_with_mask
        write_file(image, image_path, "image")

    return segmentation, confidence_scores

def file_exist(file_path, file_type_xml=True):
    '''
    Verify if file (XML file exists then do not do the whole process)
    This function is specifically created during Mask generation
    Args:
        file_path:
    Returns:
        True or False
    '''
    if not file_type_xml:
        file_path = file_path.replace("Data", "Annotations").replace("JPEG", "xml")

    if "train" in file_path:
        file_path = file_path.replace("/train/", "/train_with_masks/")
    else:
        file_path = file_path.replace("/val/", "/val_with_masks/")
    # print("File path : ", file_path)
    return os.path.exists(file_path)
def img_to_xml_path(file_path, exp):
    '''
    Args:
        file_path: JPEG file path
        exp: experiment object that has data_dir and other paths

    Returns:
        file_path: corresponding annotation xml file path
    '''

    file_path = file_path.replace("Data", "Annotations").replace("JPEG", "xml")
    file_path = os.path.join(exp.data_dir,file_path)
    return file_path


def write_file(file, file_path, file_type):
    '''
    A generate function to write file into same path with adding mask folder
    Can write both XML and JPEG file, based on file_type argument
    Args:
        file: File object can be numpy array or XMLstring object
        file_path: Data path which will be modied
        file_type: ATM can be "image" or "xml" only

    Returns:
        Nothing, just writes the file on disk at a given path by adding "_with_masks"
    '''
    if "/train/" in file_path:
        prefix, suffix = file_path.split("/train/")
        file_write_dir = os.path.join(prefix, "train_with_masks", suffix.split(os.path.basename(file_path))[0])
    elif "/val/" in file_path:
        prefix, suffix = file_path.split("/val/")
        file_write_dir = os.path.join(prefix, "val_with_masks", suffix.split(os.path.basename(file_path))[0])
    else:
        file_write_dir = file_path.split(os.path.basename(file_path))[0]

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

def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device='cuda')
    image = image.permute(2, 0, 1).contiguous()
    return image

def genrate_mask_batch_prompts(model, info_imgs, labels, imgs, json_file, image_id):
    '''
    This function takes samples in a batch to compute
    end-to-end batch prompt inference of SAM.
    The generated MASK will be propoagated to Write_file function
    To write the corresponding mask with BBox information as
    Polygon points.
    Once the polygons points are generated, the write_annotation function
    will be called that will generate a copy of XML file and add
    corresponding object of Mask in each XML file that contains bboxes.
    Args:
        info_imgs: Image paths, list of image paths
        labels: list of
        imgs: list of image objects
        json_file: to be in memory such that it can updated on runtime and returned

    Returns:
        json_file, updated json file

    '''
    resize_transform = ResizeLongestSide(model.image_encoder.img_size)
    batched_input = []

    #Transforming image batch FOR SAM

    # print(f"imgs SHAPE : {imgs.shape} and bbox shape now : {labels[0].shape}")
    imgs = imgs.cpu().numpy().astype(np.uint8)
    # print(f"AFTER CPU imgs SHAPE : {imgs.shape}")
    #List that should be treated with SAM masks
    image_list = []
    label_list = []
    #Creating SAM batch input comprising N images and NXM prompts
    for ind, img in enumerate(imgs):

        bbox_preds = labels[ind][:, :4].cpu().detach().numpy()  # [num_boxes_per_frame, 4]

        # print(f"Bbox prediction shape, {bbox_preds.shape[0]}")
        #if no bbox then no mask from SAM
        #Hence r
        if bbox_preds.shape[0] >= 1 and not file_exist(info_imgs[ind], file_type_xml=False):
            image_list.append(info_imgs[ind])
            label_list.append(bbox_preds)
            batched_input.append(
                {
                    'image': prepare_image(img, resize_transform),
                    'boxes': resize_transform.apply_boxes_torch(torch.tensor(bbox_preds, device='cuda'), img.shape[:2]),
                    'original_size': img.shape[:2]
                }
            )

    # print(f" LENGHT batched_input {len(batched_input)}")
    #Model Forward Pass in a complete batch to reduce time
    if len(batched_input)>0:
        batched_output = model(batched_input, multimask_output=False)
        # print(f" LENGHT batched_output {len(batched_output)}")

        #Now iterating over images to get polygon points for each points
        for ind, output in enumerate(batched_output):
            polygons,confidence_scores = masks_to_polygons(output, image_list[ind], label_list[ind], visualize=False)
            json_file = update_json_from_SAM(json_file, polygons, confidence_scores, image_list[ind], image_id)

    return json_file

def update_json_fron_xml_file(xml_file_path, json_file, image_id):

    # print(xml_file_path, image_id)
    # Parse the XML file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    # Extract mask
    segmentation = []
    for obj in root.findall('object'):
        mask_elem = obj.find('mask')
        #it is possible that obj is there with no mask
        if mask_elem is not None:
            mask = mask_elem.text
            mask_coords = mask.split(";")
            poly_points = []
            for coord in mask_coords:
                x, y = coord.split(',')
                poly_points.extend([int(x), int(y)])
            segmentation.append(poly_points)

        else:  # if mask is not available, use the bounding box
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            segmentation.append([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])


    # print("LENGTH OF SEGMNENTATION : ", len(segmentation))
    # Update each corresponding annotation in the loaded JSON data
    obj_index = 0
    for index in range(len(json_file['annotations'])):
        if json_file['annotations'][index]['image_id'] == image_id:
            json_file['annotations'][index]['segmentations'] = segmentation[obj_index]
            obj_index += 1
            if obj_index >= len(segmentation):  # break if all masks have been added
                break

    return json_file

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

    print(f"PRINTING NUM WORKERS {args.num_workers} AND BATCH SIZE {args.batch_size}  AND GFRAME {gframe} ")

    dataset_loader = exp.get_data_loader(batch_size=args.batch_size, is_distributed=False, cache_img=args.cache)

    json_file_path = os.path.join(exp.data_dir, exp.train_ann)
    with open(json_file_path, 'r') as f:
        json_file = json.load(f)

    # SAM STUFF
    # Intialize SAM model and forwrard Pass
    model_registry= re.search(r'sam_(.{5})', os.path.basename(args.ckpt)).group(1)

    print(f"LOADED MODEL REGISTRY : {model_registry} and JSON FILE :{exp.train_ann}")
    print(f"length of the dataset : {len(dataset_loader)}")
    SAM = sam_model_registry[model_registry](checkpoint=args.ckpt).to(device="cuda")

    count = 0
    for batch in tqdm(dataset_loader):

        #condition is needed because batch can be empty
        #We are discarding samples not comprising of bbox info.
        if batch is None:
            continue
        imgs, labels, img_info, img_ids, paths= batch
        # print(imgs.shape, labels, img_info, img_ids, paths)
        if "Data/VID" in paths[0]:
            # json_file = update_json_fron_xml_file(xml_file_path=img_to_xml_path(paths[0], exp), json_file=json_file, image_id=img_ids[0][0])
            continue

        # ADDING ANOTHER DIMENSION IN IMAGES SINCE BATCH SIZE IS 1 for tackling images of various size
        # else:
        # if count == 100:
        #     break
        json_file = genrate_mask_batch_prompts(SAM, paths, labels, imgs.unsqueeze(0), json_file, image_id=img_ids[0][0])
        # count +=1
    print("LOOP DONE NOW DUMPING JSON FILE...")


    # Save the updated JSON data
    json_file_write_path = os.path.join(exp.data_dir, exp.train_ann.split(".json")[0] + "_masks_from_SAM.json")
    with open(json_file_write_path, 'w') as j:
        json.dump(json_file, j)

    # val_loader = vid.get_vid_loader(batch_size=lframe + gframe, data_num_workers=4, dataset=dataset)

    # trainer = Trainer(exp, args,val_loader,val=True)


if __name__ == "__main__":
    args = make_parser().parse_args()

    exp = get_exp(args.exp_file, args.name)
    exp.test_size = (args.tsize, args.tsize)
    exp.merge(args.opts)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()
    args.machine_rank = 1
    dist_url = "auto" #if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args)
    )
