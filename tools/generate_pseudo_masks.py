#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
    This script generates mask by using the required SAM model
- checkpoint default is SAM_B however, you can modfiy with argparse
- All functions are written in this file Basides the data loader stuff which comes from vid

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
from yolox.data.datasets import vid, coco

#SAM Related Imports
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from skimage import measure
from xml.dom.minidom import Document
from xml.dom import minidom
import numpy as np
import sys
import cv2
import re

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
        "--visualize", default=False, action="store_true", help="Write the images with mask and confidence score on the corresponding Data path to Annotations"
    )
    parser.add_argument("-c", "--ckpt", default='../../checkpoints/sam_vit_h_4b8939.pth', type=str, help="checkpoint file")
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


def write_annotation(path, masks, confidence_scores):
    '''

    Args:
        path: Image file path
        masks: List of masks as polygon points len(list)=No. of BBoxes

    Returns:
        Nothing
    Process:
        Fetch the XML copy all corresponding data
        Add mask polygon points besides same object
        call the write_file function to write an XML file with mask
    '''


    xml = path.replace("Data", "Annotations").replace("JPEG", "xml")

    # print("reading from xml file : ",xml.replace("train/", "train_orignal/"))
    #Temporary added since train xml.replace("train/", "train_orignal/") has mask infromation from SAM VIT_B
    #Doing this only for generating SAM VIT_H
    file = minidom.parse(xml.replace("train/", "train_orignal/"))
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
        #Adding try catch since it is possible
        #That SAM did not predict any mask
        try:
            mask = masks[i]
            mask_confidence = confidence_scores[i]
        except IndexError:
            print(f"Mask cannot be appended for this object in XML file'.")
            continue

        # Create a new 'mask' element and add it to the object node.
        mask_element = new_doc.createElement("mask")

        # Set the 'confidence' attribute of the 'mask' element.
        mask_element.setAttribute("confidence", str(mask_confidence))

        # Convert all points into a single string, separating individual points by semicolons.
        #To reduce memory for each XML file
        points_str = ";".join(
            f"{point[0]},{point[1]}" if len(point) == 2 else f"{point[0][0]},{point[0][1]}" for point in mask)

        # Create a text node with the points string and append it to the 'mask' element.
        points_text = new_doc.createTextNode(points_str)
        mask_element.appendChild(points_text)

        new_obj.appendChild(mask_element)

    # Write the new XML content to the new file.
    xml_str = new_doc.toprettyxml(indent="  ")

    # Replace lines containing only whitespaces with empty lines.
    xml_str = re.sub(r'\n\s*\n', '\n', xml_str)

    # Remove the first line, which is the xml version declaration.
    xml_str = '\n'.join(xml_str.split('\n')[1:])

    # Write the new XML content to the new file by using same function.
    write_file(xml_str, xml, "xml")

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
        file_write_dir = os.path.join(prefix, "VIT_H_train_with_masks", suffix.split(os.path.basename(file_path))[0])
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
        # print(f"Writing file path : {os.path.join(file_write_dir,os.path.basename(file_path))}")
        with open(os.path.join(file_write_dir,os.path.basename(file_path)), "w") as f:
            f.write(file)
    else:
        sys.exit("File type not supported.")

def prepare_image(image, transform):
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device='cuda')
    image = image.permute(2, 0, 1).contiguous()
    return image

def genrate_mask_batch_prompts(model, info_imgs, labels, imgs, visualize=False):
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

    Returns:
        Nothing

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
        #Not using file_exist() check ATM. Since, we want to generate mask with SAM_VIT_H
        # if bbox_preds.shape[0] >= 1 and not file_exist(info_imgs[ind], file_type_xml=False):
        if bbox_preds.shape[0] >= 1:
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

        #Now iterating over images to get polygon points for each points
        for ind, output in enumerate(batched_output):
            polygons,confidence_scores = masks_to_polygons(output, image_list[ind], label_list[ind], visualize=visualize)
            write_annotation(image_list[ind], polygons, confidence_scores)


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

    print(f"PRINTING NUM WORKERS {args.num_workers} AND BATCH SIZE {args.batch_size}  AND GFRAME {gframe}")

    if args.val_split:
        dataset = vid.SAMVIDDataset(file_path='./yolox/data/datasets/val_seq.npy',
                                     img_size=(args.tsize, args.tsize), preproc=Vid_Val_Transform(), lframe=lframe,
                                     gframe=gframe, val=True, mode=args.mode, dataset_pth=exp.data_dir,
                                     tnum=int(args.tnum), generate_mask=False)
    else:
        dataset = vid.SAMVIDDataset(img_size=exp.input_size, lframe=lframe, gframe=gframe,
                                 dataset_pth=exp.data_dir, generate_mask=False)

    dataset_loader = vid.get_trans_loader_sam(batch_size=args.batch_size, data_num_workers=args.num_workers, dataset=dataset)

    # SAM STUFF
    # Intialize SAM model and forwrard Pass
    model_registry= re.search(r'sam_(.{5})', os.path.basename(args.ckpt)).group(1)

    print(f"LOADED MODEL REGISTRY : {model_registry}")
    SAM = sam_model_registry[model_registry](checkpoint=args.ckpt).to(device="cuda")

    for i, batch in enumerate(tqdm(dataset_loader, total=len(dataset_loader))):

        #condition is needed because batch can be empty
        #We are discarding samples not comprising of bbox info.
        if batch is None:
            continue
        imgs, _, labels, paths = batch

        #ADDING ANOTHER DIMENSION IN IMAGES SINCE BATCH SIZE IS 1
        genrate_mask_batch_prompts(SAM, paths, labels, imgs.unsqueeze(0), visualize=args.visualize)

    print(f"LENGTH OF THE DATASET {len(dataset)} ")

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
