import copy
import math

from loguru import logger
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from yolox.utils import compute_mask_iou
from yolox.utils.visualize import visualize_mask_pred_gt_during_loss

from .losses import IOUloss, CrossEntropyLoss
from .dice_loss import DiceLoss
from .network_blocks import BaseConv, DWConv
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from yolox.models.yolovp_msa import YOLOXHead
from yolox.models.post_process import postprocess

from torchvision.ops import roi_align


class FCNMaskHead(nn.Module):
    def __init__(self, in_channels, num_classes, width=1, hidden_layer=256, upsampling=True):
        super(FCNMaskHead, self).__init__()

        # layers of 3x3 convolutions
        self.conv1 = nn.Conv2d(in_channels, hidden_layer, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_layer, hidden_layer, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden_layer, hidden_layer, 3, padding=1)
        self.conv4 = nn.Conv2d(hidden_layer, hidden_layer, 3, padding=1)

        # Upsampling layer using bilinear interpolation
        self.upsampling = upsampling
        if self.upsampling:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Predict the masks for each class (including background)
        self.predictor = nn.Conv2d(hidden_layer, num_classes, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)

        #deconv performs inferior to simple upsampling
        # x = self.deconv(x)
        # x = self.relu(x)

        # Upsampling using bilinear interpolation
        if self.upsampling:
            x = self.upsample(x)

        return self.predictor(x)


class VOWSAM(YOLOXHead):

    def __init__(self, num_classes, width, p_level=2, mask_iou_threshold=0.5, defulat_p=30, *args, **kwargs):
        super(VOWSAM, self).__init__( num_classes, width, defualt_p=defulat_p, *args, **kwargs)
        self.mask_convs = FCNMaskHead(in_channels=int(256 * width), num_classes=num_classes, hidden_layer=int(256 * 1))
        self.mask_loss = nn.BCEWithLogitsLoss(reduction="mean")

        self.p_level = p_level  # selecting features from last level is the default setting now

        self.gt_roi_size = (32, 32)
        self.mask_iou_threshold = mask_iou_threshold

        #initailizing conv for instance maks features
        self.instance_conv = BaseConv(
            in_channels=int(256 * width),
            out_channels=int(256 * width),
            ksize=3,
            stride=1,
            act="silu",
        )

        # Set spatial_scale based on p_level
        # P_level 0 means P3, and so on
        initial_stride = 4  # The stride for P2
        stride = initial_stride * (2 ** (p_level+1))
        self.spatial_scale = 1.0 / stride  # Calculate spatial_scale

        # Add ROIAlign initialization
        self.roi_output_size = self.gt_roi_size
        self.sampling_ratio = -1  # Typically -1, which means using adaptive sampling
        self.roi_align = roi_align

    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, masks=None, num_inst_per_img=0,
                mask_confidence_score=None):
        outputs = []
        outputs_decode = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        before_nms_features = []
        before_nms_regf = []

        for k, (cls_conv, cls_conv2, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            reg_feat = reg_conv(x)
            cls_feat = cls_conv(x)
            cls_feat2 = cls_conv2(x)

            # this part should be the same as the original model
            obj_output = self.obj_preds[k](reg_feat)
            reg_output = self.reg_preds[k](reg_feat)
            cls_output = self.cls_preds[k](cls_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                # print(f"OUTPUT SHAPE BEFORE DECODE; [{output.shape} and mask shape : {mask_output.shape} and obj output shape : {obj_output.shape}")
                # output_decode variable ADDED BY THE YOLOV to use this output later on
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                # making sure output corresponds to meaningful bbox predicions in the image space.
                # adjusting stride, scaling and decoding relative pred. into absolute coordiantes.
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                # ADDED BY THE YOLOV, all these 3 lines below
                outputs.append(output)
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            else:
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )
                # ADDED BY THE YOLOV both two lines
                # which features to choose
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            outputs_decode.append(output_decode)

        '''
            Below code is written by YOLOV
            Untill the if self.training condition
        '''
        self.hw = [x.shape[-2:] for x in outputs_decode]

        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2
                                   ).permute(0, 2, 1)
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())

        pred_result, pred_idx = self.postpro_woclass(decode_res, num_classes=self.num_classes,
                                                     nms_thre=self.nms_thresh,
                                                     topK=self.Afternum)  # postprocess(decode_res,num_classes=30)

        # return pred_result
        if not self.training and imgs.shape[0] == 1:
            return self.postprocess_single_img(pred_result, self.num_classes)

        # Here instead of regression feature pass Instance Features
        # Shape [B, C, H, W)
        instance_features_list = []

        for x in before_nms_features:
            instance_features_list.append(self.instance_conv(x).flatten(start_dim=2))
        instance_features_flatten = torch.cat(instance_features_list, dim=2).permute(0, 2, 1)  #

        cls_feat_flatten = torch.cat(
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]

        features_cls, features_inst, cls_scores, fg_scores = self.find_feature_score(cls_feat_flatten, pred_idx,
                                                                                     instance_features_flatten, imgs,
                                                                                     pred_result)
        features_inst = features_inst.unsqueeze(0)  # [1, BS, 30, 320]
        features_cls = features_cls.unsqueeze(0)  # [1, BS, 30, 320]
        # print(f"After Find_Feature_score function, features_reg shape :{features_reg.shape} and features_cls shape :{features_cls.shape}")

        if not self.training:
            cls_scores = cls_scores.to(cls_feat_flatten.dtype)
            fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        if self.use_score:
            trans_cls = self.trans(features_cls, features_inst, cls_scores, fg_scores, sim_thresh=self.sim_thresh,
                                   ave=self.ave, use_mask=self.use_mask)
        else:
            trans_cls = self.trans(features_cls, features_inst, None, None, sim_thresh=self.sim_thresh, ave=self.ave)
        fc_output = self.linear_pred(trans_cls)
        fc_output = torch.reshape(fc_output, [outputs_decode.shape[0], -1, self.num_classes + 1])[:, :, :-1]

        # trans_cls shape [BS, 30, 1280]. before_nms_features[self.p_level].shape [BS, 320, 11-20, 11-20]
        # print(f"After MSA of YOLOV, trans_cls shape :{trans_cls.shape} and input feature for ROI shape : {before_nms_features[self.p_level].shape}")
        if self.training:

            '''
                Mask handling code starts here
            '''
            # Convert pred_result list to tensor
            bboxes_tensor = torch.stack(pred_result, dim=0)  # Shape: [batch_size, 30, 37]

            # Later needed these value for reshaping tensors
            batch_size, channels, H, W = before_nms_features[self.p_level].shape
            num_bboxes = bboxes_tensor.size(1)

            bboxes = bboxes_tensor[..., :4].view(-1, 4)
            bboxes = torch.clamp(bboxes, min=0)  # addressing negative values

            roi_features = self.roi_align(
                before_nms_features[self.p_level],
                # self.p_level defines which scale level feature to choose from, {0,1,2 --> [P3,P4,P5]}
                [bboxes],
                output_size=self.roi_output_size,
                spatial_scale=self.spatial_scale,
                sampling_ratio=self.sampling_ratio
            )  # shape [B, N, C, *output_size)

            roi_features = roi_features.view(batch_size * num_bboxes, channels, *self.roi_output_size)

            pred_masks = torch.sigmoid(
                self.mask_convs(roi_features))  # shape [B * N, num_classes, *self.roi_output_size * 2)
            # print("roi feature shape ", roi_features.shape, " mask preditciton shape, ", pred_masks.shape)

            # Reshape the mask outputs
            mask_height, mask_width = pred_masks.shape[-2:]
            pred_masks = pred_masks.view(batch_size, num_bboxes, self.num_classes, mask_height, mask_width)
            pred_masks = (pred_masks > 0.5).float()  # Thresholding and making values either 0 or 1 for loss

            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                refined_cls=fc_output,
                idx=pred_idx,
                pred_res=pred_result,
                mask_preds=pred_masks,  # Pass the segmentation predictions to the loss function
                masks=masks,  # Pass the ground truth masks to the loss function
                mask_confidence_score=mask_confidence_score,  # confidence scores for mask
            )
        else:

            class_conf, class_pred = torch.max(fc_output, -1, keepdim=False)  #
            result, result_ori = postprocess(copy.deepcopy(pred_result), self.num_classes, fc_output,
                                             nms_thre=nms_thresh)

            return result, result_ori  # result


    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            refined_cls,
            idx,
            pred_res,
            mask_preds,  # The predicted masks for instance segmentation
            masks,  # The ground truth masks for instance segmentation
            mask_confidence_score=None, #confidence scores for mask
    ):

        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []
        # batch_mask_preds=[]
        # batch_mask_targets = []
        individual_mask_losses = []

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                ref_target[:, -1] = 1
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]
                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target_onehot = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                )
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)
                fg_idx = torch.where(fg_mask)[0]

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                fg = 0

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)

                for ele_idx, ele in enumerate(idx[batch_idx]):
                    loc = torch.where(fg_idx == ele)[0]
                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:
                        max_idx = int(max_iou.indices[ele_idx])
                        ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                        fg += 1
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]

                ''' Mask handling and mask loss computation starts here '''

                mask_img_pred = mask_preds[batch_idx]
                mask_img_targets = masks[batch_idx]

                # Find the indices of non-zero masks.
                # Each image has N masks where N is for the whole batch. Hence, finiding masks for this image
                non_zero_mask_indices = (mask_img_targets.sum((1, 2)) != 0).nonzero(as_tuple=True)[0]
                # print("Actual GT ON: ", non_zero_mask_indices)
                # Select non-zero masks
                mask_img_targets = mask_img_targets[non_zero_mask_indices]
                # print("mask GT shape", mask_img_targets.shape)

                #Filter Predicted mask  based on maximum class_predictions
                mask_img_pred, empty_pred_flag = self.filter_mask_preds_ref_targets(mask_img_pred, ref_target)

                # initalizing image mask loss list
                image_mask_loss = []
                gt_bbox_for_mask = box_cxcywh_to_xyxy(gt_bboxes_per_image)

                # Initialize a tensor to keep track of which ground truth masks have been matched
                matched_gt_masks = torch.zeros(mask_img_targets.size(0), dtype=torch.bool)

                #Handling condition when there are predictions after the filteration from Classification
                if not empty_pred_flag:
                    #Finding best matches to create individual mask targets for mask loss
                    for pred_idx in range(mask_img_pred.size(0)):
                        best_iou = 0
                        best_target_idx = -1
                        pred_mask = mask_img_pred[pred_idx]
                        #Converting bbox to xyxy format for to resize mask

                        # print("Target size", mask_img_targets.size(0), " actual GT size", len(gt_bbox_for_mask))
                        # Calculate IoU with each ground truth mask
                        for target_idx in range(mask_img_targets.size(0)):
                            #for each target_idx, find gt_bbox
                            # Convert box coordinates to integer, as they are indices
                            x1, y1, x2, y2 = map(int, gt_bbox_for_mask[target_idx])
                            w, h = x2 - x1, y2- y1
                            # interpolate mask prediction to orignal bounding box size instead of resizing mask target
                            pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0),
                                                              # Add batch and channel dims
                                                              size=(h, w),
                                                              mode='nearest').squeeze(0).squeeze(0)  # Remove added dims

                            #Cropping mask target to fetch the same mask area of bbox
                            gt_mask = mask_img_targets[target_idx][y1:y2, x1:x2]

                            iou = compute_mask_iou(pred_mask, gt_mask, input_threshold=self.mask_iou_threshold) #threshold to convert pred_mask to 1 or 0
                            # print(iou)  # could be commented later
                            # If this IoU is better than the previous best, save it
                            if iou > best_iou:
                                best_iou = iou
                                best_target_idx = target_idx
                                best_gt_mask = gt_mask
                                best_pred_mask = pred_mask

                        # If a matching target is found, update the matched_mask_targets tensor
                        if best_target_idx != -1:
                            matched_gt_masks[best_target_idx] = True
                            instance_mask_loss = self.mask_loss(best_pred_mask, best_gt_mask)
                            image_mask_loss.append(instance_mask_loss)

                # Now penalize false negatives #
                # these can be FN, where Pred_Mask has 0 Overleap with GT Mask
                # OR FN, where no predictions were left after filteration but GT mask exist
                for target_idx, gt_mask in enumerate(mask_img_targets):
                    if not matched_gt_masks[target_idx]:
                        # for each target_idx, find gt_bbox
                        # Convert box coordinates to integer, as they are indices
                        x1, y1, x2, y2 = map(int, gt_bbox_for_mask[target_idx])
                        best_gt_mask = mask_img_targets[target_idx][y1:y2, x1:x2]
                        best_pred_mask = torch.zeros_like(best_gt_mask)
                        image_mask_loss.append(self.mask_loss(best_pred_mask, best_gt_mask))

                '''
                    ADDED MASK AND MASK GT MANIPULATION ENDS HERE
                '''
                # print("matched_mask_preds shape ",matched_mask_preds.shape, "matched_mask_targets : ",matched_mask_targets.shape)
                # Computing image mask loss
                if image_mask_loss:
                    image_mask_loss_item = torch.stack(image_mask_loss).mean()
                    # print("each image loss", image_mask_loss)
                    individual_mask_losses.append(image_mask_loss_item)

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ref_targets = torch.cat(ref_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)
        ref_masks = torch.cat(ref_masks, 0)
        # print(sum(ref_masks)/ref_masks.shape[0])
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        loss_ref = (
                       self.bcewithlog_loss(
                           refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]
                       )
                   ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 3.0


        loss_mask = 0.0

        if individual_mask_losses:
            individual_mask_losses_item = torch.stack(individual_mask_losses)
            loss_mask = individual_mask_losses_item.mean()

        # Add the mask loss to the total loss
        # We may need to adjust the weight of the loss
        loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls + loss_mask

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            2 * loss_ref,
            loss_l1,
            loss_mask,  # Return the instance segmentation loss as a separate value
            num_fg / max(num_gts, 1),
        )
    def filter_mask_preds_ref_targets(self, mask_img_pred, ref_target):
        '''
        -Leveraging refined classification predictions targets to filter mask predictions
        -Reduced training time by reducing mask loss computations

        Args:
            mask_img_pred: [N,C,H,W]
            ref_target:[N,C] N RoIs X C Classes

        Returns:
            final_filtered_mask_pred: fitered predictions
            Flag: Boolean to indicate whether fitered predictions are empty or not
        '''
        # Find the indices where the last column of ref_target is 0 (i.e., not background)
        non_background_indices = (ref_target[:, -1] == 0).nonzero(as_tuple=True)[0]
        # print("non_background_indices", non_background_indices)

        # Find the class indices for the non-background proposals
        class_indices = torch.argmax(ref_target[non_background_indices, :-1], dim=1)

        # Handling condition when after filteration, no predictions are left
        #In this case, we skip this iteration
        if len(class_indices) == 0:
            return mask_img_pred, True
        # Use advanced indexing to select the corresponding predicted masks
        final_filtered_mask_pred = mask_img_pred[non_background_indices, class_indices]
        # The shape of final_filtered_mask_pred will be [num_non_background, H, W]

        return final_filtered_mask_pred, False

