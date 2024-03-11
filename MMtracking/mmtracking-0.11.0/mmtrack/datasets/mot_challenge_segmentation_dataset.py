from mmdet.datasets import DATASETS, CocoDataset
import numpy as np

@DATASETS.register_module()
class MOTSegmentDataset(CocoDataset):
    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        
        
        for i, ann in enumerate(ann_info):
            # print(ann)
            # if i == 2:
            #     exit()
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if 'segmentation' not in ann:
                    print(ann)
                # masks = ann['segmentation']
                segmentation_list = [ ] 
                if 'segmentation' in ann:
                    masks = ann['segmentation']
                    for point in masks:
                        # print("point len:",len(point))
                        for coord in point:
                            # print("corrd value:",coord)
                            if type(coord) == list:
                                for point2 in coord:
                                    segmentation_list.append(point2)
                            else:
                                segmentation_list.append(coord)
                else:
                    print("Annotation key with no segmentation detected: ", ann)
                    segmentation_list= [x1, y1, x1+w, y1, x1 + w, y1 + h, x1, y1+h, x1, y1]
                # masks = [coord for point in masks for coord in point]
                # ann['segmentation'] = masks
                ann['segmentation'] = [segmentation_list]
                # print('ann[segmentation]: ',ann['segmentation'])
                # print('Masks: ',len(masks))
                # print('Masks[0]: ',len(masks[0]))
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')


        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            # masks=all_mask,
            seg_map=seg_map)
        # print('Ann mask',len(ann['masks']))
        # print('Ann mask[0]',len(ann['masks'][0]))
        # print('Ann mask[0][0]',len(ann['masks'][0][0]))
        # print('bbox ',len(ann['bboxes']))
        # exit()
        return ann
