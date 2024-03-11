# Multi-Object Tracking on MOT20 


## Installation

Run the following command:
```
pip install -e ./mmtracking-0.11.0
```

Unzip the mmdetection folder, then run the following command:

```
pip install -e ./mmdetection-2.19.1
```

## Dataset 
For the dataset preparation of MOT20, please refer to [dataset.md](mmtracking-0.11.0/docs/en/dataset.md)


## Results with Tracktor As Baseline


|    Method     | Detector | ReID | Train Set | Test Set | Public | MOTA | IDF1 | FP | FN | IDSw. |                                                                           Config                                                                           | Checkpoint |
|:-------------:| :------: | :--: | :-------: | :------: | :----: | :--: | :--: |:--:|:--:| :---: |:----------------------------------------------------------------------------------------------------------------------------------------------------------:| :--------: |
|   Tracktor    | R50-FasterRCNN-FPN | R50 | half-train | half-val | N | 70.5 | 65.3 | 3659 | 176118 | 1442 |                           [config](mmtracking-0.11.0/configs/mot/tracktor/tracktor_faster-rcnn_r50_fpn_8e_mot20-public-half.py)                            | [Link from MMtracking](https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot15-half_20210804_001040-ae733d0c.pth)  |
| **Tracktor+Ours** | R50-FasterRCNN-FPN+Ours | R50 | half-train | half-val | N | 71.4 | 66.7 | 3419 | 171174 | 1344 | [config](mmtracking-0.11.0/configs/mot/tracktor/tracktor_ours_r50_fpn_8e_mot20-private-half.py)| [Anonymous drive link](https://drive.google.com/file/d/1X-6L0KcWUe0smq6cML8_m8XbrZe-LhXY/view?usp=drive_link) |



## Results with ByteTrack As Baseline
|     Method     | Detector | Train Set | Test Set | Public | HOTA | MOTA | IDF1 | FP | FN | IDSw. | Config | Checkpoint |
|:--------------:| :------: | :-------: | :------: | :----: | :--: | :--: | :--: |:--:|:--:| :---: | :----: | :--------: |
|   ByteTrack    | YOLOX-X | half-train | half-val | N | 65.5 | 86.4 | 82.7 | 19176 | 63370 | 995 | [config](mmtracking-0.11.0%2Fconfigs%2Fmot%2Fbytetrack%2Fbytetrack_yolox_x_mot20-private-half_orig.py) | [Anonymous drive link](https://drive.google.com/file/d/12X_gqf7LcXUE8hw22k01xWb8LRjyEqP7/view?usp=drive_link)  |
| **ByteTrack+Ours** | YOLOX-X+Ours | half-train | half-val | N | 68.9 | 88.1 | 83.7 | 18647 | 53825 | 911 | [config](mmtracking-0.11.0%2Fconfigs%2Fmot%2Fbytetrack%2Fbytetrack_yolox_x_mot20-private-half_ours.py) | [Anonymous drive link](https://drive.google.com/file/d/1y7ZhGEHxCrhtF12yZDY-M2KXOQ2eOaIe/view?usp=drive_link) |

## Training

To train run:
```
bash ./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```

## Evaluation

To evaluate run:
```
python tools/test.py ${CONFIG_FILE} ${Checkpoint} --eval bbox track
```



