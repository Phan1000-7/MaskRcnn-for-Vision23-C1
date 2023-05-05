import os
from detectron2.structures import BoxMode, PolygonMasks, BitMasks
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from pycocotools import mask as mask_utils
from detectron2.utils.visualizer import ColorMode

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import json
import copy

# casting、ring、console、groove、lens  2、3、6、8、11
list0 = ["cable",
         "capacitor",
         "casting",
         "console",
         "cylinder",
         "electronics",
         "groove",
         "hemisphere",
         "lens",
         "pcb1",
         "pcb2",
         "ring",
         "screw",
         "wood"]

json_name = ["Cable.segm.json",
             "Capacitor.segm.json",
             "Casting.segm.json",
             "Console.segm.json",
             "Cylinder.segm.json",
             "Electronics.segm.json",
             "Groove.segm.json",
             "Hemisphere.segm.json",
             "Lens.segm.json",
             "PCB_1.segm.json",
             "PCB_2.segm.json",
             "Ring.segm.json",
             "Screw.segm.json",
             "Wood.segm.json"
             ]

# anchor_size = [
#        [66676,2550,12090,10296,6300,17388],
#        [],
#        [],
#
# ]


anchor_ratio = [
    [[4.5,2.9,0.6,1.1,1.5], [4.5,2.9,0.6,1.1,1.5], [4.5,2.9,0.6,1.1,1.5], [4.5,2.9,0.6,1.1,1.5], [4.5,2.9,0.6,1.1,1.5]],
    [[3.3,0.9,1.6,0.5,2.5], [3.3,0.9,1.6,0.5,2.5], [3.3,0.9,1.6,0.5,2.5], [3.3,0.9,1.6,0.5,2.5], [3.3,0.9,1.6,0.5,2.5]],
    [[0.3,0.5,1.0,1.7,0.7], [0.3,0.5,1.0,1.7,0.7], [0.3,0.5,1.0,1.7,0.7], [0.3,0.5,1.0,1.7,0.7], [0.3,0.5,1.0,1.7,0.7]],
    [[1.9,2.1,21.2,37.2,54.5], [1.9,2.1,21.2,37.2,54.5], [1.9,2.1,21.2,37.2,54.5], [1.9,2.1,21.2,37.2,54.5], [1.9,2.1,21.2,37.2,54.5]],
    [[0.8,1.2,1.5,3.5,5.5], [0.8,1.2,1.5,3.5,5.5], [0.8,1.2,1.5,3.5,5.5], [0.8,1.2,1.5,3.5,5.5], [0.8,1.2,1.5,3.5,5.5]],
    [[0.5,1.1,2.0,3.7,5.0], [0.5,1.1,2.0,3.7,5.0], [0.5,1.1,2.0,3.7,5.0], [0.5,1.1,2.0,3.7,5.0],[0.5,1.1,2.0,3.7,5.0]],
    [[0.6,1.3,3.4,0.3,6.4], [0.6,1.3,3.4,0.3,6.4], [0.6,1.3,3.4,0.3,6.4], [0.6,1.3,3.4,0.3,6.4], [0.6,1.3,3.4,0.3,6.4]],
    [[0.8,1.0,1.8,3.2,0.5], [0.8,1.0,1.8,3.2,0.5], [0.8,1.0,1.8,3.2,0.5], [0.8,1.0,1.8,3.2,0.5], [0.8,1.0,1.8,3.2,0.5]],
    [[3.8,1.2,2.7,1.9,0.6], [3.8,1.2,2.7,1.9,0.6], [3.8,1.2,2.7,1.9,0.6], [3.8,1.2,2.7,1.9,0.6], [3.8,1.2,2.7,1.9,0.6]],
    [[0.6,4.6,7.0,1.9,10.8], [0.6,4.6,7.0,1.9,10.8], [0.6,4.6,7.0,1.9,10.8], [0.6,4.6,7.0,1.9,10.8], [0.6,4.6,7.0,1.9,10.8]],
    [[0.5,0.8,1.2,1.8,2.1], [0.5,0.8,1.2,1.8,2.1], [0.5,0.8,1.2,1.8,2.1], [0.5,0.8,1.2,1.8,2.1], [0.5,0.8,1.2,1.8,2.1]],
    [[0.4,1.7,0.8,2.8,6.7], [0.4,1.7,0.8,2.8,6.7], [0.4,1.7,0.8,2.8,6.7], [0.4,1.7,0.8,2.8,6.7], [0.4,1.7,0.8,2.8,6.7]],
    [[0.5,3.3,1.0,2.1,4.6], [0.5,3.3,1.0,2.1,4.6], [0.5,3.3,1.0,2.1,4.6], [0.5,3.3,1.0,2.1,4.6], [0.5,3.3,1.0,2.1,4.6]],
    [[0.7,2.7,6.0,8.7,13.3], [0.7,2.7,6.0,8.7,13.3], [0.7,2.7,6.0,8.7,13.3],[0.7,2.7,6.0,8.7,13.3],[0.7,2.7,6.0,8.7,13.3]]
]


def plot_samples(dataset_name, n=1):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)

    for s in random.sample(dataset_custom, n):
        img = cv2.imread(s["file_name"])
        v = Visualizer(img[:, :, ::-1], metadata=dataset_custom_metadata, scale=0.5)
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15, 20))
        plt.imshow(v.get_image())
        plt.show()


def get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, test_dataset_name, num_classes, device,
                  output_dir, ind):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)

    # 开启增广
    cfg.TEST.AUG.ENABLED = True
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = anchor_size[i]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = anchor_ratio[ind]

    # 采用自己的预训练模型
    # cfg.MODEL.WEIGHTS = checkpoint_url
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (test_dataset_name,)

    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 2000
    #cfg.SOLVER.STEPS = (50000, 75000)
    cfg.SOLVER.STEPS = []

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    return cfg


def on_Image(test_root_path, predictor):
    json_root = "../Predictions/"

    out_path = "../FinalPredictions/"
    class_name = test_root_path.split('/')[2]
    idx = 0
    for i in range(14):
        if class_name == list0[i]:
            idx = i
            break

    out_path = out_path + json_name[idx]
    json_root = json_root + json_name[idx]

    print(json_root)

    # 读取原始的，现在需要在其annotation中填入我们每张图片的预测结果
    with open(json_root, 'r') as f:
        data = json.load(f)

    imgs = data["images"]
    annotations = []
    segid = 1
    # 标注格式
    # ano = {"iscrowd": 0, "image_id": 0, "bbox": [], "segmentation": [], "category_id": 1, "id": 0, "area": 0}
    for src in os.listdir(test_root_path):
        if src.split('.')[1] == 'jpg':
            imgid = 0
            for each in imgs:
                if src == each["file_name"]:
                    imgid = each["id"]
                    break
            # 需要给的是绝对路径
            im = cv2.imread(os.path.join(test_root_path, src))
            outputs = predictor(im)
            # 关于outputs的处理以及放在json_root的对应位置

            # 获取预测的各部分结果
            instances = outputs["instances"].to("cpu")
            bbx_l = instances.pred_boxes.tensor.shape[0]
            pred_boxes = instances.pred_boxes.tensor.numpy()

            pred_scores = instances.scores.numpy()
            pred_classes = instances.pred_classes.numpy()
            pred_masks = instances.pred_masks.numpy()

            # 将预测结果转换为指定格式的标注 有几个bbx就需要几个anno
            for i in range(bbx_l):
                bbx = BoxMode.convert(pred_boxes[i:], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                bbx = [int(x) for x in bbx[0]]
                category_id = int(pred_classes[i])
                score = float(pred_scores[i])
                # 把第一个掩码取出来
                mask = pred_masks[i, :, :].tolist()
                # 将二维列表转换为NumPy数组
                mask_np = np.array(mask, dtype=np.uint8)

                rle = mask_utils.encode(np.array(mask_np[:, :, None], order='F', dtype="uint8"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")
                area = int(np.sum(mask_np))

                # polygon格式
                # 查找轮廓
                # contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #  定义一个空列表来保存多边形
                # polygons = []
                # for contour in contours:
                #     # 近似多边形
                #     epsilon = 0.02 * cv2.arcLength(contour, True)
                #     approx = cv2.approxPolyDP(contour, epsilon, True)
                #     # 将多边形顶点展平并添加到多边形列表中
                #     polygons.append(approx.flatten().tolist())

                annotation = {
                    # "iscrowd": 0,
                    "image_id": imgid,  # 设置为0，因为不知道具体的图像ID
                    # "bbox": bbx,
                    "category_id": category_id,
                    "segmentation": rle,
                    # "id": segid,
                    "score": score
                    # "area": area
                }
                annotations.append(annotation)
                segid += 1

    data["annotations"] = annotations
    new_json = json.dumps(annotations)
    with open(out_path, 'w') as ff:
        ff.write(new_json)

    # instance_mode:
    IMAGE = 0
    """
    Picks a random color for every instance and overlay segmentations with low opacity.
    """
    SEGMENTATION = 1
    """
    Let instances of the same category have similar colors
    (from metadata.thing_colors), and overlay them with
    high opacity. This provides more attention on the quality of segmentation.
    """
    IMAGE_BW = 2
    """
    Same as IMAGE, but convert all areas without masks to gray-scale.
    Only available for drawing per-instance mask predictions.
    """


def on_Video(videoPath, predictor):
    class_names = ["break", "thunderbolt"]
    cap = cv2.VideoCapture(videoPath)
    if (cap.isOpened() == False):
        print("Error opening file...")
        return

    (success, image) = cap.read()
    while success:
        predictions = predictor(image)
        v = Visualizer(image[:, :, ::-1], metadata={'thing_classes': class_names}, scale=0.5,
                       instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

        # cv2.imread("Reuslt", output.get_image()[:,:,::-1])
        # cv2.namedWindow("result", 0)
        # cv2.resizeWindow("result", 1200, 600)

        # 调用电脑摄像头进行检测
        cv2.namedWindow("result", cv2.WINDOW_FREERATIO)  # 设置输出框的大小，参数WINDOW_FREERATIO表示自适应大小
        cv2.imshow("result", output.get_image()[:, :, ::-1])

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        (success, image) = cap.read()
