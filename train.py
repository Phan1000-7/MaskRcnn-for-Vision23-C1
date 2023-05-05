import argparse
from detectron2.utils.logger import setup_logger

setup_logger()
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer

import os
import pickle

from utils import *

# config_file_path = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"


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

list1 = [["break", "thunderbolt"],
         ["0"],
         ["Inclusoes", "Rechupe"],
         ["Collision", "Dirty", "Gap", "Scratch"],
         ["Chip", "PistonMiss", "Porosity", "RCS"],
         ["damage"],
         ["s_burr", "s_scratch"],
         ["Defect-A", "Defect-B", "Defect-C", "Defect-D"],
         ["Fiber", "Flash Particle", "Hole", "Surface Damage", "Tear"],
         ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"],
         ["defect1", "defect2", "defect3", "defect4", "defect5", "defect6", "defect7"],
         ["t_contamination", "t_scratch", "unfinished_surface"],
         ["defect"],
         ["impurities", "pits"]]

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--index', type=int, help='The index of the object in the list')
args = parser.parse_args()

'''
instance segmentation
'''
# config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
# checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"

# config_file_path = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

config_file_path = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
checkpoint_url = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"

# 用其他类别的预训练模型训练
#checkpoint_url ="../pretrained_model/output1"+list0[(args.index+1)%14]+"/model_final.pth"

output_dir = "./output/" + list0[args.index]

num_classes = len(list1[args.index])
class_names = list1[args.index]

device = "cuda"

train_dataset_name = "LP_train"
train_images_path = "../dataset/" + list0[args.index] + "/train"
train_json_annot_path = "../dataset/" + list0[args.index] + "/train/_annotations.coco.json"

val_dataset_name = "LP_val"
val_images_path = "../dataset/" + list0[args.index] + "/val"
val_json_annot_path = "../dataset/" + list0[args.index] + "/val/_annotations.coco.json"

cfg_save_path = "./cfgs/OD_cfg"+list0[args.index]+".pickle"

###########################################################
# 注册训练集
register_coco_instances("LP_train", {}, train_json_annot_path, train_images_path)
MetadataCatalog.get("LP_train").set(thing_classes=class_names,
                                    evaluator_type='coco',
                                    json_file=train_json_annot_path,
                                    image_root=train_images_path)

# 注册测试集
register_coco_instances("LP_val", {}, val_json_annot_path, val_images_path)
MetadataCatalog.get("LP_val").set(thing_classes=class_names,
                                  evaluator_type='coco',
                                  json_file=val_json_annot_path,
                                  image_root=val_images_path)


# plot_samples(dataset_name=train_dataset_name, n=3)

#####################################################
def main():
    cfg = get_train_cfg(config_file_path, checkpoint_url, train_dataset_name, val_dataset_name, num_classes, device,
                        output_dir,args.index)

    with open(cfg_save_path, 'wb') as f:
        pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    trainer.train()


if __name__ == '__main__':
    main()
