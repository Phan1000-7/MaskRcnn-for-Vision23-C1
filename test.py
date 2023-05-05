from detectron2.engine import DefaultPredictor
import os
import pickle
from utils import *
import argparse

# 创建解析器对象
parser = argparse.ArgumentParser()

# 添加一个名为 i 的参数，并指定它的类型为整数
parser.add_argument('-i', '--index', type=int, default=0, help='The index of the object in the list')
#parser.add_argument('-m','--model', type=str,default=os.path.join(cfg.OUTPUT_DIR, "model_final.pth"), help='The path of the model we use')
# 解析命令行参数
args = parser.parse_args()
#list0 各类别  list1 类别中子缺陷
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
list1 = [["break","thunderbolt"],
         ["0"],
         ["Inclusoes","Rechupe"],
         ["Collision","Dirty","Gap","Scratch"],
         ["Chip","PistonMiss","Porosity","RCS"],
         ["damage"],
         ["s_burr","s_scratch"],
         ["Defect-A","Defect-B","Defect-C","Defect-D"],
         ["Fiber","Flash Particle","Hole","Surface Damage","Tear"],
         ["missing_hole","mouse_bite","open_circuit","short","spur","spurious_copper"],
         ["defect1","defect2","defect3","defect4","defect5","defect6","defect7"],
         ["t_contamination","t_scratch","unfinished_surface"],
         ["defect"],
         ["impurities","pits"]]

#指定cfg路径
cfg_save_path = "./cfgs/OD_cfg"+list0[args.index]+".pickle"

with open(cfg_save_path, 'rb') as f:
    cfg = pickle.load(f)

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

#根目录下测试图片，测试集的json也在根目录下
#test_root_path = "1.jpg"
test_root_path = "../dataset_test/" + list0[args.index]
on_Image(test_root_path, predictor)

