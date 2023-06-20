import detectron2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import json
import csv
import numpy as np
import torch

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

train_data = 'training_dataset'
DatasetCatalog.clear()
register_coco_instances(train_data, {}, 'annotations.json', 'train/')
metadata = MetadataCatalog.get(train_data)

# Set up the configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

cfg.DATASETS.TRAIN = (train_data,)  # Name of your training dataset
# cfg.DATASETS.TEST = ()  # Name of your validation dataset
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 1 class - wheat

faster_rcnn = DefaultTrainer(cfg)
faster_rcnn.resume_or_load(resume=False)
faster_rcnn.train()

output_dir = "models"
faster_rcnn.save_model(output_dir)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')
