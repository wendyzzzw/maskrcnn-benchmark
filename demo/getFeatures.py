import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import torch

# this makes our figures bigger
pylab.rcParams['figure.figsize'] = 20, 12

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.7,
)

import os
asset_dir = '/home/wzhou14fall/scans/out/'
output_dir = os.path.join(asset_dir, 'features/')

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
scenes = os.listdir(asset_dir)


for scene in scenes:
    print(scene)
    if scene.startswith("scene"):
        scene_dir = os.path.join(asset_dir, scene)
        scene_output_dir = os.path.join(output_dir, scene)
        if not os.path.isdir(scene_output_dir):
            os.mkdir(scene_output_dir)
        count = 0
#         print(scene_dir)
        for file in os.listdir(scene_dir):
            if file.endswith(".jpg"):
                count += 1
                if count % 100 == 0:
                    print(count)
                img_fn_dir = os.path.join(scene_dir, file)
                pil_image = Image.open(img_fn_dir).convert("RGB")
                image = np.array(pil_image)[:, :, [2, 1, 0]]
                #imshow(image)

                predictions, features = coco_demo.run_on_opencv_image(image)
                #imshow(predictions)
                #for feature in features:
                #    print(feature.size())

                torch.save(features, str(scene_output_dir)+'/'+str(file)+'.pt')
                
                #break





