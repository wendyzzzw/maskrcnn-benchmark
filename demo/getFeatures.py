import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

import requests
from io import BytesIO
from PIL import Image
import numpy as np
import os
import torch
import json
import shutil

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
    confidence_threshold=0.5,
)

# extract features and masks
asset_dir = '/home/wzhou14fall/scans/out/'   # directory that contains extracted images
json_dir = '/home/wzhou14fall/selected_frame.json'
output_dir = '/home/wzhou14fall/output3' 

scenes = os.listdir(asset_dir)
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

info_all = {}

with open(json_dir, 'r') as json_file:
    data = json.load(json_file)
    for scene in data:
        
        if scene['scene_id'] in scenes:
            models = []
            scene_id = scene['scene_id']
            print(scene_id)
            for model in scene['model']:
                #print(model['id_cad'])
                models.append(model)
                #print(model['selected_frames'])
                
            info = {}
            scene_dir = os.path.join(asset_dir, scene_id)
            scene_output_dir = os.path.join(output_dir, scene_id)
            if not os.path.isdir(scene_output_dir):
                os.mkdir(scene_output_dir)
                
            features_dir = os.path.join(scene_output_dir, 'features')
            if not os.path.isdir(features_dir):
                os.mkdir(features_dir)
            masks_dir = os.path.join(scene_output_dir, 'masks')
            if not os.path.isdir(masks_dir):
                os.mkdir(masks_dir)
                
            for model in models:
                info[model['id_cad']] = {}
                for frame in model['selected_frames']:
                    frame_dict = {}
                    img_fn_dir = os.path.join(scene_dir, frame)
                    pil_image = Image.open(img_fn_dir).convert("RGB")
                    image = np.array(pil_image)[:, :, [2, 1, 0]]
                    #imshow(image)
                    result, features, top_predictions = coco_demo.run_on_opencv_image(image)
                    #imshow(result)
                    masks = top_predictions.get_field("mask").numpy()
                    masks = np.squeeze(masks, axis=1)
                    scores = top_predictions.get_field("scores").tolist()
                    labels = top_predictions.get_field("labels").tolist()
                    labels = [coco_demo.CATEGORIES[i] for i in labels]
                    
                    frame_id = frame.split('.')[0]
                    print(frame_id)
                    
                   
                    
                    
                    features_fn = 'features/'+ str(frame_id)+'.features'
                    masks_fn = 'masks/' + str(frame_id)+'.masks'

                    #np.savetxt(str(scene_output_dir)+'/'+ masks_fn, masks, delimiter=',') 
                    #masks.tofile(str(scene_output_dir)+'/'+ masks_fn, sep=",",format="%s")
                    torch.save(features[1:], str(scene_output_dir)+'/'+ features_fn)

                    with open(str(scene_output_dir)+'/'+ masks_fn, 'w') as outfile:
                        outfile.write('# Array shape: {0}\n'.format(masks.shape))
                        for mask_slice in masks:
                            np.savetxt(outfile, mask_slice)
                            outfile.write('# New mask\n')

                    frame_dict['features'] = features_fn
                    frame_dict['masks'] = masks_fn
                    frame_dict['scores'] = scores
                    frame_dict['labels'] = labels

                    info[model['id_cad']][frame] = frame_dict
                
                    
            info_all[scene_id] = info
            shutil.make_archive(masks_dir, 'zip', masks_dir)
            shutil.rmtree(masks_dir)
                    
                    
output_json_dir = str(output_dir)+'/'+ 'info.json'
with open(output_json_dir, 'w') as fp:
    json.dump(info_all, fp)