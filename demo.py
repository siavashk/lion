# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    require diffusers-0.11.1
"""
import clip
import torch
from default_config import cfg as config
from models.lion import LION
import numpy as np
import pyvista as pv

model_path = '/home/ubuntu/code/exp/0716/c1/5e4d7ch_train_lion_B8/checkpoints/epoch_1999_iters_9999.pt'
model_config = '/home/ubuntu/code/exp/0716/c1/5e4d7ch_train_lion_B8/cfg.yml'

config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)

if config.clipforge.enable:
    input_t = ["a swivel chair, five wheels"] 
    device_str = 'cuda'
    clip_model, clip_preprocess = clip.load(
                        config.clipforge.clip_model, device=device_str)    
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    print('clip_feat', clip_feat.shape)
else:
    clip_feat = None
# output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat)
output = lion.sample(1, clip_feat=None)
pts = output['points']
pts = pts.cpu().numpy().squeeze(0)

# print(np.min(pts, axis=0))
# print(np.max(pts, axis=0))

print("meshing...")
point_cloud = pv.PolyData(pts)

volume = point_cloud.delaunay_3d(alpha=0.08)

mesh = volume.extract_geometry()
mesh.save("c1.vtk")
