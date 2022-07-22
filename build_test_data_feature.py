import os
import sys
import json
import yaml
import h5py
import torch
import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, '../'))

from utils.habitat_utils import HabitatUtils

import torchvision.transforms as transforms

from Backbone.segformer import Segformer



output_dir = '/cvhci/data/VisLoc/test_data/features/'
os.makedirs(output_dir, exist_ok=True)

device = torch.device('cuda')

model = Segformer()
model = model.to(device)

print('Loading pre-trained weights: ')
model_path = './runs/gru_fullrez_lastlayer_m256/20434/smnet_mp3d_best_model.pkl'
state = torch.load(model_path)
model_state = state['model_state']

weights={}
for k, v in model_state.items():
    if k.startswith('module.encoder'):
        k = '.'.join(k.split('.')[2:])
        weights[k] = v

model.load_state_dict(weights)
model.eval()

normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])



# -- -- Load json
paths = json.load(open('data/paths.json', 'r'))

envs_splits = json.load(open('data/envs_splits.json', 'r'))
test_envs = envs_splits['test_envs']
test_envs = [x for x in test_envs if x in paths]
test_envs.sort()

for env in test_envs:

    # if os.path.isfile(os.path.join(output_dir, env+'.h5')): continue

    # -- instantiate Habitat
    house, level = env.split('_')
    scene = 'data/mp3d/{}/{}.glb'.format(house, house)
    habitat = HabitatUtils(scene, int(level))

    path = paths[env]

    N = len(path['positions'])

    features_lastlayer = np.zeros((N,64,120,160), dtype=np.float32)


    with torch.no_grad():
        for n in tqdm(range(N)):
            pos = path['positions'][n]
            ori = path['orientations'][n]

            habitat.position = list(pos)
            habitat.rotation = list(ori)
            habitat.set_agent_state()

            # -- get semantic labels
            rgb = habitat.render()
            rgb = rgb.astype(np.float32)
            rgb = rgb / 255.0
            rgb = torch.FloatTensor(rgb).permute(2,0,1)
            rgb = normalize(rgb)
            rgb = rgb.unsqueeze(0).to(device)

            # depth_enc = habitat.render(mode='depth')
            # depth_enc = depth_enc[:,:,0]
            # depth_enc = depth_enc.astype(np.float32)
            # depth_enc = torch.FloatTensor(depth_enc).unsqueeze(0)
            # depth_enc = depth_normalize(depth_enc)
            # depth_enc = depth_enc.unsqueeze(0).to(device)

            semfeat_lastlayer = model(rgb)

            semfeat_lastlayer = semfeat_lastlayer[0].cpu().numpy()

            semfeat_lastlayer = semfeat_lastlayer.astype(np.float32)
            features_lastlayer[n,...] = semfeat_lastlayer


    filename = os.path.join(output_dir, env+'.h5')
    with h5py.File(filename, 'w') as f:
        f.create_dataset('features_lastlayer', data=features_lastlayer, dtype=np.float32)
    del habitat