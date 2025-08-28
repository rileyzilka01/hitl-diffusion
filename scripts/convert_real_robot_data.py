import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
from termcolor import cprint
import re
import time


import numpy as np
import torch
import pytorch3d.ops as torch3d_ops
import torchvision
import socket
import pickle
import cv2
from scipy.spatial.transform import Rotation as R
import cv2
import torchvision.transforms.functional as F

def preprocess_image(image):
    if image.dtype != np.float32:
        image = image.astype(np.float32)
    if image.max() > 1.0:
        image /= 255.0


    # Convert to torch tensor and channel-first
    image = torch.from_numpy(image).cuda()  # [H, W, C]
    image = image.permute(2, 0, 1)   # -> [C, H, W]

    # Resize & crop like torchvision transforms
    image = F.resize(image, [224, 224])                # shorter side = 256

    # Normalize for ResNet (ImageNet stats)
    image = F.normalize(
        image,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    image = image.cpu().numpy()

    return image

expert_data_path = '/home/serg/projects/png_vision/data/block'
save_data_path = '/home/serg/projects/hitl-diffusion/hitl-diffusion/data/hitl_block.zarr'
dirs = os.listdir(expert_data_path)
dirs = sorted([int(d) for d in dirs])

demo_dirs = [os.path.join(expert_data_path, str(d)) for d in dirs if os.path.isdir(os.path.join(expert_data_path, str(d)))]

# storage
total_count = 0
# img_arrays = []
back_point_cloud_arrays = []
wrist_point_cloud_arrays = []
# depth_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []

use_gripper = False
ee_centroid = False
joint_pos = False
stage = True

if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)


for demo_dir in demo_dirs:
    cprint('Processing {}'.format(demo_dir), 'green')

    prev_ee_pos = None
    prev_ee_ori = None
    demo_timesteps = sorted([int(d) for d in os.listdir(demo_dir)])

    for step_idx in tqdm.tqdm(range(len(demo_timesteps))):
        timestep_dir = os.path.join(demo_dir, str(step_idx))

        state_info = np.load(os.path.join(timestep_dir, 'low_dim.npy'), allow_pickle=True).item()

        curr_ee_pos = np.array(state_info['ee_position'])
        curr_ee_ori = np.array(state_info['ee_orientation'])

        back_pointcloud = np.load(os.path.join(timestep_dir, 'back_depth.npy'))
        back_pointcloud[..., 3:6]
        # wrist_pointcloud = np.load(os.path.join(timestep_dir, 'wrist_depth.npy'))

        gripper = 1 if state_info['joints']['position'][7] > 0.3 else 0

        robot_state = np.hstack([curr_ee_pos])
        if prev_ee_pos is None:
            # action = [0, 0, 0] # Only happens for the first timestep
            prev_ee_pos = state_info['ee_position']
            prev_ee_ori = state_info['ee_orientation']

            action_arrays.append(curr_ee_ori)
            back_point_cloud_arrays.append(back_pointcloud)

            state_arrays.append(robot_state)
            total_count += 1
        else:
            # prev_quat = R.from_euler('xyz', prev_ee_ori, degrees=True)
            # curr_quat = R.from_euler('xyz', curr_ee_ori, degrees=True)

            # ori_diff = [(curr_ee_ori[i] - prev_ee_ori[i] + 180) % 360 - 180 for i in range(len(curr_ee_ori))]
            ori_diff = [(curr_ee_ori[i] - prev_ee_ori[i] + np.pi/2) % np.pi - np.pi/2 for i in range(len(curr_ee_ori))]
            # ori_diff = (prev_quat * curr_quat.inv()).as_rotvec()

            # take a timstep only if the orientation is different enough from a previous step
            if (np.any(np.abs(ori_diff) > [0.015, 0.015, 0.015])):
            # if (np.any(np.abs(ori_diff) > [0.85, 0.85, 0.85])):
                ori_diff = R.from_euler('xyz', ori_diff, degrees=True).as_rotvec()
                action = np.array(curr_ee_ori)
                prev_ee_pos = state_info['ee_position']
                prev_ee_ori = state_info['ee_orientation']

                action_arrays.append(action)
                back_point_cloud_arrays.append(back_pointcloud)
                state_arrays.append(robot_state)

                total_count += 1

    prev_ee_ori = None
    prev_ee_pos = None
    
    episode_ends_arrays.append(total_count)


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# back_rgb_arrays = np.stack(back_rgb_arrays, axis=0)
# wrist_rgb_arrays = np.stack(wrist_rgb_arrays, axis=0)
# if img_arrays.shape[1] == 3: # make channel last
    # img_arrays = np.transpose(img_arrays, (0,2,3,1))
back_point_cloud_arrays = np.stack(back_point_cloud_arrays, axis=0)
# wrist_point_cloud_arrays = np.stack(wrist_point_cloud_arrays, axis=0)
# depth_arrays = np.stack(depth_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
# back_rgb_chunk_size = (100, back_rgb_arrays.shape[1], back_rgb_arrays.shape[2], back_rgb_arrays.shape[3])
# wrist_rgb_chunk_size = (100, wrist_rgb_arrays.shape[1], wrist_rgb_arrays.shape[2], wrist_rgb_arrays.shape[3])
back_point_cloud_chunk_size = (100, back_point_cloud_arrays.shape[1], back_point_cloud_arrays.shape[2])
# wrist_point_cloud_chunk_size = (100, wrist_point_cloud_arrays.shape[1], wrist_point_cloud_arrays.shape[2])
# depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
# zarr_data.create_dataset('back_rgb', data=back_rgb_arrays, chunks=back_rgb_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
# zarr_data.create_dataset('wrist_rgb', data=wrist_rgb_arrays, chunks=wrist_rgb_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('back_point_cloud', data=back_point_cloud_arrays, chunks=back_point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
# zarr_data.create_dataset('wrist_point_cloud', data=wrist_point_cloud_arrays, chunks=wrist_point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
# zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
# cprint(f'back_rgb shape: {back_rgb_arrays.shape}, range: [{np.min(back_rgb_arrays)}, {np.max(back_rgb_arrays)}]', 'green')
# cprint(f'wrist_rgb shape: {wrist_rgb_arrays.shape}, range: [{np.min(wrist_rgb_arrays)}, {np.max(wrist_rgb_arrays)}]', 'green')
cprint(f'back pc shape: {back_point_cloud_arrays.shape}, range: [{np.min(back_point_cloud_arrays)}, {np.max(back_point_cloud_arrays)}]', 'green')
# cprint(f'wrist pc shape: {wrist_point_cloud_arrays.shape}, range: [{np.min(wrist_point_cloud_arrays)}, {np.max(wrist_point_cloud_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

