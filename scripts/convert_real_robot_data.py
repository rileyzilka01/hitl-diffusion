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



def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    K = [num_points]
    if use_cuda:
        points = torch.from_numpy(points).cuda()
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.cpu().numpy()
    else:
        points = torch.from_numpy(points)
        sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
        sampled_points = sampled_points.squeeze(0)
        sampled_points = sampled_points.numpy()

    return sampled_points, indices

def preprocess_point_cloud(points, use_cuda=True):
    
    num_points = 1024

    extrinsics_matrix = get_homogenous_matrix()

    point_xyz = points[..., :3]
    point_xyz = point_xyz - [-0.01789913, -0.02264747, 1.24600857]
    point_homogeneous = np.hstack((point_xyz, np.ones((point_xyz.shape[0], 1))))
    point_homogeneous = np.dot(point_homogeneous, extrinsics_matrix)
    point_xyz = point_homogeneous[..., :-1]
    points[..., :3] = point_xyz
    
    # Crop
    WORK_SPACE = [
        [-0.4, 0.5],
        [-0.3, 1],
        [-0.2, 0.3]
    ]

    points = points[np.where(
        (points[..., 0] > WORK_SPACE[0][0]) & (points[..., 0] < WORK_SPACE[0][1]) &
        (points[..., 1] > WORK_SPACE[1][0]) & (points[..., 1] < WORK_SPACE[1][1]) &
        (points[..., 2] > WORK_SPACE[2][0]) & (points[..., 2] < WORK_SPACE[2][1])
    )]
    
    points_xyz = points[..., :3]
    points_xyz, sample_indices = farthest_point_sampling(points_xyz, num_points, use_cuda)
    sample_indices = sample_indices.cpu()
    points_rgb = points[sample_indices, 3:][0]
    points = np.hstack((points_xyz, points_rgb))
    return points

def get_homogenous_matrix():
    rx_deg = 55  # Rotation around X
    ry_deg = 235  # Rotation around Y
    rz_deg = 35  # Rotation around Z

    # Convert to radians
    rx = np.radians(rx_deg)
    ry = np.radians(ry_deg)
    rz = np.radians(rz_deg)

    # Rotation matrix around X-axis
    Rx = np.array([
        [1, 0,          0,           0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx),  0],
        [0, 0,          0,           1]
    ])

    # Rotation matrix around Y-axis
    Ry = np.array([
        [np.cos(ry),  0, np.sin(ry), 0],
        [0,           1, 0,          0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0,           0, 0,          1]
    ])

    # Rotation matrix around Z-axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz),  np.cos(rz), 0, 0],
        [0,           0,          1, 0],
        [0,           0,          0, 1]
    ])

    # Original extrinsics matrix (identity in this case)
    extrinsics_matrix = np.eye(4)

    # Combine rotations (Z * Y * X) — typical convention (can change based on your coordinate system)
    rotation_combined = Rz @ Ry @ Rx

    # Apply rotation to extrinsics
    rotated_extrinsics = rotation_combined @ extrinsics_matrix

    return rotated_extrinsics

def select_evenly_spaced(array, max_length=48):
    n = len(array)
    if n <= max_length:
        return array
    indices = np.linspace(0, n - 1, max_length, dtype=int)
    return [array[i] for i in indices]
   
def preproces_image(image):
    img_size = 84
    
    image = image.astype(np.float32)
    image = torch.from_numpy(image).cuda()
    image = image.permute(2, 0, 1) # HxWx4 -> 4xHxW
    image = torchvision.transforms.functional.resize(image, (img_size, img_size))
    image = image.permute(1, 2, 0) # 4xHxW -> HxWx4
    image = image.cpu().numpy()
    return image



expert_data_path = '/home/rzilka/hitl-diffusion/data/bowl'
save_data_path = '/home/rzilka/hitl-diffusion/hitl-diffusion/data/hitl_block.zarr'
dirs = os.listdir(expert_data_path)
dirs = sorted([int(d) for d in dirs])

demo_dirs = [os.path.join(expert_data_path, str(d)) for d in dirs if os.path.isdir(os.path.join(expert_data_path, str(d)))]

# storage
total_count = 0
# img_arrays = []
point_cloud_arrays = []
# depth_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []


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

    demo_timesteps = sorted([int(d) for d in os.listdir(demo_dir)])
    demo_timesteps = select_evenly_spaced(demo_timesteps)

    # For getting the difference instead of absolute orientation
    # prev_ee_orientation = None

    for step_idx in tqdm.tqdm(range(len(demo_timesteps))):
        timestep_dir = os.path.join(demo_dir, str(step_idx))

        # obs_image = demo['image'][step_idx]
        # obs_depth = demo['depth'][step_idx]
        # obs_image = preproces_image(obs_image)
        # obs_depth = preproces_image(np.expand_dims(obs_depth, axis=-1)).squeeze(-1)
        
        state_info = np.load(os.path.join(timestep_dir, 'low_dim.npy'), allow_pickle=True).item()
        robot_state = list(state_info['joints']['position'])[:7] + state_info['ee_position']
        # Comment this line to get difference instead of absolute orientation
        action = state_info['ee_orientation']

        # Getting the difference instead of absolute orientation
        # Comment out the next 5 lines to go back to position
        # ee_orientation = state_info['ee_orientation']
        # if prev_ee_orientation == None:
        #     action = [0, 0, 0]
        # else:
        #     action = [ee_orientation[i] - prev_ee_orientation[i] for i in range(len(ee_orientation))]

        obs_pointcloud = np.load(os.path.join(timestep_dir, 'back_depth.npy'))
        # obs_pointcloud = obs_pointcloud[...,:3]
        obs_pointcloud = preprocess_point_cloud(obs_pointcloud, use_cuda=True)
        # wrist_depth = np.load(os.path.join(demo_dir, '/wrist_depth.npy'))

        # img_arrays.append(obs_image)
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)
        # depth_arrays.append(obs_depth)
        state_arrays.append(robot_state)

        total_count += 1
    
    episode_ends_arrays.append(total_count)


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

# img_arrays = np.stack(img_arrays, axis=0)
# if img_arrays.shape[1] == 3: # make channel last
    # img_arrays = np.transpose(img_arrays, (0,2,3,1))
point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
# depth_arrays = np.stack(depth_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
# img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
# depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
# zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
# zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
# cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
# cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
# cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

