import os
import zarr
import pickle
import tqdm
import numpy as np
import torch
import torchvision
from termcolor import cprint
import time
import sys
import pickle
from scipy.spatial.transform import Rotation as R

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

def euler_to_matrix(euler_xyz):
    return R.from_euler('xyz', euler_xyz).as_matrix()

def matrix_to_euler(Rm):
    return R.from_matrix(Rm).as_euler('xyz')

def unit_vector_diff(a, b, eps=1e-8):
    # Normalize to unit vectors
    a_unit = a / (np.linalg.norm(a) + eps)
    b_unit = b / (np.linalg.norm(b) + eps)
    
    # Return the L2 distance between the tips of the vectors
    return np.linalg.norm(a_unit - b_unit, axis=-1)

def normalize(a, eps=1e-6):
    mag = np.linalg.norm(a)
    if mag > eps:
        a_norm = a / mag
    else:
        a_norm = a

    return a_norm

if len(sys.argv) < 4:
    print("Usage: python scripts/convert_real_robot_data.py <model> <input_dataset_name> <output_dataset_name>")
    sys.exit(1)
if len(sys.argv) == 5:
    print("Usage: python scripts/convert_real_robot_data.py <model> <input_dataset_name> <output_dataset_name> <input_path> <output_path>")
    sys.exit(1)
if len(sys.argv) == 6:
    expert_data_path = f'{sys.argv[4]}/{sys.argv[2]}'
    save_data_path = f'{sys.argv[5]}/{sys.argv[3]}.zarr'
else:
    expert_data_path = f'/home/rzilka/png_vision/data/{sys.argv[2]}'
    save_data_path = f'/home/rzilka/hitl-diffusion/hitl-diffusion/data/{sys.argv[3]}.zarr'

model = sys.argv[1]
if model not in ["hitl_d", "hitl_hgd"]:
    print("Model non existent usage: <hitl_d, hitl_hgd>")
    sys.exit(1)

dirs = os.listdir(expert_data_path)
dirs = sorted([int(d) for d in dirs])

demo_dirs = [os.path.join(expert_data_path, str(d)) for d in dirs if os.path.isdir(os.path.join(expert_data_path, str(d)))]

# storage
total_count = 0
img_arrays = []
point_cloud_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []

shared = True
num_prompts = 4

train_demo_count = 1 #how many demonstrations to use from total, just takes first x
# train_demo_count = len(demo_dirs)+1 #default

use_pointcloud = True # use pointcloud in conditioning or not
center_point_cloud = True
use_image = False

if model == "hitl_hgd":
    use_centroids = True
    use_norm_diffs = True
    use_ee_position = False
    demo_length = 1024
else:
    use_centroids = False
    use_norm_diffs = False
    use_ee_position = True
    demo_length = 256

if shared:
    # SHARED
    use_gripper = False
    joint_pos = False
    stage = False
    auto = False
    # SHARED
else:
    # AUTO
    use_gripper = True
    joint_pos = True
    stage = False
    auto = True
    # AUTO


if os.path.exists(save_data_path):
    cprint('Data already exists at {}'.format(save_data_path), 'red')
    cprint("If you want to overwrite, delete the existing directory first.", "red")
    cprint("Do you want to overwrite? (y/n)", "red")
    user_input = input("")
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for demo_dir in demo_dirs[:train_demo_count]: 
    cprint('Processing {}'.format(demo_dir), 'green')
    simulation_count = 1
    rstart = 0
    rmax = 1
    rstep = 1
    for angle in range(rstart, rmax, rstep):
        demo_timesteps = sorted([int(d) for d in os.listdir(demo_dir)])
        demo_timesteps = select_evenly_spaced(demo_timesteps, max_length=demo_length)

        # For getting the difference instead of absolute orientation
        prev_ee_orientation = None
        prev_joint_pos = None

        for step_idx in tqdm.tqdm(range(len(demo_timesteps))):
            timestep_dir = os.path.join(demo_dir, str(demo_timesteps[step_idx]))

            # obs_image = demo['image'][step_idx]
            # obs_depth = demo['depth'][step_idx]
            # obs_image = preproces_image(obs_image)
            # obs_depth = preproces_image(np.expand_dims(obs_depth, axis=-1)).squeeze(-1)
            
            state_info = np.load(os.path.join(timestep_dir, 'low_dim.npy'), allow_pickle=True).item()

            differences = []
            if use_centroids:
                centroids = list(state_info['centroids'])

                if len(centroids) < (3*num_prompts):
                    centroids += [0] * ((3*num_prompts)-len(centroids))

                for i in range(1, num_prompts): # skip the first centroid since its the red line and we dont use it for differencess
                    for j in range(i+1, num_prompts):
                        differences += [centroids[i*3] - centroids[j*3], centroids[(i*3)+1] - centroids[(j*3)+1], centroids[(i*3)+2] - centroids[(j*3)+2]]

            norm_diffs = []
            if use_norm_diffs:
                ee_vec = np.array(centroids[:3]) - np.array(centroids[3:6])
                ee_unit_vec = normalize(ee_vec)
                
                for i in range(num_prompts-2): # only get the end effector -> object differences not interobject differences
                    raw_target_dist = np.array(differences[i*3:(i+1)*3])
                    target_vec = normalize(raw_target_dist)
                
                    diff = unit_vector_diff(ee_unit_vec, target_vec)
                    norm_diffs.append(diff)

            # GET ROBOT STATE
            robot_state = []
            if joint_pos:
                robot_state += list(state_info['joints']['position'])[:7]

            if use_gripper:
                gripper_state = list(state_info['joints']['position'])[7]
                gripper_state = 1 if gripper_state > 0.3 else -1
                robot_state += [gripper_state]

            if use_ee_position:
                robot_state += state_info['ee_position']

            if use_centroids:
                robot_state += differences

            if use_norm_diffs:
                robot_state += norm_diffs
            # GET ROBOT STATE

            # POINTCLOUD
            if use_pointcloud:
                obs_pointcloud = np.load(os.path.join(timestep_dir, 'depth.npy'), allow_pickle=True)

                if center_point_cloud:
                    centroid = obs_pointcloud.mean(axis=0)
                    obs_pointcloud = obs_pointcloud - centroid
            # POINTCLOUD

            # IMAGE
            if use_image:
                obs_image = np.load(os.path.join(timestep_dir, 'img.npy'), allow_pickle=True)
            # IMAGE

            # ROBOT ACTION
            ee_orientation = state_info['ee_orientation']
            action = ee_orientation # for shared control
            # ROBOT ACTION

            action_arrays.append(action)

            if use_pointcloud:
                point_cloud_arrays.append(obs_pointcloud)
            if use_image:
                img_arrays.append(obs_image)

            state_arrays.append(robot_state)

            total_count += 1
        
        episode_ends_arrays.append(total_count)

if auto == True:
    action_arrays = [action_arrays[1:], action_arrays[0]]


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

if use_image:
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
if use_pointcloud:
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)

action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
if use_image:
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
if use_pointcloud:
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])

if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError

if use_image:
    zarr_data.create_dataset('img', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
if use_pointcloud:
    zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
if use_image:
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
if use_pointcloud:
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

