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

if len(sys.argv) < 3:
    print("Usage: python scripts/convert_real_robot_data.py <input_dataset_name> <output_dataset_name>")
    sys.exit(1)

expert_data_path = f'/home/rzilka/png_vision/data/{sys.argv[1]}'
save_data_path = f'/home/rzilka/hitl-diffusion/hitl-diffusion/data/{sys.argv[2]}.zarr'
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

shared = True
demo_length = 1024
num_prompts = 3

train_demo_count = 5 #how many demonstrations to use from total, just takes first x
# train_demo_count = len(demo_dirs)+1 #default

centroid_only = False
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
    user_input = 'y'
    if user_input == 'y':
        cprint('Overwriting {}'.format(save_data_path), 'red')
        os.system('rm -rf {}'.format(save_data_path))
    else:
        cprint('Exiting', 'red')
        exit()
os.makedirs(save_data_path, exist_ok=True)

for demo_dir in demo_dirs[:train_demo_count]: 
    cprint('Processing {}'.format(demo_dir), 'green')

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
        if use_gripper:
            if shared:
                # ABSOLUTE
                robot_state = list(state_info['joints']['position'])[:8] + state_info['ee_position']
                # ABSOLUTE
            else:
                # DIFF
                if prev_joint_pos is None:
                    robot_state = list(np.zeros(8))
                    prev_joint_pos = list(state_info['joints']['position'])[:7]
                else:
                    current = list(state_info['joints']['position'])[:7]
                    robot_state = [current[i] - prev_joint_pos[i] for i in range(len(current))] + [0]
                    prev_joint_pos = current
                # DIFF

            robot_state[7] = 1 if robot_state[7] > 0.3 else -1
        else: # shared control
            centroids = list(state_info['centroids'])
            # print(f"CENTS: {[f'{abc: .4f}' for abc in centroids]}")
            if len(centroids) < 9:
                centroids += [0] * ((3*num_prompts)-len(centroids))
            differences = []
            for i in range(num_prompts):
                for j in range(i+1, num_prompts):
                    differences += [centroids[i*3] - centroids[j*3], centroids[(i*3)+1] - centroids[(j*3)+1], centroids[(i*3)+2] - centroids[(j*3)+2]]

            robot_state = list(state_info['joints']['position'])[:7] + state_info['ee_position'] + differences

        if not joint_pos:
            robot_state = robot_state[7:]

        obs_pointcloud = np.load(os.path.join(timestep_dir, 'depth.npy'), allow_pickle=True)

        ee_orientation = state_info['ee_orientation']

        # ABSOLUTE
        if not auto:
            action = ee_orientation # for shared control
        elif auto:
            action = robot_state # for auto
            robot_state = list(state_info['joints']['position'])[:7]
        # ABSOLUTE

        # TESTING ONLY HAVING DIFFERENCES
        if centroid_only:
            robot_state = robot_state[-9:]
            # print(f"ROBOS: {[f'{abc: .4f}' for abc in robot_state]}")

        # DIFF wasn't really useful because for shared controls diferences may be arbitrary, and if the human changes lots at first and not at goal nothing will happen
        # if prev_ee_orientation == None:
        #     action = [0, 0, 0]
        # else:
        #     # Consider angle wrapping
        #     action = [(ee_orientation[i] - prev_ee_orientation[i] + 180) % 360 - 180 for i in range(len(ee_orientation))]
        # prev_ee_orientation = ee_orientation

        # # Convert to quaternion (x, y, z, w)
        # euler = action
        # r = R.from_euler('xyz', euler, degrees=True)
        # action = r.as_quat()
        # DIFF

        # Point cloud is processed during recording now
        # obs_pointcloud = preprocess_point_cloud(obs_pointcloud, use_cuda=True)

        # img_arrays.append(obs_image)
        action_arrays.append(action)
        point_cloud_arrays.append(obs_pointcloud)
        # depth_arrays.append(obs_depth)
        state_arrays.append(robot_state)

        total_count += 1
    
    episode_ends_arrays.append(total_count)

if auto == True:
    action_arrays = [action_arrays[1:], action_arrays[0]]


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

