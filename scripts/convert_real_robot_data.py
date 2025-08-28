import os
import zarr
import tqdm
import numpy as np
from termcolor import cprint


import numpy as np
import sys
from scipy.spatial.transform import Rotation as R

def select_evenly_spaced(array, max_length=48):
    n = len(array)
    if n <= max_length:
        return array
    indices = np.linspace(0, n - 1, max_length, dtype=int)
    return [array[i] for i in indices]

if len(sys.argv) < 3:
    print("Usage: python scripts/convert_real_robot_data.py <input_dataset_name> <output_dataset_name>")
    sys.exit(1)

if os.getenv("USER") == "serg":
    expert_data_path = f'/home/serg/projects/png_vision/data/{sys.argv[1]}'
    save_data_path = f'/home/serg/projects/hitl-diffusion/hitl-diffusion/data/{sys.argv[2]}.zarr'
elif os.getenv("USER") == "rzilka":
    expert_data_path = f'/home/rzilka/png_vision/data/{sys.argv[1]}'
    save_data_path = f'/home/rzilka/hitl-diffusion/hitl-diffusion/data/{sys.argv[2]}.zarr'
else:
    raise NotImplementedError("who dis")

dirs = os.listdir(expert_data_path)
dirs = sorted([int(d) for d in dirs])

demo_dirs = [os.path.join(expert_data_path, str(d)) for d in dirs if os.path.isdir(os.path.join(expert_data_path, str(d)))]

# storage
total_count = 0
point_cloud_arrays = []
state_arrays = []
action_arrays = []
episode_ends_arrays = []
record_frame = False

use_gripper = False
ee_centroid = False
joint_pos = False
stage = True
evenly_spaced = False

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
    if evenly_spaced:
        demo_timesteps = select_evenly_spaced(demo_timesteps, max_length=512)

    for step_idx in tqdm.tqdm(range(len(demo_timesteps))):
        timestep_dir = os.path.join(demo_dir, str(step_idx))

        state_info = np.load(os.path.join(timestep_dir, 'low_dim.npy'), allow_pickle=True).item()
        gripper = 1 if state_info['joints']['position'][7] > 0.3 else 0

        if use_gripper:
            robot_state = np.hstack([state_info['joints']['position'][:7], [gripper], state_info['ee_position'], ([state_info['stage']] if stage else [])])
        else:
            robot_state = np.hstack([state_info['joints']['position'][:7], state_info['ee_position'], ([state_info['stage']] if stage else [])])


        if not joint_pos:
            robot_state = robot_state[7:]


        curr_ee_pos = np.array(state_info['ee_position'])
        curr_ee_ori = np.array(state_info['ee_orientation'])

        obs_pointcloud = np.load(os.path.join(timestep_dir, 'depth.npy'))

        if ee_centroid:
            a = state_info['ee_position']
            b = np.mean(obs_pointcloud[..., :3], axis=0)
            robot_state += [a[i] - b[i] for i in range(3)]


        if evenly_spaced or prev_ee_pos is None:
            record_frame = True
        else:
            # prev_quat = R.from_euler('xyz', prev_ee_ori, degrees=True)
            # curr_quat = R.from_euler('xyz', curr_ee_ori, degrees=True)

            # ori_diff = [(curr_ee_ori[i] - prev_ee_ori[i] + 180) % 360 - 180 for i in range(len(curr_ee_ori))]
            ori_diff = [(curr_ee_ori[i] - prev_ee_ori[i] + np.pi/2) % np.pi - np.pi/2 for i in range(len(curr_ee_ori))]
            pos_diff = curr_ee_pos - prev_ee_pos
            # ori_diff = (prev_quat * curr_quat.inv()).as_rotvec()

            # take a timstep only if the orientation is different enough from a previous step
            if (np.any(np.abs(ori_diff) > [0.015, 0.015, 0.015]) or np.any(np.abs(pos_diff) > [0.005, 0.005, 0.005])):
            # if (np.any(np.abs(ori_diff) > [0.85, 0.85, 0.85])):

                record_frame = True

        if record_frame:
            action = np.array(curr_ee_ori)
            prev_ee_pos = state_info['ee_position']
            prev_ee_ori = state_info['ee_orientation']

            action_arrays.append(action)
            point_cloud_arrays.append(obs_pointcloud)
            state_arrays.append(robot_state)

            total_count += 1
            record_frame = False

    episode_ends_arrays.append(total_count)


# create zarr file
zarr_root = zarr.group(save_data_path)
zarr_data = zarr_root.create_group('data')
zarr_meta = zarr_root.create_group('meta')

point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
action_arrays = np.stack(action_arrays, axis=0)
state_arrays = np.stack(state_arrays, axis=0)
episode_ends_arrays = np.array(episode_ends_arrays)

compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
if len(action_arrays.shape) == 2:
    action_chunk_size = (100, action_arrays.shape[1])
elif len(action_arrays.shape) == 3:
    action_chunk_size = (100, action_arrays.shape[1], action_arrays.shape[2])
else:
    raise NotImplementedError
zarr_data.create_dataset('point_cloud', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float64', overwrite=True, compressor=compressor)
zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
zarr_data.create_dataset('state', data=state_arrays, chunks=(100, state_arrays.shape[1]), dtype='float32', overwrite=True, compressor=compressor)
zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, chunks=(100,), dtype='int64', overwrite=True, compressor=compressor)

# print shape
cprint(f'pc shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
cprint(f'episode_ends shape: {episode_ends_arrays.shape}, range: [{np.min(episode_ends_arrays)}, {np.max(episode_ends_arrays)}]', 'green')
cprint(f'total_count: {total_count}', 'green')
cprint(f'Saved zarr file to {save_data_path}', 'green')

