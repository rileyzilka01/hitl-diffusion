from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.replay_buffer import ReplayBuffer
from diffusion_policy_3d.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy_3d.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy_3d.dataset.base_dataset import BaseDataset

class HitlDataset(BaseDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=1,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['state', 'action', 'back_point_cloud', 'wrist_point_cloud', 'back_rgb', 'wrist_rgb'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'], # EE orientation
            # 'agent_pos': self.replay_buffer['state'][...,:], # Joint position and EE position, '...,:' selects all dimensions of array for variable size array (different tasks have different state dimensions)
            'agent_pos': self.replay_buffer['state'][...,:], # ee pose
            'back_point_cloud': self.replay_buffer['back_point_cloud'], # Colorless point cloud
            'wrist_point_cloud': self.replay_buffer['wrist_point_cloud'], # Colorless point cloud
            # 'back_img': self.replay_buffer['back_img'], # Colorless point cloud
            # 'wrist_img': self.replay_buffer['wrist_img'], # Colorless point cloud
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        # normalizer['back_point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        # normalizer['wrist_point_cloud'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['back_img'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['wrist_img'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,].astype(np.float32) # (agent_posx2, block_posex3)
        wrist_point_cloud = sample['wrist_point_cloud'][:,].astype(np.float32) # (T, 512, 3)
        back_point_cloud = sample['back_point_cloud'][:,].astype(np.float32) # (T, 512, 3)
        wrist_rgb = sample['wrist_rgb'][:,].astype(np.float32) # (T, 640, 480)
        back_rgb = sample['back_rgb'][:,].astype(np.float32) # (T, 640, 480)

        data = {
            'obs': {
                'back_point_cloud': back_point_cloud, # T, 1024, 6
                'wrist_point_cloud': wrist_point_cloud, # T, 1024, 6
                'back_rgb': back_rgb, # T, 640, 480
                'wrist_rgb': wrist_rgb, # T, 640, 480
                'agent_pos': agent_pos, # T, D_pos
            },
            'action': sample['action'].astype(np.float32) # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

