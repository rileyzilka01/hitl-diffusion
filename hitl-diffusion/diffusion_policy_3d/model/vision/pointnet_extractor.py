import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import resnet34, resnet18
import torchvision
import copy
from PIL import Image

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()

    
_encoders = {'resnet34' : resnet34, 'resnet18' : resnet18, }
# do the transforms beforehand
# _transforms = {
# 	'resnet34' :
# 		transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
#     'resnet18' :
# 		transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])
#     ]),
# }

class ImgEncoder(nn.Module):
    def __init__(self, encoder_type):
        super(ImgEncoder, self).__init__()
        self.encoder_type = encoder_type
        if self.encoder_type in _encoders :
            self.model = _encoders[encoder_type](pretrained=True)
        else :
            print("Please enter a valid encoder type")
            raise Exception
        for param in self.model.parameters():
            param.requires_grad = False
        if self.encoder_type in _encoders :
            num_ftrs = self.model.fc.in_features
            self.num_ftrs = num_ftrs
            self.model.fc = Identity() # fc layer is replaced with identity

    def forward(self, x):
        x = self.model(x)
        return x

    # the transform is resize - center crop - normalize (imagenet normalize) No data aug here

    def get_features(self, x):
        with torch.no_grad():
            z = self.model(x)
        return z.cpu().data.numpy() ### Can't store everything in GPU :/

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 rgb_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        # self.back_point_cloud_key = 'back_point_cloud'
        # self.wrist_point_cloud_key = 'wrist_point_cloud'
        self.back_rgb_key = 'back_rgb'
        self.wrist_rgb_key = 'wrist_rgb'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        # self.back_point_cloud_shape = observation_space[self.back_point_cloud_key]
        # self.wrist_point_cloud_shape = observation_space[self.wrist_point_cloud_key]
        self.back_rgb_shape = observation_space[self.back_rgb_key]
        self.wrist_rgb_shape = observation_space[self.wrist_rgb_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        # cprint(f"[DP3Encoder] back point cloud shape: {self.back_point_cloud_shape}", "yellow")
        # cprint(f"[DP3Encoder] wrist point cloud shape: {self.wrist_point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] back rgb shape: {self.back_rgb_shape}", "yellow")
        cprint(f"[DP3Encoder] wrist rgb shape: {self.wrist_rgb_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            pointcloud_encoder_cfg.in_channels = 3
            # self.back_p_extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            # self.wrist_p_extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
            self.back_i_extractor = ImgEncoder(**rgb_encoder_cfg)
            self.wrist_i_extractor = ImgEncoder(**rgb_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")
    
    def forward(self, observations: Dict) -> torch.Tensor:
        # back_points = observations[self.back_point_cloud_key]
        # wrist_points = observations[self.wrist_point_cloud_key]
        back_rgb = observations[self.back_rgb_key]
        wrist_rgb = observations[self.wrist_rgb_key]

        # assert len(back_points.shape) == 3, cprint(f"point cloud shape: {back_points.shape}, length should be 3", "red")
        # if self.use_imagined_robot:
        #     img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
        #     points = torch.concat([points, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        # back_pn_feat = self.back_p_extractor(back_points)    # B * out_channel
        # wrist_pn_feat = self.wrist_p_extractor(wrist_points)    # B * out_channel
        back_in_feat = self.back_i_extractor(back_rgb)    # B * out_channel
        wrist_in_feat = self.wrist_i_extractor(wrist_rgb)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([back_in_feat, wrist_in_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
