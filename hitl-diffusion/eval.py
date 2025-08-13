if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from train import TrainHITLWorkspace

import zmq
import json
import numpy as np
import time

from scipy.spatial.transform import Rotation as R

import msgpack
import msgpack_numpy
msgpack_numpy.patch()  # adds np.ndarray support
msgpack_numpy_encode = msgpack_numpy.encode
msgpack_numpy_decode = msgpack_numpy.decode

OmegaConf.register_new_resolver("eval", eval, replace=True)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    workspace = TrainHITLWorkspace(cfg)
    workspace.eval(server=cfg.server)

    if cfg.server:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://192.168.1.161:5555")  # Listen on all interfaces

        print("Server up!")

        while True:
            message = socket.recv()
            
            if message.startswith(b'\x81\xa4ping'):  # msgpack-encoded dict with 'ping'
                socket.send(msgpack.packb({"pong": True}, use_bin_type=True))
                continue

            data = msgpack.unpackb(message, object_hook=msgpack_numpy_decode, raw=False)

            obs_dict = {
                "point_cloud": torch.from_numpy(np.expand_dims(data['point_cloud'], axis=0)).cuda(non_blocking=True),
                "agent_pos": torch.from_numpy(np.expand_dims(data['agent_pos'], axis=0)).cuda(non_blocking=True)
            }

            # Run inference
            start_time = time.time()
            with torch.no_grad():
                result = workspace.model_inference(server_call=True, data=obs_dict)
            inference_time = time.time() - start_time
            print(f"Inference took {inference_time:.6f} seconds")

            # Convert quaternions to Euler in one shot (vectorized)
            action_quats = result['action_pred'].cpu().numpy()  # shape: (1, horizon, 4)
            euler_deg = R.from_quat(action_quats[0]).as_euler('xyz', degrees=True)  # shape: (horizon, 3)

            # Send back
            response = {
                "action": euler_deg.tolist()
            }
            socket.send(msgpack.packb(response, default=msgpack_numpy_encode, use_bin_type=True))

if __name__ == "__main__":
    main()
