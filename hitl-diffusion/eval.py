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
            data = json.loads(message.decode('utf-8'))
            
            if data.get("ping", False):
                socket.send_string(json.dumps({"pong": True}))
                continue

            obs_dict = {
                "point_cloud": torch.tensor(np.expand_dims(data['back_point_cloud'], axis=0)).cuda(),
                "agent_pos": torch.tensor(np.expand_dims(data['agent_pos'], axis=0)).cuda()
            }

            start_time = time.time()
            with torch.no_grad():
                result = workspace.model_inference(server_call=True, data=obs_dict)
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.6f} seconds")

            action = result['action_pred'].cpu().numpy().tolist()[0]

            new_action = []
            for i in range(len(action)):
                rotvec = R.from_rotvec(action[i])
                deg = rotvec.as_euler('xyz', degrees=True)
                # deg = r_back.as_euler('xyz', degrees=True)
                new_action.append(rotvec)

            response = json.dumps({"action": new_action})
            socket.send_string(response)

if __name__ == "__main__":
    main()
