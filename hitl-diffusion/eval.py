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
        socket.bind("tcp://192.168.1.186:5555")  # Listen on all interfaces

        print("Server up!")

        while True:
            message = socket.recv()
            data = json.loads(message.decode('utf-8'))
            
            if data.get("ping", False):
                socket.send_string(json.dumps({"pong": True}))
                continue

            obs_dict = {
                "point_cloud": torch.tensor(np.expand_dims(data['point_cloud'], axis=0)).cuda(),
                "agent_pos": torch.tensor(np.expand_dims(data['agent_pos'], axis=0)).cuda()
            }

            with torch.no_grad():
                result = workspace.model_inference(server_call=True, data=obs_dict)

            action = result['action_pred'].cpu().numpy().tolist()
            response = json.dumps({"action": action})
            socket.send_string(response)

if __name__ == "__main__":
    main()
