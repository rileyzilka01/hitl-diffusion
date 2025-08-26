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
import pickle

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
        socket.bind("tcp://0.0.0.0:5555")  # Listen on all interfaces

        print("Server up!")

        while True:
            parts = socket.recv_multipart()
            meta = pickle.loads(parts[0])
            
            if meta.get("ping", False):
                reply_meta = pickle.dumps({"pong": True})
                socket.send_multipart([reply_meta])
                continue

            obs_dict = {}
            for (key, md), buf in zip(meta.items(), parts[1:]):
                arr = np.frombuffer(buf, dtype=md["dtype"]).reshape(md["shape"])
                obs_dict[key] = torch.tensor(np.expand_dims(arr, axis=0)).cuda()

            with torch.no_grad():
                result = workspace.model_inference(server_call=True, data=obs_dict)

            action = result['action_pred'].cpu().numpy().tolist()[0]
            socket.send_multipart(pickle.dumps({"action": action}))

if __name__ == "__main__":
    main()
