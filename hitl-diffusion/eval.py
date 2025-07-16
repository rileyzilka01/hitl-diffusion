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

from flask import Flask, request, jsonify
import threading

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
        app = Flask(__name__)

        # Load your workspace once and reuse
        workspace = TrainHITLWorkspace(cfg)
        workspace.eval()

        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                data = request.get_json(force=True)
                # Convert data to appropriate format for predict_action
                obs_dict = {
                    "point_cloud": torch.tensor(data['point_cloud']).unsqueeze(0).cuda()  # shape [1, T, N, 3]
                    "agent_pos": torch.tensor(data['agent_pos']).unsqueeze(0).cuda()  # shape [1, T, N, 3]
                }

                result = workspace.model_inference(server_call=True, data=obs_dict)

                action = result['action'].cpu().numpy().tolist()
                return jsonify({"action": action})
            except Exception as e:
                return jsonify({"error": str(e)}), 500

        print("🚀 Server running at http://localhost:5000")
        app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
