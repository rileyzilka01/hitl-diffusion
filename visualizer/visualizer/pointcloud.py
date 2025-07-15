from flask import Flask, render_template_string
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import matplotlib.cm as cm
from termcolor import cprint
import os
import torch
import pytorch3d.ops as torch3d_ops
    
class Visualizer:
    def __init__(self):
        self.app = Flask(__name__)
        self.pointclouds = []
        
    def _generate_trace(self, pointcloud, color:tuple=None, size=5, opacity=0.7):
        x_coords = pointcloud[:, 0]
        y_coords = pointcloud[:, 1]
        z_coords = pointcloud[:, 2]

        if pointcloud.shape[1] == 3:
            if color is None:
                # design a colorful point cloud based on 3d coordinates
                # Normalize coordinates to range [0, 1]
                min_coords = pointcloud.min(axis=0)
                max_coords = pointcloud.max(axis=0)
                normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
                try:
                    # Use normalized coordinates as RGB values
                    colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
                except: # maybe meet NaN error
                    # use simple cyan color
                    colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]
            else:    
                colors = ['rgb({},{},{})'.format(color[0], color[1], color[2]) for _ in range(len(x_coords))]
        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]

        return go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='markers',
            marker=dict(
                size=size,
                opacity=opacity,
                color=colors
            )
        )


    def colorize(self, pointcloud):
        if pointcloud.shape[1] == 3:

            # design a colorful point cloud based on 3d coordinates
            # Normalize coordinates to range [0, 1]
            min_coords = pointcloud.min(axis=0)
            max_coords = pointcloud.max(axis=0)
            normalized_coords = (pointcloud - min_coords) / (max_coords - min_coords)
            try:
                # Use normalized coordinates as RGB values
                colors = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in normalized_coords]
            except: # maybe meet NaN error
                # use simple cyan color
                x_coords = pointcloud[:, 0]
                colors = ['rgb(0,255,255)' for _ in range(len(x_coords))]

        else:
            colors = ['rgb({},{},{})'.format(int(r), int(g), int(b)) for r, g, b in pointcloud[:, 3:6]]
        return colors

    
    def farthest_point_sampling(self, points, num_points=1024, use_cuda=True):
        K = [num_points]
        if use_cuda:
            points = torch.from_numpy(points).cuda()
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.cpu().numpy()
        else:
            points = torch.from_numpy(points)
            sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
            sampled_points = sampled_points.squeeze(0)
            sampled_points = sampled_points.numpy()

        return sampled_points, indices
    

    def visualize_pointcloud(self, pointcloud, color:tuple=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                yaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                zaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                bgcolor='white',
            )
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)

    def visualize_pointclouds(self, pointclouds, color:tuple=None):
        all_frames = []
        first_trace = None

        for i, pointcloud in enumerate(pointclouds):
            print(f"Pointcloud {i}")
            trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)

            if i == 0:
                first_trace = trace

            all_frames.append(go.Frame(data=[trace], name=str(i)))

        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout, frames=all_frames)

        camera = dict(
            eye=dict(x=1.25, y=1.25, z=1.25)  # Or adjust as needed
        )

        # Create animated figure
        fig.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                yaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                zaxis=dict(showbackground=False, showgrid=True, showline=True, linecolor='grey', zerolinecolor='grey', zeroline=False, gridcolor='grey'),
                bgcolor='white',
                camera=camera
            ),
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                buttons=[dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 500, "redraw": False},
                        "transition": {"duration": 0},
                        "fromcurrent": True,
                        "mode": "immediate",
                        "loop": True
                    }]
                )]
            )]
        )
        div = pio.to_html(fig, full_html=False)

        @self.app.route('/')
        def index():
            return render_template_string('''<div>{{ div|safe }}</div>''', div=div)
        
        self.app.run(debug=True, use_reloader=False)

    def visualize_pointcloud_and_save(self, pointcloud, color:tuple=None, save_path=None):
        trace = self._generate_trace(pointcloud, color=color, size=6, opacity=1.0)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = go.Figure(data=[trace], layout=layout)
        
        fig.update_layout(
            
            scene=dict(
                # aspectmode='cube', 
                xaxis=dict(
                    showbackground=False,  # 隐藏背景网格
                    showgrid=True,        # 隐藏网格
                    showline=True,         # 显示轴线
                    linecolor='grey',      # 设置轴线颜色为灰色
                    zerolinecolor='grey',  # 设置0线颜色为灰色
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                    
                ),
                yaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                zaxis=dict(
                    showbackground=False,
                    showgrid=True,
                    showline=True,
                    linecolor='grey',
                    zerolinecolor='grey',
                    zeroline=False,        # 关闭0线
                    gridcolor='grey',      # 设置网格颜色为灰色
                ),
                bgcolor='white'  # 设置背景色为白色
            )
        )
        # save
        fig.write_image(save_path, width=800, height=600)
        

    def save_visualization_to_file(self, pointcloud, file_path, color:tuple=None):
        # visualize pointcloud and save as html
        trace = self._generate_trace(pointcloud, color=color)
        layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig_html = pio.to_html(go.Figure(data=[trace], layout=layout), full_html=True)

        with open(file_path, 'w') as file:
            file.write(fig_html)
        print(f"Visualization saved to {file_path}")
    

def plot_sequence():
    expert_data_path = '/home/rzilka/hitl-diffusion/data/bowl/0'
    dirs = os.listdir(expert_data_path)
    dirs = sorted([int(d) for d in dirs])

    pc_paths = [os.path.join(expert_data_path, str(d), 'back_depth.npy') for d in dirs if os.path.isdir(os.path.join(expert_data_path, str(d)))] 
        
    vis = Visualizer()

    WORK_SPACE = [
        [-0.4, 0.4],
        [-20, 20],
        [0, 1.1]
    ]

    pcs = []
    for pc_path in pc_paths:
        print(pc_path)
        pc = np.load(pc_path)
        pc = pc[...,:3]

        # crop
        # pc = pc[np.where(
        #     (pc[..., 0] > WORK_SPACE[0][0]) & (pc[..., 0] < WORK_SPACE[0][1]) &
        #     (pc[..., 1] > WORK_SPACE[1][0]) & (pc[..., 1] < WORK_SPACE[1][1]) &
        #     (pc[..., 2] > WORK_SPACE[2][0]) & (pc[..., 2] < WORK_SPACE[2][1])
        # )]

        pc, sample_indices = vis.farthest_point_sampling(pc, use_cuda=True)
        pcs.append(pc)

    color:tuple=None
    vis.visualize_pointclouds(pcs, color=color)

def plot_one():
    # pc_path = '/home/rzilka/hitl-diffusion/data/bowl/0/10/back_depth.npy'
    pc_path = '/home/rzilka/hitl-diffusion/data/bowl/0/10/low_dim.npy'
        
    vis = Visualizer()

    pc = np.load(pc_path, allow_pickle=True)
    print(pc)

    x = a

    pc = pc[...,:3]

    # Crop
    # WORK_SPACE = [
    #     [-0.4, 0.4],
    #     [-20, 20],
    #     [0, 1.1]
    # ]

    # pc = pc[np.where(
    #     (pc[..., 0] > WORK_SPACE[0][0]) & (pc[..., 0] < WORK_SPACE[0][1]) &
    #     (pc[..., 1] > WORK_SPACE[1][0]) & (pc[..., 1] < WORK_SPACE[1][1]) &
    #     (pc[..., 2] > WORK_SPACE[2][0]) & (pc[..., 2] < WORK_SPACE[2][1])
    # )]

    pc, sample_indices = vis.farthest_point_sampling(pc, use_cuda=True)

    color:tuple=None
    vis.visualize_pointcloud(pc, color=color)


if __name__ == "__main__":
    # plot_sequence()
    plot_one()