import os

os.environ['MUJOCO_GL'] = 'egl'
os.environ['PATH'] = '/usr/local/cuda-11.6/bin:' + os.environ['PATH']
if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../') #changing directories so that output/gsplat_full etc. exists

import mujoco
from utils.mujoco_utils import simulate_mujoco_scene

from contextlib import redirect_stdout
from video_api import initialize_gaussians
from gaussian_renderer import render
from scene.cameras import Camera_Pose 

import sys 
import torch 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

robot_model_path = 'output/universal_robots_ur5e_0625'
#assumes that the model and dataset are in the same directory as this notebook
sys.argv = ['']
gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians(model_path=robot_model_path)
n = len(kinematic_chain.get_joint_parameter_names())    

def display_render(image_tensor: torch.Tensor):
    if type(image_tensor) == torch.Tensor:
        image_tensor = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
    if image_tensor.dtype == torch.float32:
        image_tensor = (image_tensor * 255).astype(np.uint8)
    plt.imshow(image_tensor)
    plt.axis('off')
    plt.show()

example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(robot_model_path, 'robot_xml', 'scene.xml'))
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 224, 224)

joint_limits = np.array([model.jnt_range[i] for i in range(n)])
def unnormalize_joint_pose(joint_pose, joint_limits): #from [-1, 1] to [min, max]
    lower = joint_limits[:, 0]
    upper = joint_limits[:, 1]
    joint_pose = (joint_pose + 1)/2 * (upper - lower) + lower
    return joint_pose

unnormalized_joint_pose = unnormalize_joint_pose(example_camera.joint_pose.detach().cpu().numpy(), joint_limits)
azimuth = 0
elevation = 0
distance = 2
image, mujoco_extrinsic_matrix, info = simulate_mujoco_scene(unnormalized_joint_pose, azimuth, elevation, distance, model, data, renderer, lookat=[0, 0, 0], max_seg_id = 80)

import shutil
out_path = 'out_0'
if os.path.exists(out_path):
    shutil.rmtree(out_path)
os.makedirs(out_path)

display_render(image)
plt.savefig(os.path.join(out_path, 'mujoco_render.png'))
#render gaussians from the same angle and distance
example_camera_mujoco = Camera_Pose(torch.tensor(mujoco_extrinsic_matrix).float().cuda(), example_camera.FoVx, example_camera.FoVy,\
                            224, 224, example_camera.trans, example_camera.scale, joint_pose=example_camera.joint_pose).cuda()

frame = torch.clamp(render(example_camera_mujoco, gaussians, background_color)['render'], 0, 1)
print(frame.shape)
display_render(frame)
plt.savefig(os.path.join(out_path, 'gaussian_render.png'))
