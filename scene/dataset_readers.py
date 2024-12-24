#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import torch
from PIL import Image
from typing import NamedTuple
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import pandas as pd

from utils.sh_utils import SH2RGB
from utils.audio_utils import get_audio_features, AudDataset, AudioEncoder
from utils.image_utils import morph_fn
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    background: np.array
    talking_dict: dict

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def save_mask_image(mask_array, save_dir, save_name):
    """
    Save a mask array as a PNG image.

    Parameters:
    - mask_array (numpy.ndarray): The mask array (should be dtype=np.uint8 and values 0 or 255).
    - save_dir (str): Directory where the image will be saved.
    - save_name (str): Name of the saved image file.
    """
    # Convert mask array to PIL Image
    mask_image = Image.fromarray(mask_array)

    # Save the mask image
    save_path = os.path.join(save_dir, save_name)
    mask_image.save(save_path)


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".jpg", audio_file='', audio_extractor='deepspeech'):
    cam_infos = []
    postfix_dict = {"deepspeech": "ds", "hubert": "hu"}
    write_to_disk = True
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        focal_len = contents["focal_len"]
        bg_img = np.array(Image.open(os.path.join(path, 'bc.jpg')).convert("RGB"))

        frames = contents["frames"]
        
        if audio_file == '':
            if audio_extractor == 'ave':
                from torch.utils.data import DataLoader
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = AudioEncoder().to(device).eval()
                ckpt = torch.load('pretrained_models/audio_visual_encoder.pth')
                model.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
                dataset = AudDataset(os.path.join(path, 'aud.wav'))
                data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
                outputs = []
                for mel in data_loader:
                    mel = mel.to(device)
                    with torch.no_grad():
                        out = model(mel)
                    outputs.append(out)
                outputs = torch.cat(outputs, dim=0).cpu()
                first_frame, last_frame = outputs[:1], outputs[-1:]
                aud_features = torch.cat([first_frame.repeat(2, 1), outputs, last_frame.repeat(2, 1)],
                                            dim=0).numpy()
            elif audio_extractor in ['deepspeech', 'hubert']:
                aud_features = np.load(os.path.join(path, 'aud_{}.npy'.format(postfix_dict[audio_extractor])))
            else:
                raise NotImplementedError
        else:
            aud_features = np.load(audio_file)
        aud_features = torch.from_numpy(aud_features)
        aud_features = aud_features.float().permute(0, 2, 1)
        auds = aud_features
        # https://imotions.com/blog/learning/research-fundamentals/facial-action-coding-system/
        au_info=pd.read_csv(os.path.join(path, 'au.csv'))
        try:
            au_blink = au_info['AU45_r'].values
            au25 = au_info['AU25_r'].values
        except:
            print('No AU25')
        au25 = np.clip(au25, 0, np.percentile(au25, 95))
        # 嘴巴部分的统计分布
        au25_25, au25_50, au25_75, au25_100 = np.percentile(au25, 25), np.percentile(au25, 50), np.percentile(au25, 75), au25.max()

        au_exp = []
        for i in [1,4,5,6,7,45]:
            _key = 'AU' + str(i).zfill(2) + '_r'
            au_exp_t = au_info[_key].values
            if i == 45:
                au_exp_t = au_exp_t.clip(0, 2)
            au_exp.append(au_exp_t[:, None])
        au_exp = np.concatenate(au_exp, axis=-1, dtype=np.float32)

        flame_dict = torch.load(os.path.join(path, '3dmm.pt'))
        exp_params = flame_dict['expression_params']
        jaw_params = flame_dict['jaw_params']
        eyelid_params = flame_dict['eyelid_params']
        shape_params = flame_dict['shape_params']
        
        # Blendshape
        # bs = np.load(os.path.join(path, 'bs.npy'))
        # bs = np.hstack((bs[:, 0:5], bs[:, 8:10])) # 5 + 2
        # au_exp = bs
        ldmks_lips = []
        ldmks_mouth = []
        ldmks_lhalf = []
        
        for idx, frame in tqdm(enumerate(frames)):
            lms = np.loadtxt(os.path.join(path, 'ori_imgs', str(frame['img_id']) + '.lms')) # [68, 2]
            lips = slice(48, 60)
            mouth = slice(60, 68)
            xmin, xmax = int(lms[lips, 1].min()), int(lms[lips, 1].max())
            ymin, ymax = int(lms[lips, 0].min()), int(lms[lips, 0].max())

            ldmks_lips.append([int(xmin), int(xmax), int(ymin), int(ymax)])
            ldmks_mouth.append([int(lms[mouth, 1].min()), int(lms[mouth, 1].max())])

            lh_xmin, lh_xmax = int(lms[31:36, 1].min()), int(lms[:, 1].max()) # actually lower half area
            xmin, xmax = int(lms[:, 1].min()), int(lms[:, 1].max())
            ymin, ymax = int(lms[:, 0].min()), int(lms[:, 0].max())
            # self.face_rect.append([xmin, xmax, ymin, ymax])
            ldmks_lhalf.append([lh_xmin, lh_xmax, ymin, ymax])
            
        ldmks_lips = np.array(ldmks_lips)
        ldmks_mouth = np.array(ldmks_mouth)
        ldmks_lhalf = np.array(ldmks_lhalf)
        mouth_lb = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).min()
        mouth_ub = (ldmks_mouth[:, 1] - ldmks_mouth[:, 0]).max()

        print(mouth_lb, mouth_ub)

        os.makedirs(os.path.join(path, 'bg'), exist_ok=True)

        for idx, frame in tqdm(enumerate(frames)):
            cam_name = os.path.join(path, 'gt_imgs', str(frame["img_id"]) + extension)
            
            # 使用ori_imgs来做监督，不用经过parsing处理的图片
            cam_name = os.path.join(path, 'ori_imgs', str(frame["img_id"]) + extension)
            
            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            w, h = image.size[0], image.size[1]
            image = np.array(image.convert("RGB"))
            # torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            # torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
            # bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg_img * (1 - torso_img[..., 3:] / 255.0)
            # Image.fromarray(bg).save('bg.jpg')
            # bg = face_mask_img[..., :3] * face_mask_img[..., 3:] / 255.0 + bg_img * (1 - face_mask_img[..., 3:] / 255.0)
            # bg = bg.astype(np.uint8)

            talking_dict = {}
            talking_dict['img_id'] = frame['img_id']

            teeth_mask_path = os.path.join(path, 'teeth_mask', str(frame['img_id']) + '.npy')
            teeth_mask = np.load(teeth_mask_path)

            mask_path = os.path.join(path, 'parsing', str(frame['img_id']) + '.png')
            mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0
            # face_mask实际上保留了mouth_mask，所以在融合的时候更好
            talking_dict['face_mask'] = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
            talking_dict['hair_mask'] = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
            talking_dict['mouth_mask'] = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask
            talking_dict['neck_mask'] = (mask[:, :, 0] == 0) * (mask[:, :, 1] == 255) * (mask[:, :, 2] == 0)
            talking_dict['cloth_mask'] = (mask[:, :, 0] == 255) * (mask[:, :, 1] == 0) * (mask[:, :, 2] == 0)
            talking_dict['face_only_mask'] = (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) * (mask[:, :, 2] == 255)

            # face_only_mask_np = np.array(talking_dict['face_only_mask'], dtype=np.uint8) * 255
            # save_mask_image(face_only_mask_np, './', 'face_only_mask.png')
            # mouth_mask_np = np.array(talking_dict['mouth_mask'], dtype=np.uint8) * 255
            # save_mask_image(face_only_mask_np, './', 'face_only_mask.png')
            

            # # 进行腐蚀得到 去除parsing中除人脸的部分的数据
            # face_region = talking_dict['face_only_mask']+talking_dict['mouth_mask']
            # bg = image * ~( morph_fn(face_region, operation='erode') + talking_dict['mouth_mask']+talking_dict['neck_mask']+talking_dict['cloth_mask'])[:,:,np.newaxis]
            # torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
            # torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
            # bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg * (1 - torso_img[..., 3:] / 255.0)
            # # bg中为0,0,0的部分变为bg_image
            # bg_mask = (bg[:,:, 0]==0) * (bg[:,:, 1] ==0) * (bg[:,:, 2]==0)
            # bg[bg_mask] = bg_img[bg_mask]
            # bg = bg.astype(np.uint8)

            bg_path = os.path.join(path, 'bg', f'{str(frame["img_id"])}.png')
            if os.path.exists(bg_path):
                bg = np.array(Image.open(bg_path).convert("RGB"))
                bg = bg.astype(np.uint8)
            else:
                # 进行腐蚀得到 去除parsing中除人脸的部分的数据
                face_region = talking_dict['face_only_mask']+talking_dict['mouth_mask']
                bg = image * ~( morph_fn(face_region, operation='erode') + talking_dict['mouth_mask']+talking_dict['neck_mask']+talking_dict['cloth_mask'])[:,:,np.newaxis]
                torso_img_path = os.path.join(path, 'torso_imgs', str(frame['img_id']) + '.png')
                torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
                bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg * (1 - torso_img[..., 3:] / 255.0)
                # bg中为0,0,0的部分变为bg_image
                bg_mask = (bg[:,:, 0]==0) * (bg[:,:, 1] ==0) * (bg[:,:, 2]==0)
                bg[bg_mask] = bg_img[bg_mask]
                bg = bg.astype(np.uint8)
                if write_to_disk:
                    Image.fromarray(bg).save(bg_path)

            # save_mask_image(face_only_mask_np, './debug', 'face_only_mask_dilate.png')
            # Image.fromarray(bg).save('bg.png')
            # exit(0)
            
            if audio_file == '':
                talking_dict['auds'] = get_audio_features(auds, 2, frame['img_id'])
                if frame['img_id'] > auds.shape[0]:
                    print("[warnining] audio feature is too short")
                    break
            else:
                talking_dict['auds'] = get_audio_features(auds, 2, idx)
                if idx >= auds.shape[0]:
                    break


            talking_dict['blink'] = torch.as_tensor(np.clip(au_blink[frame['img_id']], 0, 2) / 2)
            talking_dict['au25'] = [au25[frame['img_id']], au25_25, au25_50, au25_75, au25_100]

            talking_dict['au_exp'] = torch.as_tensor(au_exp[frame['img_id']])
            
            talking_dict['3dmm_exp'] = exp_params[frame['img_id']] # 3dmm的exp系数
            talking_dict['3dmm_eyelid'] = eyelid_params[frame['img_id']] # 眼睛的 blendshape 系数
            talking_dict['3dmm_exp'] = torch.cat((exp_params[frame['img_id']], talking_dict['3dmm_eyelid']), 0) # exp 和 eye 进行结合 方便后续进行操作
            talking_dict['3dmm_jaw'] = jaw_params[frame['img_id']] # jaw系数 对于嘴唇有帮助
            talking_dict['3dmm_shape'] = shape_params[frame['img_id']] # shape系数，有助于对人脸建模

            [xmin, xmax, ymin, ymax] = ldmks_lips[idx].tolist()
            # padding to H == W
            cx = (xmin + xmax) // 2
            cy = (ymin + ymax) // 2

            l = max(xmax - xmin, ymax - ymin) // 2
            xmin = cx - l
            xmax = cx + l
            ymin = cy - l
            ymax = cy + l

            talking_dict['lips_rect'] = [xmin, xmax, ymin, ymax]
            talking_dict['lhalf_rect'] = ldmks_lhalf[idx]
            talking_dict['mouth_bound'] = [mouth_lb, mouth_ub, ldmks_mouth[idx, 1] - ldmks_mouth[idx, 0]]
            talking_dict['img_id'] = frame['img_id']


            # norm_data = im_data / 255.0
            # arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            # image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            FovX = focal2fov(focal_len, w)
            FovY = focal2fov(focal_len, h)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=w, height=h, background=bg, talking_dict=talking_dict))

            # if idx > 200: break
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".jpg", args=None):
    audio_file = args.audio
    audio_extractor = args.audio_extractor
    if not eval:
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, audio_file, audio_extractor)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_val.json", white_background, extension, audio_file, audio_extractor)
    
    # if not eval:
    #     train_cam_infos.extend(test_cam_infos)
    #     test_cam_infos = []
    if eval:
        train_cam_infos = test_cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos) 


    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path) or True:
        # Since this data set has no colmap data, we start with random points
        num_pts = args.init_num
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 0.2 - 0.1
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": None,
    "Blender" : readNerfSyntheticInfo
}
