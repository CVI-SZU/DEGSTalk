import torch
import cv2
import numpy as np
from skimage.transform import estimate_transform, warp
from src.smirk_encoder import SmirkEncoder
from src.renderer.renderer import Renderer
import argparse
import os
from utils.mediapipe_utils import run_mediapipe
import torch.nn.functional as F
from tqdm import tqdm

def crop_face(frame, landmarks, scale=1.0, image_size=224):
    left = np.min(landmarks[:, 0])
    right = np.max(landmarks[:, 0])
    top = np.min(landmarks[:, 1])
    bottom = np.max(landmarks[:, 1])

    h, w, _ = frame.shape
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])

    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)

    return tform

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', type=str, default='samples/dafoe.mp4', help='Path to the input image/video')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on')
    parser.add_argument('--checkpoint', type=str, default='data_utils/smirk/pretrained_models/SMIRK_em1.pt', help='Path to the checkpoint')
    parser.add_argument('--crop', action='store_true', help='Crop the face using mediapipe')
    # parser.add_argument('--out_path', type=str, default='output', help='Path to save the output (will be created if not exists)')
    args = parser.parse_args()
    # args.crop = True # 默认为crop
    input_image_size = 224
    

    # ----------------------- initialize configuration ----------------------- #
    smirk_encoder = SmirkEncoder().to(args.device)
    checkpoint = torch.load(args.checkpoint)
    checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k} # checkpoint includes both smirk_encoder and smirk_generator

    smirk_encoder.load_state_dict(checkpoint_encoder)
    smirk_encoder.eval()

    # ---- visualize the results ---- #
    renderer = Renderer().to(args.device)
    cap = cv2.VideoCapture(args.input_path)

    if not cap.isOpened():
        print('Error opening video file')
        exit()

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video parameters:')
    print(f'FPS: {video_fps}, Width: {video_width}, Height: {video_height}， Frame Count: {frame_count}')


    video_outputs = {
        'pose_params': [],
        'cam': [],
        'shape_params': [],
        'expression_params': [],
        'eyelid_params': [],
        'jaw_params': []
    }
    
    # while True:
    #     ret, image = cap.read()

    #     if not ret:
    #         break
    
    #     kpt_mediapipe = run_mediapipe(image)

    #     # crop face if needed
    #     if args.crop:
    #         if (kpt_mediapipe is None):
    #             print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
    #             exit()
            
    #         kpt_mediapipe = kpt_mediapipe[..., :2]

    #         tform = crop_face(image,kpt_mediapipe,scale=1.4,image_size=input_image_size)
            
    #         cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

    #         cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0],1])]).T).T
    #         cropped_kpt_mediapipe = cropped_kpt_mediapipe[:,:2]
    #     else:
    #         cropped_image = image
    #         cropped_kpt_mediapipe = kpt_mediapipe

        
    #     cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    #     cropped_image = cv2.resize(cropped_image, (224,224))
    #     cropped_image = torch.tensor(cropped_image).permute(2,0,1).unsqueeze(0).float()/255.0
    #     cropped_image = cropped_image.to(args.device)

    #     outputs = smirk_encoder(cropped_image)

    #     for item, key in outputs.items():
    #         # print(item, key.shape)
    #         video_outputs[item].append(key.detach()[0].cpu())


    # cap.release()
    # print("Video parameters saved. Tensor shapes:")
    # for item, key in video_outputs.items():
    #     video_outputs[item] = torch.stack(key)
    #     print(f'{item}: {video_outputs[item].shape}')

    # torch.save(video_outputs, os.path.join(args.out_path, f"{args.input_path.split('/')[-1].split('.')[0]}_outputs.pt"))
    # # print(os.path.join(args.out_path, f"{args.input_path.split('/')[-1].split('.')[0]}_outputs.pt"))
    for _ in tqdm(range(frame_count), desc="Processing frames"):
        ret, image = cap.read()

        if not ret:
            break

        kpt_mediapipe = run_mediapipe(image)

        # crop face if needed
        if args.crop:
            if kpt_mediapipe is None:
                print('Could not find landmarks for the image using mediapipe and cannot crop the face. Exiting...')
                exit()

            kpt_mediapipe = kpt_mediapipe[..., :2]
            tform = crop_face(image, kpt_mediapipe, scale=1.4, image_size=input_image_size)
            cropped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(np.uint8)

            cropped_kpt_mediapipe = np.dot(tform.params, np.hstack([kpt_mediapipe, np.ones([kpt_mediapipe.shape[0], 1])]).T).T
            cropped_kpt_mediapipe = cropped_kpt_mediapipe[:, :2]
        else:
            cropped_image = image
            cropped_kpt_mediapipe = kpt_mediapipe

        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        cropped_image = cv2.resize(cropped_image, (224, 224))
        cropped_image = torch.tensor(cropped_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cropped_image = cropped_image.to(args.device)

        outputs = smirk_encoder(cropped_image)

        for item, key in outputs.items():
            video_outputs[item].append(key.detach()[0].cpu())

    cap.release()
    print("Video parameters saved. Tensor shapes:")
    for item, key in video_outputs.items():
        video_outputs[item] = torch.stack(key)
        print(f'{item}: {video_outputs[item].shape}')

    torch.save(video_outputs, os.path.join(os.path.dirname(args.input_path), f"3dmm.pt"))