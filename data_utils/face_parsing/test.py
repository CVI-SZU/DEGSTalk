#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
from model import BiSeNet

import torch

import os
import os.path as osp

from PIL import Image
import torchvision.transforms as transforms
import cv2
from pathlib import Path
import configargparse
import tqdm

# import ttach as tta

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg',
                     img_size=(512, 512)):
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros(
        (vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + np.array([255, 255, 255])  # + 255 默认全是白色
    # BRG格式
    num_of_class = np.max(vis_parsing_anno)
    # print(num_of_class)
    for pi in range(1, 14):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0]) # 蓝色

    for pi in [11]:
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([100, 100, 100]) # 嘴唇 灰色

    for pi in range(14, 16):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 255, 0]) # 红色
    for pi in range(16, 17):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 255]) # 绿色
    for pi in (7,8,9,17, 18):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([0, 0, 0]) # 黑色 头发部分
    for pi in range(18, num_of_class+1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = np.array([255, 0, 0]) # 蓝色

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    index = np.where(vis_parsing_anno == num_of_class-1)
    vis_im = cv2.resize(vis_parsing_anno_color, img_size,
                        interpolation=cv2.INTER_NEAREST)
    if save_im:
        cv2.imwrite(save_path, vis_im)


def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    Path(respth).mkdir(parents=True, exist_ok=True)

    print(f'[INFO] loading model...')
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load(cp))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    image_paths = sorted(os.listdir(dspth))

    with torch.no_grad():
        for image_path in tqdm.tqdm(image_paths):
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                parsing_image_path = int(image_path[:-4])
                parsing_image_path = str(parsing_image_path) + '.png'
                parsing_image_path = osp.join(respth, parsing_image_path)
                if os.path.exists(parsing_image_path):
                    continue
                img = Image.open(osp.join(dspth, image_path))
                ori_size = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                image = image.convert("RGB")
                img = to_tensor(image)

                # test-time augmentation.
                inputs = torch.unsqueeze(img, 0) # [1, 3, 512, 512]
                outputs = net(inputs.cuda())
                parsing = outputs.mean(0).cpu().numpy().argmax(0)

                vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=parsing_image_path, img_size=ori_size)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--respath', type=str, default='./result/', help='result path for label')
    parser.add_argument('--imgpath', type=str, default='./imgs/', help='path for input images')
    parser.add_argument('--modelpath', type=str, default='data_utils/face_parsing/79999_iter.pth')
    args = parser.parse_args()
    evaluate(respth=args.respath, dspth=args.imgpath, cp=args.modelpath)
