import os
import glob
import tqdm
import json
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import face_alignment
from face_tracking.util import euler2rot
import subprocess
from PIL import Image
from scipy.ndimage import minimum_filter, maximum_filter

# export MKL_SERVICE_FORCE_INTEL=1
# export OMP_NUM_THREADS=8
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'

# Set OpenFace PATH
# openface_path = '~/3DTFG/OpenFace/build/bin/FeatureExtraction'
# 从环境变量中得到，否则为空
openface_path = os.environ.get('OPENFACE_PATH') or ""

def get_duration(file_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
        stdout=subprocess.PIPE,
        text=True
    )
    duration = float(result.stdout)
    return duration

def extract_audio(path, out_path, sample_rate=16000):
    
    print(f'[INFO] ===== extract audio from {path} to {out_path} =====')
    cmd = f'ffmpeg -y -i {path} -f wav -ar {sample_rate} {out_path}'
    os.system(cmd)
    
    total_duration = get_duration(path)
    train_duration = total_duration * 10 / 11
    test_duration = total_duration / 11
    
    directory = os.path.dirname(out_path)
    train_path = os.path.join(directory, 'aud_train.wav')
    test_path = os.path.join(directory, 'aud_novel.wav')
    
    train_cmd = f'ffmpeg -y -i "{path}" -f wav -ar {sample_rate} -t {train_duration} {train_path}'
    os.system(train_cmd)

    start_time = total_duration - test_duration
    test_cmd = f'ffmpeg -y -i "{path}" -f wav -ar {sample_rate} -ss {start_time} -t {test_duration} {test_path}'
    os.system(test_cmd)
    
    print(f'[INFO] ===== extracted audio =====')


def extract_audio_features(path, mode='ave'):

    print(f'[INFO] ===== extract audio labels for {path} =====')
    if mode == 'ave':
        print(f'AVE has been integrated into the training code, no need to extract audio features')
    elif mode == "deepspeech": # deepspeech save to data/<name>_ds.npy
        if os.path.exists(path[:-4] + '_ds.npy'):
            print(f'[INFO] ===== ds_features.npy already exists, skip extracting audio features =====')
            return
        cmd = f'python data_utils/deepspeech_features/extract_ds_features.py --input {path}'
        os.system(cmd)
    elif mode == 'hubert':
        if os.path.exists(path[:-4] +'_hu.npy'):
            print(f'[INFO] ===== hubert_features.npy already exists, skip extracting audio features =====')
            return
        cmd = f'python data_utils/hubert.py --wav {path}' # save to data/<name>_hu.npy
        os.system(cmd)
    print(f'[INFO] ===== extracted audio labels =====')


def extract_images(path, out_path, fps=25):

    print(f'[INFO] ===== extract images from {path} to {out_path} =====')
    cmd = f'ffmpeg -i {path} -vf fps={fps} -qmin 1 -q:v 1 -start_number 0 {os.path.join(out_path, "%d.jpg")}'
    os.system(cmd)
    print(f'[INFO] ===== extracted images =====')


def extract_semantics(ori_imgs_dir, parsing_dir):

    print(f'[INFO] ===== extract semantics from {ori_imgs_dir} to {parsing_dir} =====')
    cmd = f'python data_utils/face_parsing/test.py --respath={parsing_dir} --imgpath={ori_imgs_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted semantics =====')


def extract_teethmask(path_dir):
    print(f'[INFO] ===== extract teeth mask from {path_dir}=====')
    cmd = f'PYTHONPATH=./data_utils/easyportrait  python data_utils/easyportrait/create_teeth_mask.py {path_dir}'
    os.system(cmd)
    print(f'[INFO] ===== extracted teeth mask =====')

def extract_landmarks(ori_imgs_dir):

    print(f'[INFO] ===== extract face landmarks from {ori_imgs_dir} =====')
    try:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    except:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    for image_path in tqdm.tqdm(image_paths):
        lmk_path = image_path.replace('jpg', 'lms')
        if os.path.exists(lmk_path):
            continue
        input = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        preds = fa.get_landmarks(input)
        if len(preds) > 0:
            lands = preds[0].reshape(-1, 2)[:,:2]
            np.savetxt(lmk_path, lands, '%f')
    del fa
    print(f'[INFO] ===== extracted face landmarks =====')


def extract_background(base_dir, ori_imgs_dir):
    
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} =====')

    from sklearn.neighbors import NearestNeighbors

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    # only use 1/20 image_paths 
    image_paths = image_paths[::20]
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    # nearest neighbors
    all_xys = np.mgrid[0:h, 0:w].reshape(2, -1).transpose()
    distss = []
    for image_path in tqdm.tqdm(image_paths):
        parse_img = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        bg = (parse_img[..., 0] == 255) & (parse_img[..., 1] == 255) & (parse_img[..., 2] == 255)
        fg_xys = np.stack(np.nonzero(~bg)).transpose(1, 0)
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
        dists, _ = nbrs.kneighbors(all_xys)
        distss.append(dists)

    distss = np.stack(distss)
    max_dist = np.max(distss, 0)
    max_id = np.argmax(distss, 0)

    bc_pixs = max_dist > 5
    bc_pixs_id = np.nonzero(bc_pixs)
    bc_ids = max_id[bc_pixs]

    imgs = []
    num_pixs = distss.shape[1]
    for image_path in image_paths:
        img = cv2.imread(image_path)
        imgs.append(img)
    imgs = np.stack(imgs).reshape(-1, num_pixs, 3)

    bc_img = np.zeros((h*w, 3), dtype=np.uint8)
    bc_img[bc_pixs_id, :] = imgs[bc_ids, bc_pixs_id, :]
    bc_img = bc_img.reshape(h, w, 3)

    max_dist = max_dist.reshape(h, w)
    bc_pixs = max_dist > 5
    bg_xys = np.stack(np.nonzero(~bc_pixs)).transpose()
    fg_xys = np.stack(np.nonzero(bc_pixs)).transpose()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fg_xys)
    distances, indices = nbrs.kneighbors(bg_xys)
    bg_fg_xys = fg_xys[indices[:, 0]]
    bc_img[bg_xys[:, 0], bg_xys[:, 1], :] = bc_img[bg_fg_xys[:, 0], bg_fg_xys[:, 1], :]

    cv2.imwrite(os.path.join(base_dir, 'bc.jpg'), bc_img)

    print(f'[INFO] ===== extracted background image =====')

def morph_fn(bin_img, ksize=13, operation='erode'):
    """
    对图像进行膨胀或腐蚀操作。

    参数:
    - bin_img: 二值图像 (numpy 数组)
    - ksize: 内核大小 (默认为13)
    - operation: 操作类型, 'dilate' 为膨胀, 'erode' 为腐蚀

    返回:
    - 处理后的图像
    """
    pad = (ksize - 1) // 2
    padded_img = np.pad(bin_img, pad_width=pad, mode='reflect')
    
    if operation == 'dilate':
        result_img = maximum_filter(padded_img, size=ksize, mode='constant')
    elif operation == 'erode':
        result_img = minimum_filter(padded_img, size=ksize, mode='constant')
    else:
        raise ValueError(f"Unsupported operation: {operation}")
    
    return result_img[pad:-pad, pad:-pad]

def extract_background_w_hair(bash_dir, ori_imgs_dir):
    print(f'[INFO] ===== extract background image from {ori_imgs_dir} with hair =====')
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    os.makedirs(os.path.join(bash_dir, 'bg'), exist_ok=True)

    
    for image_path in tqdm.tqdm(image_paths):
        bg_img = np.array(Image.open(os.path.join(bash_dir, 'bc.jpg')).convert("RGB"))
        bg_path = image_path.replace('ori_imgs', 'bg').replace('.jpg', '.png')
        if os.path.exists(bg_path):
            continue
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        
        # read semantics
        teeth_mask_path = image_path.replace('ori_imgs', 'teeth_mask').replace('.jpg', '.npy')
        teeth_mask = np.load(teeth_mask_path)

        mask_path = image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png')
        mask = np.array(Image.open(mask_path).convert("RGB")) * 1.0

        # face_mask实际上保留了mouth_mask，所以在融合的时候更好
        # face_mask = (mask[:, :, 2] > 254) * (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) ^ teeth_mask
        # hair_mask = (mask[:, :, 0] < 1) * (mask[:, :, 1] < 1) * (mask[:, :, 2] < 1)
        mouth_mask = (mask[:, :, 0] == 100) * (mask[:, :, 1] == 100) * (mask[:, :, 2] == 100) + teeth_mask
        neck_mask = (mask[:, :, 0] == 0) * (mask[:, :, 1] == 255) * (mask[:, :, 2] == 0)
        cloth_mask = (mask[:, :, 0] == 255) * (mask[:, :, 1] == 0) * (mask[:, :, 2] == 0)
        face_only_mask = (mask[:, :, 0] == 0) * (mask[:, :, 1] == 0) * (mask[:, :, 2] == 255)

        # 进行腐蚀得到 去除parsing中除人脸的部分的数据
        face_region = face_only_mask + mouth_mask
        bg = image * ~( morph_fn(face_region, operation='erode') + mouth_mask + neck_mask + cloth_mask)[:,:,np.newaxis]
        torso_img_path = image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png')
        torso_img = np.array(Image.open(torso_img_path).convert("RGBA")) * 1.0
        bg = torso_img[..., :3] * torso_img[..., 3:] / 255.0 + bg * (1 - torso_img[..., 3:] / 255.0)
        # bg中为0,0,0的部分变为bg_image
        bg_mask = (bg[:,:, 0]==0) * (bg[:,:, 1] ==0) * (bg[:,:, 2]==0)
        bg[bg_mask] = bg_img[bg_mask]
        bg = bg.astype(np.uint8)
        Image.fromarray(bg).save(bg_path)
    print(f'[INFO] ===== extracted background image =====')

def extract_torso_and_gt(base_dir, ori_imgs_dir):

    print(f'[INFO] ===== extract torso and gt images for {base_dir} =====')

    from scipy.ndimage import binary_erosion, binary_dilation

    # load bg
    bg_image = cv2.imread(os.path.join(base_dir, 'bc.jpg'), cv2.IMREAD_UNCHANGED)
    
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))

    for image_path in tqdm.tqdm(image_paths):
        torso_image_path = image_path.replace('ori_imgs', 'torso_imgs').replace('.jpg', '.png')
        if os.path.exists(torso_image_path):
            continue
        # read ori image
        ori_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED) # [H, W, 3]

        # read semantics
        seg = cv2.imread(image_path.replace('ori_imgs', 'parsing').replace('.jpg', '.png'))
        mask_img = np.zeros_like(seg)
        head_part = (seg[..., 0] == 255) & (seg[..., 1] == 0) & (seg[..., 2] == 0)
        neck_part = (seg[..., 0] == 0) & (seg[..., 1] == 255) & (seg[..., 2] == 0)
        torso_part = (seg[..., 0] == 0) & (seg[..., 1] == 0) & (seg[..., 2] == 255)
        bg_part = (seg[..., 0] == 255) & (seg[..., 1] == 255) & (seg[..., 2] == 255)
        mask_img[head_part, :] = 255
        cv2.imwrite(image_path.replace('ori_imgs', 'face_mask').replace('.jpg', '.png'), mask_img)
        # get gt image
        gt_image = ori_image.copy()
        gt_image[bg_part] = bg_image[bg_part]
        cv2.imwrite(image_path.replace('ori_imgs', 'gt_imgs'), gt_image)

        # get torso image
        torso_image = gt_image.copy() # rgb
        torso_image[head_part] = bg_image[head_part]
        torso_alpha = 255 * np.ones((gt_image.shape[0], gt_image.shape[1], 1), dtype=np.uint8) # alpha
        
        # torso part "vertical" in-painting...
        L = 8 + 1
        torso_coords = np.stack(np.nonzero(torso_part), axis=-1) # [M, 2]
        # lexsort: sort 2D coords first by y then by x, 
        # ref: https://stackoverflow.com/questions/2706605/sorting-a-2d-numpy-array-by-multiple-axes
        inds = np.lexsort((torso_coords[:, 0], torso_coords[:, 1]))
        torso_coords = torso_coords[inds]
        # choose the top pixel for each column
        u, uid, ucnt = np.unique(torso_coords[:, 1], return_index=True, return_counts=True)
        top_torso_coords = torso_coords[uid] # [m, 2]
        # only keep top-is-head pixels
        top_torso_coords_up = top_torso_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_torso_coords_up.T)] 
        if mask.any():
            top_torso_coords = top_torso_coords[mask]
            # get the color
            top_torso_colors = gt_image[tuple(top_torso_coords.T)] # [m, 3]
            # construct inpaint coords (vertically up, or minus in x)
            inpaint_torso_coords = top_torso_coords[None].repeat(L, 0) # [L, m, 2]
            inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
            inpaint_torso_coords += inpaint_offsets
            inpaint_torso_coords = inpaint_torso_coords.reshape(-1, 2) # [Lm, 2]
            inpaint_torso_colors = top_torso_colors[None].repeat(L, 0) # [L, m, 3]
            darken_scaler = 0.98 ** np.arange(L).reshape(L, 1, 1) # [L, 1, 1]
            inpaint_torso_colors = (inpaint_torso_colors * darken_scaler).reshape(-1, 3) # [Lm, 3]
            # set color
            torso_image[tuple(inpaint_torso_coords.T)] = inpaint_torso_colors

            inpaint_torso_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
            inpaint_torso_mask[tuple(inpaint_torso_coords.T)] = True
        else:
            inpaint_torso_mask = None

        push_down = 4
        L = 48 + push_down + 1

        neck_part = binary_dilation(neck_part, structure=np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool), iterations=3)

        neck_coords = np.stack(np.nonzero(neck_part), axis=-1) # [M, 2]
        inds = np.lexsort((neck_coords[:, 0], neck_coords[:, 1]))
        neck_coords = neck_coords[inds]
        u, uid, ucnt = np.unique(neck_coords[:, 1], return_index=True, return_counts=True)
        top_neck_coords = neck_coords[uid] # [m, 2]
        top_neck_coords_up = top_neck_coords.copy() - np.array([1, 0])
        mask = head_part[tuple(top_neck_coords_up.T)] 
        
        top_neck_coords = top_neck_coords[mask]
        offset_down = np.minimum(ucnt[mask] - 1, push_down)
        top_neck_coords += np.stack([offset_down, np.zeros_like(offset_down)], axis=-1)
        # get the color
        top_neck_colors = gt_image[tuple(top_neck_coords.T)] # [m, 3]

        # construct inpaint coords (vertically up, or minus in x)
        inpaint_neck_coords = top_neck_coords[None].repeat(L, 0) # [L, m, 2]
        inpaint_offsets = np.stack([-np.arange(L), np.zeros(L, dtype=np.int32)], axis=-1)[:, None] # [L, 1, 2]
        inpaint_neck_coords += inpaint_offsets
        inpaint_neck_coords = inpaint_neck_coords.reshape(-1, 2) # [Lm, 2]

        #add
        neck_avg_color = np.mean(gt_image[neck_part], axis=0)
        inpaint_neck_colors = top_neck_colors[None].repeat(L, 0)  # [L, m, 3]
        alpha_values = np.linspace(1, 0, L).reshape(L, 1, 1)  # [L, 1, 1]
        inpaint_neck_colors = inpaint_neck_colors * alpha_values + neck_avg_color * (1 - alpha_values)
        inpaint_neck_colors = inpaint_neck_colors.reshape(-1, 3)  # [Lm, 3]
        torso_image[tuple(inpaint_neck_coords.T)] = inpaint_neck_colors

        inpaint_mask = np.zeros_like(torso_image[..., 0]).astype(bool)
        inpaint_mask[tuple(inpaint_neck_coords.T)] = True

        blur_img = torso_image.copy()
        blur_img = cv2.GaussianBlur(blur_img, (5, 5), cv2.BORDER_DEFAULT)

        torso_image[inpaint_mask] = blur_img[inpaint_mask]

        # set mask
        mask = (neck_part | torso_part | inpaint_mask)
        if inpaint_torso_mask is not None:
            mask = mask | inpaint_torso_mask
        torso_image[~mask] = 0
        torso_alpha[~mask] = 0

        cv2.imwrite(torso_image_path, np.concatenate([torso_image, torso_alpha], axis=-1))
    print(f'[INFO] ===== extracted torso and gt images =====')


def face_tracking(ori_imgs_dir):

    print(f'[INFO] ===== perform face tracking =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    cmd = f'python data_utils/face_tracking/face_tracker.py --path={ori_imgs_dir} --img_h={h} --img_w={w} --frame_num={len(image_paths)}'

    os.system(cmd)

    print(f'[INFO] ===== finished face tracking =====')

# ref: https://github.com/ShunyuYao/DFA-NeRF
def extract_flow(base_dir,ori_imgs_dir,mask_dir, flow_dir):
    print(f'[INFO] ===== extract flow =====')
    torch.cuda.empty_cache()
    ref_id = 2
    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]
    valid_img_ids = []
    for i in range(100000):
        if os.path.isfile(os.path.join(ori_imgs_dir, '{:d}.lms'.format(i))):
            valid_img_ids.append(i)
    valid_img_num = len(valid_img_ids)
    with open(os.path.join(base_dir, 'flow_list.txt'), 'w') as file:
        for i in range(0, valid_img_num):
            file.write(base_dir + '/ori_imgs/' + '{:d}.jpg '.format(ref_id) +
                       base_dir + '/face_mask/' + '{:d}.png '.format(ref_id) +
                       base_dir + '/ori_imgs/' + '{:d}.jpg '.format(i) +
                       base_dir + '/face_mask/' + '{:d}.png\n'.format(i))
        file.close()
    ext_flow_cmd = 'python data_utils/UNFaceFlow/test_flow.py --datapath=' + base_dir + '/flow_list.txt ' + \
        '--savepath=' + base_dir + '/flow_result' + \
        ' --width=' + str(w) + ' --height=' + str(h)
    os.system(ext_flow_cmd)
    face_img = cv2.imread(os.path.join(ori_imgs_dir, '{:d}.jpg'.format(ref_id)))
    face_img_mask = cv2.imread(os.path.join(mask_dir, '{:d}.png'.format(ref_id)))

    rigid_mask = face_img_mask[..., 0] > 250
    rigid_num = np.sum(rigid_mask)
    flow_frame_num = 2500
    flow_frame_num = min(flow_frame_num, valid_img_num)
    rigid_flow = np.zeros((flow_frame_num, 2, rigid_num), np.float32)
    for i in range(flow_frame_num):
        flow = np.load(os.path.join(flow_dir, '{:d}_{:d}.npy'.format(ref_id, valid_img_ids[i])))
        rigid_flow[i] = flow[:, rigid_mask]
    rigid_flow = rigid_flow.transpose((2, 1, 0))
    rigid_flow = torch.as_tensor(rigid_flow).cuda()
    lap_kernel = torch.Tensor(
        (-0.5, 1.0, -0.5)).unsqueeze(0).unsqueeze(0).float().cuda()
    flow_lap = F.conv1d(
        rigid_flow.reshape(-1, 1, rigid_flow.shape[-1]), lap_kernel)
    flow_lap = flow_lap.view(rigid_flow.shape[0], 2, -1)
    flow_lap = torch.norm(flow_lap, dim=1)
    valid_frame = torch.mean(flow_lap, dim=0) < (torch.mean(flow_lap) * 3)
    flow_lap = flow_lap[:, valid_frame]
    rigid_flow_mean = torch.mean(flow_lap, dim=1)
    rigid_flow_show = (rigid_flow_mean - torch.min(rigid_flow_mean)) / \
                      (torch.max(rigid_flow_mean) - torch.min(rigid_flow_mean)) * 255
    rigid_flow_show = rigid_flow_show.byte().cpu().numpy()
    rigid_flow_img = np.zeros((h, w, 1), dtype=np.uint8)
    rigid_flow_img[...] = 255
    rigid_flow_img[rigid_mask, 0] = rigid_flow_show
    cv2.imwrite(os.path.join(base_dir, 'rigid_flow.jpg'), rigid_flow_img)
    win_size, d_size = 5, 5
    sel_xys = np.zeros((h, w), dtype=np.int32)
    xys = []
    for y in range(0, h - win_size, win_size):
        for x in range(0, w - win_size, win_size):
            min_v = int(40)
            id_x = -1
            id_y = -1
            for dy in range(0, win_size):
                for dx in range(0, win_size):
                    if rigid_flow_img[y + dy, x + dx, 0] < min_v:
                        min_v = rigid_flow_img[y + dy, x + dx, 0]
                        id_x = x + dx
                        id_y = y + dy
            if id_x >= 0:
                if (np.sum(sel_xys[id_y - d_size:id_y + d_size + 1, id_x - d_size:id_x + d_size + 1]) == 0):
                    cv2.circle(face_img, (id_x, id_y), 1, (255, 0, 0))
                    xys.append(np.array((id_x, id_y), np.int32))
                    sel_xys[id_y, id_x] = 1

    cv2.imwrite(os.path.join(base_dir, 'keypts.jpg'), face_img)
    np.savetxt(os.path.join(base_dir, 'keypoints.txt'), xys, '%d')
    key_xys = np.loadtxt(os.path.join(base_dir, 'keypoints.txt'), np.int32)
    track_xys = np.zeros((valid_img_num, key_xys.shape[0], 2), dtype=np.float32)
    track_dir = os.path.join(base_dir,'flow_result')
    track_paths = sorted(glob.glob(os.path.join(track_dir, '*.npy')), key=lambda x: int(x.replace('\\', '/').split('/')[-1].split('.')[0]))

    for i, path in enumerate(track_paths):

        flow = np.load(path)
        for j in range(key_xys.shape[0]):
            x = key_xys[j, 0]
            y = key_xys[j, 1]
            track_xys[i, j, 0] = x + flow[0, y, x]
            track_xys[i, j, 1] = y + flow[1, y, x]
    np.save(os.path.join(base_dir, 'track_xys.npy'), track_xys)

    pose_opt_cmd = 'python data_utils/face_tracking/bundle_adjustment.py --path=' + base_dir + ' --img_h=' + \
        str(h) + ' --img_w=' + str(w)
    os.system(pose_opt_cmd)
    print(f'[INFO] ===== finished flow =====')

def save_transforms(base_dir, ori_imgs_dir):
    print(f'[INFO] ===== save transforms =====')

    image_paths = glob.glob(os.path.join(ori_imgs_dir, '*.jpg'))
    
    # read one image to get H/W
    tmp_image = cv2.imread(image_paths[0], cv2.IMREAD_UNCHANGED) # [H, W, 3]
    h, w = tmp_image.shape[:2]

    params_dict = torch.load(os.path.join(base_dir, 'bundle_adjustment.pt'))
    focal_len = params_dict['focal']
    euler_angle = params_dict['euler']
    trans = params_dict['trans']
    valid_num = euler_angle.shape[0]

    train_val_split = int(valid_num * 10 / 11)
    train_ids = torch.arange(0, train_val_split)
    val_ids = torch.arange(train_val_split, valid_num)

    rot = euler2rot(euler_angle)
    rot_inv = rot.permute(0, 2, 1)
    trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

    pose = torch.eye(4, dtype=torch.float32)
    save_ids = ['train', 'val']
    train_val_ids = [train_ids, val_ids]
    mean_z = -float(torch.mean(trans[:, 2]).item())

    for split in range(2):
        transform_dict = dict()
        transform_dict['focal_len'] = float(focal_len[0])
        transform_dict['cx'] = float(w/2.0)
        transform_dict['cy'] = float(h/2.0)
        transform_dict['frames'] = []
        ids = train_val_ids[split]
        save_id = save_ids[split]

        for i in ids:
            i = i.item()
            frame_dict = dict()
            frame_dict['img_id'] = i
            frame_dict['aud_id'] = i

            pose[:3, :3] = rot_inv[i]
            pose[:3, 3] = trans_inv[i, :, 0]

            frame_dict['transform_matrix'] = pose.numpy().tolist()

            transform_dict['frames'].append(frame_dict)

        with open(os.path.join(base_dir, 'transforms_' + save_id + '.json'), 'w') as fp:
            json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

    print(f'[INFO] ===== finished saving transforms =====')

def extract_au(base_dir, path):
    if not os.path.exists(openface_path):
        print(f'[ERROR] ===== openface not found =====')
        return
    print(f'[INFO] ===== extract au =====')
    if os.path.exists(os.path.join(base_dir, 'au.csv')):
        print(f'[INFO] ===== finished saving au =====')
        return
    import subprocess, shutil
    # export OPENBLAS_NUM_THREADS=1
    # export GOTO_NUM_THREADS=1
    # export OMP_NUM_THREADS=1
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['GOTO_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    command = f'{openface_path} -f "{path}"'
    subprocess.run(command, shell=True)
    shutil.copy(f'processed/{os.path.basename(path).replace("mp4", "csv")}', 
                os.path.join(base_dir, 'au.csv'))
    print(f'[INFO] ===== finished saving au =====')

def extract_blendshape(base_dir):
    print(f'[INFO] ===== extract blendshape =====')
    blendshape_cmd = 'python data_utils/blendshape_capture/main.py --path=' + base_dir
    os.system(blendshape_cmd)
    print(f'[INFO] ===== finished saving blendshape =====')


def extract_3dmm(path):
    print(f'[INFO] ===== extract 3dmm =====')
    smirk_cmd = 'python data_utils/smirk/process_video.py --input_path=' + path
    os.system(smirk_cmd)
    print(f'[INFO] ===== finished saving 3dmm =====')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="path to video file")
    parser.add_argument('--task', type=int, default=-1, help="-1 means all")
    parser.add_argument('--asr', type=str, default='deepspeech', help="ave, hubert or deepspeech")

    opt = parser.parse_args()

    base_dir = os.path.dirname(opt.path)
    
    wav_path = os.path.join(base_dir, 'aud.wav')
    ori_imgs_dir = os.path.join(base_dir, 'ori_imgs')
    parsing_dir = os.path.join(base_dir, 'parsing')
    gt_imgs_dir = os.path.join(base_dir, 'gt_imgs')
    torso_imgs_dir = os.path.join(base_dir, 'torso_imgs')
    mask_imgs_dir = os.path.join(base_dir, 'face_mask')
    flow_dir = os.path.join(base_dir, 'flow_result')

    os.makedirs(ori_imgs_dir, exist_ok=True)
    os.makedirs(parsing_dir, exist_ok=True)
    os.makedirs(gt_imgs_dir, exist_ok=True)
    os.makedirs(torso_imgs_dir, exist_ok=True)
    os.makedirs(mask_imgs_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    # extract audio
    if opt.task == -1 or opt.task == 1:
        extract_audio(opt.path, wav_path)
        extract_audio_features(wav_path, mode=opt.asr)

        directory = os.path.dirname(wav_path)
        train_path = os.path.join(directory, 'aud_train.wav')
        test_path = os.path.join(directory, 'aud_novel.wav')
        extract_audio_features(train_path, mode=opt.asr)
        extract_audio_features(test_path, mode=opt.asr)
        
    # extract images
    if opt.task == -1 or opt.task == 2:
        extract_images(opt.path, ori_imgs_dir)


    # face parsing and teeth parsing
    if opt.task == -1 or opt.task == 3:
        # teeth parsing 
        extract_teethmask(base_dir)
        # face parsing
        extract_semantics(ori_imgs_dir, parsing_dir)

    # extract bg
    if opt.task == -1 or opt.task == 4:
        extract_background(base_dir, ori_imgs_dir)
        
    # extract torso images and gt_images
    if opt.task == -1 or opt.task == 5:
        extract_torso_and_gt(base_dir, ori_imgs_dir)

    # extract bg w hair
    if opt.task == -1 or opt.task == 6:
        extract_background_w_hair(base_dir, ori_imgs_dir)

    # extract face landmarks
    if opt.task == -1 or opt.task == 7:
        extract_landmarks(ori_imgs_dir)

    # face tracking
    if opt.task == -1 or opt.task == 8:
        face_tracking(ori_imgs_dir)

    # extract flow & pose optimization
    if opt.task == -1 or opt.task == 9:
        extract_flow(base_dir, ori_imgs_dir, mask_imgs_dir, flow_dir)

    # save transforms.json
    if opt.task == -1 or opt.task == 10:
        save_transforms(base_dir, ori_imgs_dir)

    # extract exp representation
    if opt.task == -1 or opt.task == 11:
        # extract_blendshape(base_dir) # extract blendshape
        extract_au(base_dir, opt.path) # extract au
        extract_3dmm(opt.path) # extract 3dmm