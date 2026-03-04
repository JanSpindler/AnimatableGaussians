import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import importlib
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance

import config
import utils.net_util as net_util
from utils.net_util import to_cuda
from utils.eval_utils import eval_images
from dataset.dataset_mv_rgb import MvRgbDatasetTHuman4
from network.avatar import AvatarNet


def load_img_mask(data_dir: str, view_idx: int, pose_idx: int):
    img_path = data_dir + '/images/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(img_path):
        img_path = data_dir + '/images/cam%02d/%08d.png' % (view_idx, pose_idx)

    mask_path = data_dir + '/masks/cam%02d/%08d.jpg' % (view_idx, pose_idx)
    if not os.path.exists(mask_path):
        mask_path = data_dir + '/masks/cam%02d/%08d.png' % (view_idx, pose_idx)

    color_img = cv.imread(img_path, cv.IMREAD_UNCHANGED)
    mask_img = cv.imread(mask_path, cv.IMREAD_UNCHANGED)

    if mask_img is not None and len(mask_img.shape) == 3:
        if mask_img.shape[2] == 2:
            mask_img = mask_img[:, :, 1]
        elif mask_img.shape[2] == 4:
            mask_img = mask_img[:, :, 3]
        else:
            mask_img = cv.cvtColor(mask_img, cv.COLOR_BGR2GRAY)

    return color_img, mask_img


@torch.no_grad()
def test(test_run, visualize=False):
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    bg_color = (0., 0., 0.)
    device = "cuda"

    # Adjust global config
    config.opt["train"]["data"] = {
        "data_dir": data_path,
    }

    # Eval output path
    eval_name = (
        f'eval_{subject_name}'
        f'_frames{start_frame}_{end_frame}'
        f'_views{"_".join(map(str, views))}.txt'
    )
    eval_path = os.path.join(data_path, eval_name)

    # Load network using the same pattern as main_avatar.py
    avatar_net = AvatarNet(config.opt['model']).to(device)
    avatar_net.eval()

    print('Loading network from ', ckpt_path)
    net_dict = torch.load(ckpt_path, map_location=device)
    if 'avatar_net' in net_dict:
        avatar_net.load_state_dict(net_dict['avatar_net'])
    else:
        raise KeyError('Cannot find "avatar_net" in checkpoint: %s' % ckpt_path)
    iter_idx = net_dict.get('iter_idx', 0)
    print('Loaded checkpoint at iter %d' % iter_idx)

    # Load dataset using the same pattern as main_avatar.py
    dataset = MvRgbDatasetTHuman4(
        data_dir=data_path, 
        frame_range=[start_frame, end_frame],
        used_cam_ids=views, 
        training=False,
        subject_name=subject_name,
        load_smpl_pos_map=False)
    print('Initialized dataset with %d items.' % len(dataset))

    # Clear eval file
    if os.path.exists(eval_path):
        open(eval_path, 'w').close()

    all_metrics = []
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    for cam_id in views:
        intr = dataset.intr_mats[cam_id].copy()
        extr = dataset.extr_mats[cam_id].copy()
        img_w = dataset.img_widths[cam_id]
        img_h = dataset.img_heights[cam_id]

        for frame_idx in tqdm(range(start_frame, end_frame), desc='Rendering cam %d' % cam_id):
            dataset_idx = frame_idx - start_frame

            # Load reference image and mask
            ref_color_img, mask_img = load_img_mask(data_path, cam_id, frame_idx)
            if ref_color_img is None or mask_img is None:
                print('Warning: Missing image or mask for cam %d, frame %d. Skipping.' % (cam_id, frame_idx))
                continue

            # Get dataset item and move to device (matching main_avatar.py mini_test pattern)
            getitem_func = dataset.getitem_fast if hasattr(dataset, 'getitem_fast') else dataset.getitem
            item = getitem_func(
                dataset_idx,
                training=False,
                extr=extr,
                intr=intr,
                img_w=img_w,
                img_h=img_h,
            )
            items = to_cuda(item, add_batch=False)

            # Render (matching main_avatar.py test/mini_test pattern)
            if 'smpl_pos_map' not in items:
                avatar_net.get_pose_map(items)

            output = avatar_net.render(items, bg_color=bg_color)

            rgb_map = output['rgb_map']           # (H, W, 3)
            rgb_map.clip_(0., 1.)

            # Build masked ground truth
            gt_img = torch.from_numpy(ref_color_img[:, :, :3].astype(np.float32) / 255.0).to(device)                   # (H, W, 3), BGR
            gt_mask = torch.from_numpy((mask_img > 0).astype(np.float32)).to(device).unsqueeze(-1)     # (H, W, 1)
            gt_img_masked = gt_img * gt_mask

            # Compute per-frame metrics (add batch dim for eval_images)
            frame_metrics = eval_images(rgb_map.unsqueeze(0), gt_img_masked.unsqueeze(0), fid)
            all_metrics.append(frame_metrics)

            with open(eval_path, 'a') as f:
                f.write('cam %d frame %d: %s\n' % (cam_id, frame_idx, frame_metrics))

            if visualize:
                print('cam %d frame %d: %s' % (cam_id, frame_idx, frame_metrics))
                vis = np.concatenate([
                    (rgb_map.cpu().numpy() * 255).astype(np.uint8),
                    (gt_img_masked.cpu().numpy() * 255).astype(np.uint8),
                ], axis=1)
                cv.imshow('rendered | gt', vis)
                cv.waitKey(1)

            torch.cuda.empty_cache()

    if not all_metrics:
        print('No frames were evaluated.')
        return

    # Aggregate metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    fid_score = fid.compute().item()

    print('Average metrics across all frames and views:')
    for key, value in avg_metrics.items():
        print('  %s: %f' % (key, value))
    print('Final FID score: %f' % fid_score)

    with open(eval_path, 'a') as f:
        f.write('Average metrics: %s\n' % avg_metrics)
        f.write('Final FID score: %f\n' % fid_score)


tests = [
    # subject00_julian
    {
        "subject_name": "subject00_julian",
        "ckpt_path": "./results/subject00_julian/avatar/batch_500000/net.pt",
        "data_path": "./thuman/subject00",
        "start_frame": 2000,
        "end_frame": 2500,
        "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    },
    {
        "subject_name": "subject00_julian",
        "ckpt_path": "./results/subject00_julian/avatar/batch_500000/net.pt",
        "data_path": "./thuman/subject00",
        "start_frame": 0,
        "end_frame": 2000,
        "views": [23],
    },
    # 0165_08
    {
        "subject_name": "0165_08",
        "ckpt_path": "./results/0165_08/avatar/batch_500000/net.pt",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 180,
        "end_frame": 225,
        "views": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    },
    {
        "subject_name": "0165_08",
        "ckpt_path": "./results/0165_08/avatar/batch_500000/net.pt",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 0,
        "end_frame": 180,
        "views": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
    }
]


if __name__ == '__main__':
    config.opt = {
        'mode': 'test',
        'train': {
            'dataset': 'MvRgbDatasetTHuman4',
            'data': {
                # 'subject_name': 'subject00_julian',
                # 'data_dir': './thuman/subject00',
                # 'frame_range': [0, 2000, 1],
                # 'used_cam_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
                # 'load_smpl_pos_map': True,
            },
            # 'pretrained_dir': None,
            # 'net_ckpt_dir': './results/subject00_julian/avatar',
            # 'prev_ckpt': None,
            # 'ckpt_interval': {
            #     'epoch': 1,
            #     'batch': 50000,
            # },
            # 'eval_interval': 1000,
            # 'eval_training_ids': [310, 19],
            # 'eval_testing_ids': [354, 1],
            # 'eval_img_factor': 1.0,
            # 'lr': {
            #     'network': {
            #         'type': 'Constant',
            #         'value': 0.0005,
            #     },
            # },
            # 'loss_weight': {
            #     'l1': 1.0,
            #     'lpips': 0.1,
            #     'offset': 0.005,
            # },
            # 'finetune_color': False,
            # 'batch_size': 1,
            # 'num_workers': 8,
            # 'random_bg_color': True,
        },
        'test': {
            'dataset': 'MvRgbDatasetTHuman4',
            # 'data': {
            #     'data_dir': './thuman/subject00',
            #     'frame_range': [0, 2000, 1],
            #     'subject_name': 'subject00_julian',
            # },
            # 'pose_data': {
            #     'data_path': './thuman/subject00/pose_00.npz',
            #     'frame_range': [2000, 2500],
            #     'hand_pose_type': 'fist',
            # },
            'view_setting': 'camera',
            'render_view_idx': 23,
            'global_orient': True,
            'img_scale': 1.0,
            'save_mesh': False,
            'render_skeleton': False,
            'save_tex_map': False,
            'save_ply': False,
            'n_pca': 20,
            'sigma_pca': 2.0,
            # 'prev_ckpt': './results/subject00_julian/avatar/batch_500000',
        },
        'model': {
            'with_viewdirs': True,
            'random_style': False,
            'multires': 6,
            'multires_viewdir': 3,
            'use_viewdir': False,
            'with_hand': True,
            'volume_type': 'diff',
            'use_root_finding': True,
        },
    }

    # for test_run in tests:
    #     test(test_run, True)
    test(tests[1], visualize=False)
