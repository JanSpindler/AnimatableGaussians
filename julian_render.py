import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm

import config
import utils.net_util as net_util
from utils.net_util import to_cuda
from dataset.dataset_mv_rgb import MvRgbDatasetTHuman4
from network.avatar import AvatarNet


@torch.no_grad()
def render(test_run):
    subject_name = test_run['subject_name']
    ckpt_path = test_run['ckpt_path']
    data_path = test_run['data_path']
    start_frame = test_run['start_frame']
    end_frame = test_run['end_frame']
    views = test_run['views']
    bg_color = (1., 1., 1.)
    out_dir = test_run.get(
        'out_dir',
        os.path.join('renders', subject_name, f'frames{start_frame}_{end_frame}')
    )
    device = "cuda"

    # Adjust global config
    config.opt["train"]["data"] = {
        "data_dir": data_path,
    }

    # Load network
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

    # Load dataset
    dataset = MvRgbDatasetTHuman4(
        data_dir=data_path,
        frame_range=[start_frame, end_frame],
        used_cam_ids=views,
        training=False,
        subject_name=subject_name,
        load_smpl_pos_map=False)
    print('Initialized dataset with %d items.' % len(dataset))

    for cam_id in views:
        intr = dataset.intr_mats[cam_id].copy()
        extr = dataset.extr_mats[cam_id].copy()
        img_w = dataset.img_widths[cam_id]
        img_h = dataset.img_heights[cam_id]

        cam_out_dir = os.path.join(out_dir, 'cam%02d' % cam_id)
        os.makedirs(cam_out_dir, exist_ok=True)

        for frame_idx in tqdm(range(start_frame, end_frame), desc='Rendering cam %d' % cam_id):
            dataset_idx = frame_idx - start_frame

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

            if 'smpl_pos_map' not in items:
                avatar_net.get_pose_map(items)

            output = avatar_net.render(items, bg_color=bg_color)

            rgb_map = output['rgb_map']
            rgb_map.clip_(0., 1.)
            rgb_map_np = (rgb_map.cpu().numpy() * 255).astype(np.uint8)

            out_path = os.path.join(cam_out_dir, '%08d.png' % frame_idx)
            cv.imwrite(out_path, rgb_map_np)

            torch.cuda.empty_cache()

    print('Saved renders to %s' % out_dir)


tests = [
    # subject00_julian
    # {
    #     "subject_name": "subject00_julian",
    #     "ckpt_path": "./results/subject00_julian/avatar/batch_500000/net.pt",
    #     "data_path": "./thuman/subject00",
    #     "start_frame": 0,
    #     "end_frame": 2500,
    #     "views": list(range(24)),
    # },
    {
        "subject_name": "subject01_julian",
        "ckpt_path": "./results/subject01_julian/avatar/batch_500000/net.pt",
        "data_path": "./thuman/subject01",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    {
        "subject_name": "subject02_julian",
        "ckpt_path": "./results/subject02_julian/avatar/batch_500000/net.pt",
        "data_path": "./thuman/subject02",
        "start_frame": 0,
        "end_frame": 2500,
        "views": list(range(24)),
    },
    # DNA Rendering
    {
        "subject_name": "0165_08",
        "ckpt_path": "./results/0165_08/avatar/batch_500000/net.pt",
        "data_path": "./dnarendering/0165_08",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    {
        "subject_name": "0166_04",
        "ckpt_path": "./results/0166_04/avatar/batch_500000/net.pt",
        "data_path": "./dnarendering/0166_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
    {
        "subject_name": "0206_04",
        "ckpt_path": "./results/0206_04/avatar/batch_500000/net.pt",
        "data_path": "./dnarendering/0206_04",
        "start_frame": 0,
        "end_frame": 225,
        "views": list(range(60)),
    },
]


if __name__ == '__main__':
    config.opt = {
        'mode': 'test',
        'train': {
            'dataset': 'MvRgbDatasetTHuman4',
            'data': {},
        },
        'test': {
            'dataset': 'MvRgbDatasetTHuman4',
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

    print('Found %d test(s). Running them sequentially...' % len(tests))
    for test_run in tests:
        render(test_run)
