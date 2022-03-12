import os
import torch
import numpy as np
import json

import utils.general as utils
from utils import rend_util


class SceneCollectionDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects from a collection of different objects,
     where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 collection_id=0,
                 cam_file=None
                 ):

        self.instance_path = utils.ROOT_PATH / 'data' / data_dir / 'collection{0}'.format(collection_id)
        assert self.instance_path.exists(), "Data directory is empty"
        with open(str(self.instance_path / "specs.json"), "r") as f:
            _collection_specs = json.load(f)
        self.n_objs = _collection_specs["n_objs"]

        self.img_res = img_res
        self.total_pixels = img_res[0] * img_res[1]

        self.sampling_idx = None
        self.train_cameras = train_cameras

        self.n_images = 0
        self.cam_paths = []
        self.obj_idxs = []

        self.intrinsics_all = []
        self.pose_all = []
        self.rgb_images = []
        self.object_masks = []

        for i in range(self.n_objs):
            scene_dir = self.instance_path / "scan{0}".format(i)

            image_dir = str(scene_dir / 'image')
            image_paths = sorted(utils.glob_imgs(image_dir))
            mask_dir = str(scene_dir / 'mask')
            mask_paths = sorted(utils.glob_imgs(mask_dir))

            n_imgs = len(image_paths)
            self.obj_idxs += [i] * n_imgs

            cam_path = os.path.join(scene_dir, "cameras.npz")
            if cam_file is not None:
                cam_path = os.path.join(scene_dir, cam_file)
            self.cam_paths.append(cam_path)

            camera_dict = np.load(cam_path)
            scale_mats = [camera_dict['scale_mat_%d' % j].astype(np.float32) for j in range(n_imgs)]
            world_mats = [camera_dict['world_mat_%d' % j].astype(np.float32) for j in range(n_imgs)]
            self.n_images += n_imgs


            for scale_mat, world_mat in zip(scale_mats, world_mats):
                P = world_mat @ scale_mat
                P = P[:3, :4]
                intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
                self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
                self.pose_all.append(torch.from_numpy(pose).float())

            for path in image_paths:
                rgb = rend_util.load_rgb(path)
                rgb = rgb.reshape(3, -1).transpose(1, 0)
                self.rgb_images.append(torch.from_numpy(rgb).float())

            for path in mask_paths:
                object_mask = rend_util.load_mask(path)
                object_mask = object_mask.reshape(-1)
                self.object_masks.append(torch.from_numpy(object_mask).bool())
        self.obj_idxs = torch.tensor(self.obj_idxs)

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
            "obj": self.obj_idxs[idx]
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = str(self.instance_path /'cameras_linear_init.npz')
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat
