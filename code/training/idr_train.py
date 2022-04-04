import math
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from pyhocon import ConfigFactory
import sys
import torch
import pickle
import utils.general as utils
import utils.plots as plt
import tqdm
from torch.utils.tensorboard import SummaryWriter

from model.implicit_differentiable_renderer import IDRNetwork
from model.loss import IDRLoss
from training.exp_storage import ExpStorage


class IDRTrainRunner:
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = kwargs['exps_folder_name']
        self.GPU_INDEX = kwargs['gpu_index']
        self.train_cameras = kwargs['train_cameras']

        self.expname = self.conf.get_string('train.expname') + kwargs['expname']
        collection = kwargs['collection_id'] if kwargs['collection_id'] != -1 else self.conf.get_int(
            'dataset.collection_id', default=-1)
        if collection != -1:
            self.expname = self.expname + '_{0}'.format(collection)

        self._is_continue = False

        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join('../', kwargs['exps_folder_name'], self.expname)):
                timestamps = os.listdir(os.path.join('../', kwargs['exps_folder_name'], self.expname))
                if (len(timestamps)) == 0:
                    self._is_continue = False
                    timestamp = None
                else:
                    timestamp = sorted(timestamps)[-1]
                    self._is_continue = True
            else:
                self._is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            self._is_continue = kwargs['is_continue']

        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)

        # create checkpoints dirs
        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))

        storage_path = Path(os.path.join(self.expdir, self.timestamp, "storage.pickle"))
        if self._is_continue:
            old_storage_path = Path(os.path.join(self.expdir, timestamp, "storage.pickle"))
            self.storage = ExpStorage.load(old_storage_path)
            self.storage.change_path(storage_path)
        else:
            self.storage = ExpStorage(storage_path)

        if self.train_cameras:
            self.optimizer_cam_params_subdir = "OptimizerCamParameters"
            self.cam_params_subdir = "CamParameters"

            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.cam_params_subdir))

        os.system(
            """cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.expdir, self.timestamp, 'runconf.conf')))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)

        print('shell command : {0}'.format(' '.join(sys.argv)))

        print('Loading data ...')

        dataset_conf = self.conf.get_config('dataset')
        if kwargs['collection_id'] != -1:
            dataset_conf['collection_id'] = kwargs['collection_id']

        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras,
                                                                                          **dataset_conf)
        self.num_datapoints = -1
        self.n_objs = self.train_dataset.n_objs

        self.lat_size = self.conf.get_int('model.latent_vector_size')

        self.lat_vecs = torch.nn.Embedding(self.n_objs, self.lat_size)
        torch.nn.init.normal_(
            self.lat_vecs.weight.data,
            0.0,
            1 / math.sqrt(self.lat_size),
        )

        print('Finish loading data ...')

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                            batch_size=self.batch_size,
                                                            shuffle=True,
                                                            collate_fn=self.train_dataset.collate_fn
                                                            )
        self.plot_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.conf.get_int('plot.plot_nimgs'),
                                                           shuffle=True,
                                                           collate_fn=self.train_dataset.collate_fn
                                                           )

        self.model = IDRNetwork(conf=self.conf.get_config('model'))

        if torch.cuda.is_available():
            self.model.cuda()

        self.loss = IDRLoss(self.model, **self.conf.get_config('loss'))
        self.optimization_steps = 0

        self.lr = self.conf.get_float('train.learning_rate')

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": self.lr
                },
                {
                    "params": self.lat_vecs.parameters(),
                    "lr": self.lr
                },
            ]
        )

        self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
        self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones,
                                                              gamma=self.sched_factor)

        # settings for camera optimization
        if self.train_cameras:
            num_images = len(self.train_dataset)
            self.pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).cuda()
            self.pose_vecs.weight.data.copy_(self.train_dataset.get_pose_init())

            self.optimizer_cam = torch.optim.SparseAdam(self.pose_vecs.parameters(),
                                                        self.conf.get_float('train.learning_rate_cam'))

        self.start_epoch = 0
        if self._is_continue:
            old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

            print("Continuing from", old_checkpnts_dir)
            self.lat_vecs = torch.nn.Embedding.from_pretrained(self.storage.get_latest("lat_vecs"))

            saved_model_state = torch.load(
                os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))
            self.model.load_state_dict(saved_model_state["model_state_dict"])
            self.start_epoch = self.storage.get_latest("epoch")
            self.optimization_steps = self.storage.get_latest("optimization_steps")

            data = torch.load(
                os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
            self.optimizer.load_state_dict(data["optimizer_state_dict"])

            data = torch.load(
                os.path.join(old_checkpnts_dir, self.scheduler_params_subdir, str(kwargs['checkpoint']) + ".pth"))
            self.scheduler.load_state_dict(data["scheduler_state_dict"])

            if self.train_cameras:
                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.optimizer_cam_params_subdir,
                                 str(kwargs['checkpoint']) + ".pth"))
                self.optimizer_cam.load_state_dict(data["optimizer_cam_state_dict"])

                data = torch.load(
                    os.path.join(old_checkpnts_dir, self.cam_params_subdir, str(kwargs['checkpoint']) + ".pth"))
                self.pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

        self.num_pixels = self.conf.get_int('train.num_pixels')
        self.total_pixels = self.train_dataset.total_pixels
        self.img_res = self.train_dataset.img_res
        self.n_batches = len(self.train_dataloader)
        self.plot_freq = self.conf.get_int('train.plot_freq')
        self.checkpoint_freq = self.conf.get_int('train.checkpoint_freq')
        self.storage_freq = self.conf.get_int("train.storage_freq")
        self.plot_conf = self.conf.get_config('plot')

        self.alpha_milestones = self.conf.get_list('train.alpha_milestones', default=[])
        self.alpha_factor = self.conf.get_float('train.alpha_factor', default=0.0)
        for acc in self.alpha_milestones:
            if self.start_epoch > acc:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

    def save_checkpoints(self, epoch, opt_steps):
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(opt_steps) + ".pth"))
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(opt_steps) + ".pth"))
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(opt_steps) + ".pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.train_cameras:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, str(opt_steps) + ".pth"))
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_cam_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, str(opt_steps) + ".pth"))
            torch.save(
                {"epoch": epoch, "pose_vecs_state_dict": self.pose_vecs.state_dict()},
                os.path.join(self.checkpoints_path, self.cam_params_subdir, "latest.pth"))

    def plot(self, epoch, n_plots=3):
        torch.cuda.empty_cache()
        self.model.eval()
        if self.train_cameras:
            self.pose_vecs.eval()
        for i in range(n_plots):
            self.train_dataset.change_sampling_idx(-1)
            indices, model_input, ground_truth = next(iter(self.plot_dataloader))

            model_input["intrinsics"] = model_input["intrinsics"].cuda()
            model_input["uv"] = model_input["uv"].cuda()
            model_input["object_mask"] = model_input["object_mask"].cuda()
            model_input["obj"] = self.lat_vecs(model_input["obj"]).cuda()

            if self.train_cameras:
                pose_input = self.pose_vecs(indices.cuda())
                model_input['pose'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].cuda()

            split = utils.split_input(model_input, self.total_pixels)
            res = []
            for s in split:
                out = self.model(s)
                res.append({
                    'points': out['points'].detach(),
                    'rgb_values': out['rgb_values'].detach(),
                    'network_object_mask': out['network_object_mask'].detach(),
                    'object_mask': out['object_mask'].detach()
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, self.total_pixels, batch_size)

            plt.plot(self.model,
                     indices,
                     model_outputs,
                     model_input['pose'],
                     ground_truth['rgb'],
                     self.plots_dir,
                     epoch * 1000 + i,
                     self.img_res,
                     lat_vec=model_input["obj"],
                     hypo_params=self.model.hyper_net(model_input["obj"]),
                     **self.plot_conf,
                     )
            torch.cuda.empty_cache()

        self.model.train()
        if self.train_cameras:
            self.pose_vecs.train()
        torch.cuda.empty_cache()

    def backward(self, model_outputs, ground_truth):
        self.model.deform_reg_strength = self.get_deform_get_str()
        loss_output = self.loss(model_outputs, ground_truth, self.optimization_steps)
        loss = loss_output['loss']

        self.optimizer.zero_grad()
        if self.train_cameras:
            self.optimizer_cam.zero_grad()

        loss.backward()

        self.optimizer.step()
        if self.train_cameras:
            self.optimizer_cam.step()

        if self.optimization_steps % 50 == 0:
            self.storage.add_entry("rgb_loss", loss_output['rgb_loss'].item())
            self.storage.add_entry("eikonal_loss", loss_output['eikonal_loss'].item())
            self.storage.add_entry("mask_loss", loss_output['mask_loss'].item())
            self.storage.add_entry("deform_loss", loss_output["deform_loss"].item())
            self.storage.add_entry("total_loss", loss_output["loss"].item())
            self.storage.add_entry("deform_reg_str", self.model.deform_reg_strength)
            self.storage.add_entry("optimization_steps", self.optimization_steps)
            self.storage.add_entry("epoch", self.optimization_steps // self.num_datapoints)

        if self.optimization_steps % 200 == 0:
            deformation_mags = torch.linalg.norm(model_outputs["deformation"], dim=1)
            corr_mags = torch.abs(model_outputs["correction"]).flatten()

            deform_mags = {
                "deform": np.random.choice(deformation_mags.detach().cpu().numpy(), size=150),
                "correction": np.random.choice(corr_mags.detach().cpu().numpy(), size=150)
            }
            self.storage.add_entry("deformnet_magnitude", deform_mags)

        if self.optimization_steps % self.plot_freq == 0:
            self.plot(self.optimization_steps // self.plot_freq)

        if self.optimization_steps % self.checkpoint_freq == 0:
            self.save_checkpoints(self.optimization_steps // self.num_datapoints, self.optimization_steps)

        if self.optimization_steps % self.storage_freq == 0:
            self.storage.save()

        if self.optimization_steps % 50 == 0:
            epoch = self.optimization_steps // self.num_datapoints
            idx = self.optimization_steps % self.num_datapoints
            print(
                '{0} [{1}] ({2}/{3}): loss = {4}, rgb_loss = {5}, eikonal_loss = {6}, mask_loss = {7}, deform_loss = {8}, alpha = {9}, lr = {10}'
                    .format(self.expname, epoch, idx, self.n_batches, loss.item(),
                            loss_output['rgb_loss'].item(),
                            loss_output['eikonal_loss'].item(),
                            loss_output['mask_loss'].item(),
                            loss_output["deform_loss"].item(),
                            self.loss.alpha,
                            self.scheduler.get_lr()[0]))

    def run(self):
        print("training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):
            lv = torch.clone(self.lat_vecs.weight)
            self.storage.add_entry("lat_vecs", lv)

            if epoch in self.alpha_milestones:
                self.loss.alpha = self.loss.alpha * self.alpha_factor

            self.train_dataset.change_sampling_idx(self.num_pixels)

            datapoints = list(self.train_dataloader)
            self.num_datapoints = len(datapoints)
            random.shuffle(datapoints)

            for data_index, (indices, model_input, ground_truth) in tqdm.tqdm(enumerate(datapoints),
                                                                              total=self.num_datapoints):
                model_input["intrinsics"] = model_input["intrinsics"].cuda()
                model_input["uv"] = model_input["uv"].cuda()
                model_input["object_mask"] = model_input["object_mask"].cuda()
                model_input["obj"] = self.lat_vecs(model_input["obj"]).cuda()

                if self.train_cameras:
                    pose_input = self.pose_vecs(indices.cuda())
                    model_input['pose'] = pose_input
                else:
                    model_input['pose'] = model_input['pose'].cuda()

                model_outputs = self.model(model_input)
                self.backward(model_outputs, ground_truth)
                self.optimization_steps += 1
            self.scheduler.step()

    def get_deform_get_str(self):
        a = self.model.base_deform_reg / (
                1 + (self.optimization_steps * self.model.deform_reg_decay))
        b = self.model.base_deform_reg * 0.1 - (20 * self.model.deform_reg_decay) * self.optimization_steps
        if self.optimization_steps > self.model.base_deform_reg:
            out = b
        else:
            out = a
        return max(out, 0)
