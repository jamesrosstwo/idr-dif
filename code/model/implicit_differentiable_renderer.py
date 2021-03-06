import torch
import torch.nn as nn
import numpy as np

from model import dif_modules
from model.hyper_net import HyperNetwork
from utils import rend_util
from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork


def add_latent(x, latent_code):
    code_const = torch.broadcast_to(latent_code, (x.shape[0], latent_code.shape[1]))
    return torch.cat([x, code_const], dim=1)


class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            deform_net,
            latent_code_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0,
    ):
        super().__init__()
        self.deform_net = deform_net

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, x_in, hypo_params, latent_code, deform, ret_deforms=False):
        assert hypo_params is not None

        deformation = torch.zeros(x_in.shape[0], 3).cuda()
        scalar_correction = torch.zeros(x_in.shape[0], 1).cuda()

        x_deform = x_in
        if deform:
            adj_x = self.deform_net(x_in, params=hypo_params)["model_out"]
            deformation = adj_x[0, :, :3]
            scalar_correction = adj_x[0, :, 3:]
            x_deform = x_in + deformation

        if self.embed_fn is not None:
            x_deform = self.embed_fn(x_deform)
        x = x_deform

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, x_deform], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)


        if deform:
            x[:, :1] += scalar_correction
        if ret_deforms:
            return x, deformation, scalar_correction
        return x

    def gradient(self, x, hypo_params, latent_code, deform):
        x.requires_grad_(True)
        y = self.forward(x, hypo_params, latent_code, deform=deform)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            latent_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size + latent_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors, latent_code):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = add_latent(rendering_input, latent_code)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x


class IDRNetwork(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.feature_vector_size = conf.get_int('feature_vector_size')

        latent_code_dim = conf.get_int('latent_vector_size')
        implicit_conf = conf.get_config('implicit_network')
        implicit_conf["latent_code_size"] = latent_code_dim

        rendering_conf = conf.get_config('rendering_network')
        rendering_conf["latent_vector_size"] = latent_code_dim

        self.should_deform = conf["deform"]
        # Deform-Net
        deform_config = conf.get_config("deform_network")
        self.deform_net = dif_modules.SingleBVPNet(mode='mlp', in_features=3, out_features=4, **deform_config)
        self.base_deform_reg = deform_config["base_reg_strength"]
        self.deform_reg_decay = deform_config["reg_decay"]
        self.deform_reg_strength = self.base_deform_reg

        self.implicit_network = ImplicitNetwork(self.feature_vector_size, self.deform_net, **implicit_conf)
        self.rendering_network = RenderingNetwork(self.feature_vector_size, **rendering_conf)

        hyper_config = conf.get_config("hyper_network")

        # Hyper-Net
        self.hyper_net = HyperNetwork(hyper_in_features=latent_code_dim,
                                      hypo_module=self.deform_net, **hyper_config)
        self.hyper_reg_strength = 3

        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')

    def forward(self, input):
        # Parse model input
        intrinsics = input["intrinsics"]
        uv = input["uv"]
        pose = input["pose"]
        object_mask = input["object_mask"].reshape(-1)
        latent_code = input["obj"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        batch_size, num_pixels, _ = ray_dirs.shape

        hypo_params = self.hyper_net(latent_code)

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(
                sdf=lambda x: self.implicit_network(x, hypo_params, latent_code, self.should_deform)[:, 0],
                cam_loc=cam_loc,
                object_mask=object_mask,
                ray_directions=ray_dirs)

        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) + dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output, deformations, scalar_correction = \
            self.implicit_network.forward(points, hypo_params, latent_code, self.should_deform, ret_deforms=True)
        sdf_output = sdf_output[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points], 0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points, hypo_params, latent_code, self.should_deform)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all, hypo_params, latent_code, self.should_deform)
            surface_points_grad = g[:N, 0, :].clone().detach()
            # For eikonal loss. Don't include deformations.
            grad_theta = self.implicit_network.gradient(points_all, hypo_params, latent_code, False)[N:, 0, :]

            differentiable_surface_points = self.sample_network(surface_output,
                                                                surface_sdf_values,
                                                                surface_points_grad,
                                                                surface_dists,
                                                                surface_cam_loc,
                                                                surface_ray_dirs)

        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(differentiable_surface_points, view, latent_code)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta,
            'deformation': deformations,
            'correction': scalar_correction
        }

        return output

    def get_rbg_value(self, points, view_dirs, latent_code):

        hypo_params = self.hyper_net(latent_code)
        output = self.implicit_network(points, hypo_params, latent_code, self.should_deform)

        # Do not deform here to avoid normals issues
        g = self.implicit_network.gradient(points, hypo_params, latent_code, deform=False)
        normals = torch.zeros(g[:, 0, :].shape).cuda()
        v_dirs = torch.zeros(view_dirs.shape).cuda()

        feature_vectors = output[:, 1:]

        rgb_vals = self.rendering_network(points, normals, v_dirs, feature_vectors, latent_code)

        return rgb_vals
