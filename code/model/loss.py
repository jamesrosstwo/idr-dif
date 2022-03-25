import torch
from torch import nn
from torch.nn import functional as F

from model.implicit_differentiable_renderer import IDRNetwork


class IDRLoss(nn.Module):
    def __init__(self, model: IDRNetwork, eikonal_weight, mask_weight, alpha):
        super().__init__()
        self.model = model
        self.eikonal_weight = eikonal_weight
        self.mask_weight = mask_weight
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt,
                                                                          reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)

        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)

        # Regularization of the deformations
        deformation_magnitude = torch.linalg.norm(model_outputs["deformation"], dim=1).mean()
        correction_magnitude = torch.abs(model_outputs["correction"]).mean()
        mask_loss += (deformation_magnitude + correction_magnitude) * 50

        # deform_params = self.model.deform_net.parameters()
        # deform_reg_loss = sum([torch.linalg.norm(p) for p in deform_params]) * self.model.deform_reg_strength
        # mask_loss += deform_reg_loss

        # hyper_params = self.model.hyper_net.parameters()
        # hyper_reg_loss = sum([torch.linalg.norm(p) for p in hyper_params]) * self.model.hyper_reg_strength
        # print("\nHyper regularization loss: {0}".format(hyper_reg_loss))
        # mask_loss += hyper_reg_loss

        # mask_loss += model_outputs["mean_deform"] * 400
        # mask_loss += model_outputs["mean_correction"] * 4000

        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
        }
