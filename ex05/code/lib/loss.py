from matplotlib.pyplot import grid
import torch
import torch.nn as nn
import torch.nn.functional as F


from .metrics import aepe, pointwise_epe


class FlowLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.weight_decay = 1e-4
        self.gt_interpolation = 'bilinear'

        self.loss_weights = [1 / 16, 1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1]
        self.loss_weights = [100 * weight for weight in self.loss_weights]
        self.reg_params = self.get_regularization_parameters(model)

    def get_regularization_parameters(self, model):
        reg_params = []
        for name, param in model.named_parameters():
            if 'pred' not in name and not name.endswith('bias') and not name.endswith('bn.weight') and param.requires_grad:
                reg_params.append((name, param))

        print("Applying regularization loss with weight decay {} on:".format(self.weight_decay))
        for i, val in enumerate(reg_params):
            name, param = val
            print("\t#{} {}: {} ({})".format(i, name, param.shape, param.numel()))
        print()

        return reg_params

    def forward(self, sample, model_output):

        pointwise_losses = {}
        sub_losses = {}

        gt_flow = sample['gt_flow']
        pred_flows_all = model_output['pred_flows_all']

        total_aepe_loss = 0
        total_reg_loss = 0

        for level, pred_flow in enumerate(pred_flows_all):

            with torch.no_grad():
                gt_flow_resampled = F.interpolate(gt_flow, size=pred_flow.shape[-2:], mode=self.gt_interpolation,
                                                  align_corners=(False if self.gt_interpolation != 'nearest' else None))

            aepe_loss = aepe(gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level])
            pointwise_epe_ = pointwise_epe(gt=gt_flow_resampled, pred=pred_flow, weight=self.loss_weights[level])

            sub_losses['0_aepe/level_%d' % level] = aepe_loss
            pointwise_losses['1_epe/level_%d' % level] = pointwise_epe_

            total_aepe_loss += aepe_loss

        for name, param in self.reg_params:
            reg_loss = torch.sum(torch.mul(param, param)) / 2.0
            total_reg_loss += reg_loss

        total_reg_loss *= self.weight_decay

        total_loss = total_aepe_loss + total_reg_loss

        sub_losses['1_total_aepe'] = total_aepe_loss
        sub_losses['2_reg'] = total_reg_loss

        return total_loss, sub_losses, pointwise_losses


class PhotometricLoss(nn.Module):
    def __init__(self, model, use_smoothness_loss=True):
        super().__init__()
        self.use_smoothness_loss = use_smoothness_loss
        self.img_interpolation = 'bilinear'
        self.eps = 1e-9
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, sample, model_output):
        # TODO: implement the forward loss function of the photometric loss
        # print(sample.keys()) dict_keys(['_base', '_name', '_keyview_idx', 'images', 'gt_flow', '_index', '_dataset', '_orig_height', '_orig_width', '_spatial_aug_scaled_height', '_spatial_aug_scaled_width', '_spatial_aug_crop_y', '_spatial_aug_crop_x'])
        # print(model_output.keys()) dict_keys(['pred_flows_all', 'pred_flow'])

        images = sample['images'] # list of size 2 with [4, 3, 384, 768]
        image_0 = images[0]
        image_1 = images[1]
        pred_flows_all = model_output['pred_flows_all']

        #print(len(pred_flows_all)) # 5
        #print(sample['images'][0].shape)
        #print(pred_flows_all[0].shape) # torch.Size([4, 2, 6, 12])

        total_photometric_loss = 0
        total_smoothness_loss = 0

        for level, pred_flow in enumerate(pred_flows_all):
            #print(pred_flow.shape) [4, 2, 6, 12]

            with torch.no_grad():
                image_0_resampled = F.interpolate(image_0, size=pred_flow.shape[-2:], mode=self.img_interpolation,
                                                  align_corners=(False if self.img_interpolation != 'nearest' else None)) # (batch, channels, height width)
                image_1_resampled = F.interpolate(image_1, size=pred_flow.shape[-2:], mode=self.img_interpolation,
                                                  align_corners=(False if self.img_interpolation != 'nearest' else None))

            _, _, height, width = pred_flow.shape
            x = torch.arange(height).to(self.device)
            y = torch.arange(width).to(self.device)
            grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
            for pred_flow_i, image_0_resampled_i, image_1_resampled_i in zip(pred_flow, image_0_resampled, image_1_resampled):
                print(pred_flow_i.dtype)
                x_2 = grid_x + pred_flow_i[0, :, :]
                y_2 = grid_y + pred_flow_i[1, :, :]
                image_warp = image_1_resampled_i * torch.maximum(0, 1 - torch.abs(x_2 - grid_x))
                #total_photometric_loss += torch.sum((image_0_resampled_i - image_1_resampled_i[:, grid_x + pred_flow_i[0, :, :], grid_y + pred_flow_i[1, :, :]])**2 + self.eps**2)
                #pred_flow_i - torch.roll(pred_flow_i, shifts=1, dims=1)

        return 
