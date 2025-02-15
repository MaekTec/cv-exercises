import os
import os.path as osp
import importlib

import torch
import torch.nn as nn


def get_function(name):  # from https://github.com/aschampion/diluvian/blob/master/diluvian/util.py
    mod_name, func_name = name.rsplit('.', 1)
    mod = importlib.import_module(mod_name)
    func = getattr(mod, func_name)
    return func


def get_class(name):
    return get_function(name)


def save_model(model, base_path=None, base_name=None, evo=None, epoch=None, iter=None, max_to_keep=None):

    name = base_name if base_name is not None else "checkpoint-model"
    name = name + "-evo-{:02d}".format(evo) if evo is not None else name
    name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
    name = name + "-iter-{:09d}".format(iter) if iter is not None else name
    name += ".pt"
    path = osp.join(base_path, name)

    torch.save(model.state_dict(), path)

    if max_to_keep is not None:
        base_name = base_name if base_name is not None else "checkpoint-model"
        files = sorted([x for x in os.listdir(base_path) if x.startswith(base_name) and x.endswith(".pt")])

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(osp.join(base_path, file_to_be_removed))
            del files[0]

    return path


def save_all(model, optim, scheduler=None, info_dict=None,
             base_path=None, base_name=None, evo=None, epoch=None, iter=None, max_to_keep=None):

    name = base_name if base_name is not None else "checkpoint-train"
    name = name + "-evo-{:02d}".format(evo) if evo is not None else name
    name = name + "-epoch-{:04d}".format(epoch) if epoch is not None else name
    name = name + "-iter-{:09d}".format(iter) if iter is not None else name
    name += ".pt"
    path = osp.join(base_path, name)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    if info_dict is not None:
        checkpoint.update(info_dict)

    torch.save(checkpoint, path)

    if max_to_keep is not None:
        base_name = base_name if base_name is not None else "checkpoint-train"
        files = sorted([x for x in os.listdir(base_path) if x.startswith(base_name) and x.endswith(".pt")])

        while len(files) > max_to_keep:
            file_to_be_removed = files[0]
            os.remove(osp.join(base_path, file_to_be_removed))
            del files[0]

    return path


def load_model(path, model, strict=True):
    model.load_state_dict(torch.load(path), strict=strict)


def load_all(path, model, optim=None, scheduler=None):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint


def is_checkpoint(path, base_name=None):
    file = osp.split(path)[1]
    return osp.isfile(path) and file.endswith(".pt") and (file.startswith(base_name) if base_name is not None else True)


def get_checkpoints(path, base_name=None, include_iter=False):

    if osp.isdir(path):
        checkpoints = [x for x in os.listdir(path)]
        checkpoints = [osp.join(path, checkpoint) for checkpoint in checkpoints]
        checkpoints = [checkpoint for checkpoint in checkpoints if is_checkpoint(checkpoint, base_name)]
    elif osp.isfile(path):
        checkpoints = [path] if is_checkpoint(path) else []
    else:
        checkpoints = []

    if include_iter:
        checkpoints = [(iter_from_path(checkpoint), checkpoint) for checkpoint in checkpoints]

    return checkpoints


def iter_from_path(path):
    idx = path.find('-iter-')
    iter = int(path[idx + 6: idx + 6 + 9])
    return iter


class WeightsOnlySaver:
    def __init__(self, model=None, base_path=None, base_name=None, max_to_keep=None):

        self.model = model
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "checkpoint-model"
        self.max_to_keep = max_to_keep

    def save(self, evo=None, epoch=None, iter=None):

        save_path = save_model(model=self.model, base_path=self.base_path, base_name=self.base_name, 
                               evo=evo, epoch=epoch, iter=iter, max_to_keep=self.max_to_keep)

        return save_path

    def get_checkpoints(self, include_iter=False):

        checkpoints = get_checkpoints(path=self.base_path, base_name=self.base_name, include_iter=include_iter)
        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iter=False):
        return self.get_checkpoints(include_iter=include_iter)[-1]

    def has_checkpoint(self, path):
        return path in self.get_checkpoints()

    def load(self, full_path=None, strict=True):

        checkpoint = get_checkpoints(path=full_path)[-1] if full_path is not None else self.get_latest_checkpoint()
        print("Loading checkpoint {} (strict: {}).".format(checkpoint, strict))
        load_model(path=checkpoint, model=self.model, strict=strict)


class TrainStateSaver:
    def __init__(self, model=None, optim=None, scheduler=None, base_path=None, base_name=None, max_to_keep=None):

        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.base_path = base_path
        self.base_name = base_name if base_name is not None else "checkpoint-train"
        self.max_to_keep = max_to_keep

    def save(self, info_dict=None, evo=None, epoch=None, iter=None):

        save_path = save_all(model=self.model, optim=self.optim, scheduler=self.scheduler, info_dict=info_dict,
                             base_path=self.base_path, base_name=self.base_name,
                             evo=evo, epoch=epoch, iter=iter, max_to_keep=self.max_to_keep)

        return save_path

    def get_checkpoints(self, include_iter=False):

        checkpoints = get_checkpoints(path=self.base_path, base_name=self.base_name, include_iter=include_iter)
        return sorted(checkpoints)

    def get_latest_checkpoint(self, include_iter=False):
        return self.get_checkpoints(include_iter=include_iter)[-1]

    def has_checkpoint(self, path):
        return path in self.get_checkpoints()

    def load(self, full_path=None):

        checkpoint = get_checkpoints(path=full_path)[-1] if full_path is not None else self.get_latest_checkpoint()
        print("Loading checkpoint {}).".format(checkpoint))
        out_dict = load_all(path=checkpoint, model=self.model, optim=self.optim, scheduler=self.scheduler)
        return out_dict


def warp(x, offset=None, grid=None, padding_mode='border'):  # based on PWC-Net Github Repo
    """Samples from a tensor according to an offset (e.g. optical flow), or according to given grid locations.

    Input can either be an offset or a grid of absolute sampling locations.

    Args:
        x: Input tensor with shape (N,C,H,W).
        offset: Offset with shape (N,2,h_out,w_out) where h_out can w_out can differ from size H and W of x.
        grid: Grid of absolute sampling locations with shape (N,2,h_out,w_out) where h_out can w_out can
         differ from size H and W of x.
        padding_mode: Padding for sampling out of the image border. Can be 'border' or 'zeros'.
    Returns:
        Tuple (output, mask) containing
            output: Sampled points from x with shape (N,C,h_out,w_out).
            mask: Sampling mask with shape (N,1,h_out,w_out).
    """

    N = x.shape[0]

    assert (offset is None and grid is not None) or (grid is None and offset is not None)

    if offset is not None:

        h_out, w_out = offset.shape[-2:]
        device = x.get_device()

        yy, xx = torch.meshgrid(torch.arange(h_out), torch.arange(w_out))  # both (h_out, w_out)
        xx = xx.to(device)
        yy = yy.to(device)
        xx = (xx + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        yy = (yy + 0.5).unsqueeze_(0).unsqueeze_(0)  # 1, 1, h_out, w_out
        xx = xx.repeat(N, 1, 1, 1)  # TODO: maybe this can be removed?
        yy = yy.repeat(N, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()  # N, 2, h_out, w_out

        grid.add_(offset)

    # scale grid to [-1,1]
    h_out, w_out = grid.shape[-2:]
    h_x, w_x = x.shape[-2:]

    grid = grid.permute(0, 2, 3, 1)  # N, h_out, w_out, 2
    xgrid, ygrid = grid.split([1, 1], dim=-1)
    xgrid = 2*xgrid/w_x - 1
    ygrid = 2*ygrid/h_x - 1
    grid = torch.cat([xgrid, ygrid], dim=-1)

    output = nn.functional.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)  # N, C, h_out, w_out

    if padding_mode == 'border':
        mask = torch.ones(size=(N, 1, h_out, w_out), device=output.device, requires_grad=False)

    else:
        mask = torch.ones(size=(N, 1, h_x, w_x), device=output.device, requires_grad=False)
        mask = nn.functional.grid_sample(mask, grid, padding_mode='zeros', align_corners=False)  # N, 1, h_out, w_out
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

    return output, mask  # N, C, h_out, w_out ; N, 1, h_out, w_out


def shift_multi(x, offsets, padding_mode='zeros'):
    """Shifts a tensor x with given integer offsets.

    Args:
        x: Input tensor with shape (N,C,H,W).
        offsets: Shift offsets with shape (S,2).
        padding_mode: Padding for shifting out of the image border. Can be 'border' or 'zeros'.

    Returns:
        Tuple (shifteds, masks) containing
            shifteds: Shifted tensor x with shape (N,S,C,H,W).
            masks: Masks with shape (N,S,H,W). Masks are 0 for shifts out of the image border when
              padding_mode is 'zeros', otherwise 1.
    """

    offsets_ = offsets.int()
    assert torch.all(offsets_ == offsets)

    dxs, dys = offsets_[:, 0], offsets_[:, 1]
    N, _, H, W = x.shape
    device = x.device
    base_mask = torch.ones(size=(N, 1, H, W), dtype=torch.float32, device=device, requires_grad=False)

    pad_l, pad_r = max(0, -1*torch.min(dxs).item()), max(0, torch.max(dxs).item())
    pad_top, pad_bot = max(0, -1*torch.min(dys).item()), max(0, torch.max(dys).item())
    pad_size = (pad_l, pad_r, pad_top, pad_bot)
    pad_fct = nn.ConstantPad2d(pad_size, 0) if padding_mode == 'zeros' else torch.nn.ReplicationPad2d(pad_size)
    x = pad_fct(x)
    base_mask = pad_fct(base_mask)

    shifteds = []
    masks = []
    for dx, dy in zip(dxs, dys):
        shifted = x[:, :, pad_top+dy : pad_top+dy+H, pad_l+dx : pad_l+dx+W]  # N, C, H, W
        mask = base_mask[:, :, pad_top+dy : pad_top+dy+H, pad_l+dx : pad_l+dx+W]  # N, 1, H, W
        shifteds.append(shifted)
        masks.append(mask)

    shifteds = torch.stack(shifteds, 1)  # N, S, C, H, W
    masks = torch.cat(masks, 1)  # N, S, H, W

    return shifteds, masks


def shift(x, offset, padding_mode='zeros'):
    """Shifts a tensor x with a given integer offset.

    Args:
        x: Input tensor with shape (N,C,H,W).
        offset: Shift offset with shape (2).
        padding_mode: Padding for shifting out of the image border. Can be 'border' or 'zeros'.

    Returns:
        Tuple (shifted, mask) containing
            shifted: Shifted tensor x with shape (N,C,H,W).
            mask: Mask with shape (N,1,H,W). Mask is 0 for shifts out of the image border when
              padding_mode is 'zeros', otherwise 1.
    """

    offsets = offset.unsqueeze(0)
    shifteds, masks = shift_multi(x=x, offsets=offsets, padding_mode=padding_mode)
    shifted = shifteds.squeeze(1)

    return shifted, mask
