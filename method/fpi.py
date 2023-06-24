import numpy as np
import torch

def create_mask_batch(batch, patch_center, patch_width, patch_interp):
    dims = batch.size()[2:4]
    mask_i = torch.zeros_like(batch)

    coor_min = patch_center - patch_width
    coor_max = patch_center + patch_width

    coor_min = np.clip(coor_min,0,dims)     
    coor_max = np.clip(coor_max,0,dims)

    mask_i[:, 0,
            coor_min[0]:coor_max[0],
            coor_min[1]:coor_max[1]] = patch_interp

    return mask_i

def patch_ex_batch(batch1, batch2, brain_core=0.7):
    # Exchange patches between two batches based on a random interpolation factor

    # Create random anomaly
    dims = np.array(batch1.size())
    img_size = dims[2]

    core = brain_core * img_size
    offset = (1 - brain_core) * img_size / 2

    min_width = np.round(0.05 * img_size)
    max_width = np.round(0.2 * img_size)

    center_x = np.random.randint(offset, offset+core)
    center_y = np.random.randint(offset, offset+core)
    patch_center = np.stack((center_x, center_y))
    patch_width = np.random.randint(min_width, max_width)

    interpolation_factor = np.random.uniform(0.05, 0.95)
  
    offset = 1E-5 # separate 0 pixels from background
    mask = create_mask_batch(batch1, patch_center, patch_width, interpolation_factor + offset)
    patch_mask = np.clip(np.ceil(mask), 0, 1) # all pixels set to 1
    mask = mask - patch_mask * offset # get rid of offset
    mask_inv = patch_mask - mask
    zero_mask = 1 - patch_mask # zero in the region of the patch

    # Interpolate between patches
    patch_set1 = mask * batch1 + mask_inv * batch2
    patch_set2 = mask_inv * batch1 + mask * batch2

    patch_batch1 = batch1 * zero_mask + patch_set1
    patch_batch2 = batch2 * zero_mask + patch_set2

    valid_label = np.any((patch_mask*batch1 != patch_mask*batch2).numpy())
    label = valid_label[...,None] * mask_inv.numpy()

    return patch_batch1, patch_batch2, torch.Tensor(label)