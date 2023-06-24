import numpy as np
import torch


def create_interp_mask(ima, patch_center, patch_width, patch_interp):
    dims = ima.size()
    mask_i = torch.zeros_like(ima)
    for frame_ind in range(dims[0]):
        coor_min = patch_center[frame_ind] - patch_width[frame_ind]
        coor_max = patch_center[frame_ind] + patch_width[frame_ind]
        print("1", coor_min)
        print("2", coor_max)
        # Clip coordinates to within image dims
        coor_min = np.clip(coor_min,0,dims[1:3])     
        coor_max = np.clip(coor_max,0,dims[1:3])
        print("3", coor_min)
        print("4", coor_max)
        mask_i[frame_ind,
                coor_min[0]:coor_max[0],
                coor_min[1]:coor_max[1]] = patch_interp[frame_ind]
        print("Done")

    return mask_i

def patch_ex(ima1, ima2, num_classes=None, core_percent=0.8, tolerance=None):
    # Exchange patches between two image tensors based on a random interpolation factor

    # Create random anomaly
    dims = np.array(ima1.size())
    core = core_percent * dims  # Width of core region
    offset = (1 - core_percent) * dims / 2  # Offset to center core

    min_width = np.round(0.05*dims[1])
    max_width = np.round(0.2*dims[1])

    center_dim1 = np.random.randint(offset[1],offset[1]+core[1],size=dims[0])
    center_dim2 = np.random.randint(offset[2],offset[2]+core[2],size=dims[0])
    patch_center = np.stack((center_dim1,center_dim2),1)
    patch_width = np.random.randint(min_width,max_width,size=dims[0])
    if num_classes == None:
        #interpolation factor between 5 and 95%
        patch_interp = np.random.uniform(0.05,0.95,size=dims[0])
        print(patch_interp)
    else:
        #interpolation between 0 and 1, num class options
        patch_interp = np.random.choice(num_classes-1,size=dims[0])/(num_classes-1)#subtract 1 to exclude default class
    print(patch_center)
    print(patch_width) 
    offset = 1E-5#offset to separate 0 patches from background
    mask_i = create_interp_mask(ima1, patch_center, patch_width, patch_interp + offset)
    patch_mask = np.clip(np.ceil(mask_i),0,1)#all patches set to 1
    mask_i = mask_i-patch_mask*offset#get rid of offset
    mask_inv = patch_mask-mask_i
    zero_mask = 1-patch_mask#zero in the region of the patch

    patch_set1 = mask_i*ima1 + mask_inv*ima2 #interpolate between patches
    patch_set2 = mask_inv*ima1 + mask_i*ima2

    patchex1 = ima1*zero_mask + patch_set1
    patchex2 = ima2*zero_mask + patch_set2

    # if tolerance:
    #     valid_label = np.any(
    #         np.floor(patch_mask*ima1*tolerance**-1)*tolerance != \
    #         np.floor(patch_mask*ima2*tolerance**-1)*tolerance,
    #         axis=3)
            
    # else:
    valid_label = np.any((patch_mask*ima1 != patch_mask*ima2).numpy())
    label = valid_label[...,None]*mask_inv.numpy()

    # if num_classes is not None:
    #     label = label*(num_classes-1)
    #     label = to_categorical(label,num_classes)

    return [patchex1, patchex2, mask_i, label]

def create_interp_mask_batch(batch, patch_center, patch_width, patch_interp):
    dims = batch.size()
    mask_i = torch.zeros_like(batch)
    for frame_ind in range(dims[1]):
        coor_min = patch_center[frame_ind] - patch_width[frame_ind]
        coor_max = patch_center[frame_ind] + patch_width[frame_ind]

        # Clip coordinates to within image dims
        coor_min = np.clip(coor_min,0,dims[2:4])     
        coor_max = np.clip(coor_max,0,dims[2:4])

        mask_i[:, frame_ind,
                coor_min[0]:coor_max[0],
                coor_min[1]:coor_max[1]] = patch_interp[frame_ind]
    return mask_i

def patch_ex_batch(batch1, batch2, num_classes=None, core_percent=0.8, tolerance=None):
    # Exchange patches between two image tensors based on a random interpolation factor

    # Create random anomaly
    dims = np.array(batch1.size())
    core = core_percent * dims  # Width of core region
    offset = (1 - core_percent) * dims / 2  # Offset to center core

    min_width = np.round(0.05*dims[2])
    max_width = np.round(0.2*dims[2])

    center_dim1 = np.random.randint(offset[2],offset[2]+core[2],size=dims[1])
    center_dim2 = np.random.randint(offset[3],offset[3]+core[3],size=dims[1])
    patch_center = np.stack((center_dim1,center_dim2),1)
    patch_width = np.random.randint(min_width,max_width,size=dims[1])
    if num_classes == None:
        #interpolation factor between 5 and 95%
        patch_interp = np.random.uniform(0.05,0.95,size=dims[1])
        # print(patch_interp)
    else:
        #interpolation between 0 and 1, num class options
        patch_interp = np.random.choice(num_classes-1,size=dims[1])/(num_classes-1)#subtract 1 to exclude default class
  
    offset = 1E-5#offset to separate 0 patches from background
    mask_i = create_interp_mask_batch(batch1, patch_center, patch_width, patch_interp + offset)
    patch_mask = np.clip(np.ceil(mask_i),0,1)#all patches set to 1
    mask_i = mask_i-patch_mask*offset#get rid of offset
    mask_inv = patch_mask-mask_i
    zero_mask = 1-patch_mask#zero in the region of the patch

    patch_set1 = mask_i*batch1 + mask_inv*batch2 #interpolate between patches
    patch_set2 = mask_inv*batch1 + mask_i*batch2

    patchex1 = batch1*zero_mask + patch_set1
    patchex2 = batch2*zero_mask + patch_set2

    # if tolerance:
    #     valid_label = np.any(
    #         np.floor(patch_mask*ima1*tolerance**-1)*tolerance != \
    #         np.floor(patch_mask*ima2*tolerance**-1)*tolerance,
    #         axis=3)
            
    # else:
    valid_label = np.any((patch_mask*batch1 != patch_mask*batch2).numpy())
    label = valid_label[...,None]*mask_inv.numpy()

    # if num_classes is not None:
    #     label = label*(num_classes-1)
    #     label = to_categorical(label,num_classes)

    return patchex1, patchex2, torch.Tensor(label)