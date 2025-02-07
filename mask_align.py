import numpy as np
import cv2
import torch
from scipy.ndimage import label, find_objects, binary_dilation


# modified from https://github.com/dougalferg/PyIR
def mask_cleaner(mask, min_blob_size=500, dilation=0, pixel_proportion=0.95):
    original_input = mask.copy()

    # Expand the binary array by the given number of pixels
    if dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)

    # Label connected components
    labeled_array, num_features = label(mask)

    # Find slices corresponding to each blob
    slices = find_objects(labeled_array)

    # List to store blob sizes and corresponding slices
    blob_data = []

    # Calculate total number of 'True' pixels in the original binary array
    total_pixels = np.sum(mask)

    # Gather information about blobs larger than the minimum size
    for i, s in enumerate(slices):
        blob = (labeled_array[s] == (i + 1))
        blob_size = np.sum(blob)

        # Check if the blob is larger than the minimum size
        if blob_size >= min_blob_size:
            blob_data.append((blob_size, blob, s))

    # Sort blobs by size in descending order
    blob_data.sort(key=lambda x: x[0], reverse=True)

    # Create an empty array to accumulate the blobs
    output_array = np.zeros_like(mask)

    # Track cumulative pixel count
    cumulative_pixels = 0
    target_pixels = pixel_proportion * total_pixels

    # Combine blobs from largest to smallest until target proportion is reached
    for blob_size, blob, s in blob_data:
        output_array[s] += blob
        cumulative_pixels += blob_size

        # Stop if we have reached the required proportion of total pixels
        if cumulative_pixels >= target_pixels:
            break

    output_array = output_array * original_input

    return output_array

def he_to_mask(he_image,blur_radius=7,bg_threshold=0.85,clean_mask=True,**clean_mask_kwargs):
    # blur to mitigate speckles in mask. larger radius blur better for larger inputs
    mask = cv2.GaussianBlur(he_image,(blur_radius,blur_radius),0)

    if mask.max() > 1.0: mask /= 255.0

    # threshold pixels in background
    if bg_threshold > 1.0: bg_threshold /= 255.0
    mask = mask[:,:,1] < bg_threshold

    if clean_mask:
        mask = mask_cleaner(mask,**clean_mask_kwargs)

    return mask.astype(float)


# Modified from https://discuss.pytorch.org/t/differentiable-affine-transforms-with-grid-sample/79305
class DifferentiableAffine(torch.nn.Module):
    def __init__(self, scale_w=1.0, scale_h=1.0, trans_x=0.0, trans_y=0.0, rot_1=0.0, rot_2=0.0, requires_grad=True):
        super().__init__()

        def make_parameter(x):
            x = torch.tensor(x).float()
            return torch.nn.Parameter(x)

        self.scale_w = make_parameter(scale_w)
        self.scale_h = make_parameter(scale_h)
        self.rot_1 = make_parameter(rot_1)
        self.rot_2 = make_parameter(rot_2)
        self.trans_x = make_parameter(trans_x)
        self.trans_y = make_parameter(trans_y)

        self.matrix = torch.nn.Parameter(torch.tensor([
            [self.scale_w, self.rot_1, self.trans_x],
            [self.rot_2, self.scale_h, self.trans_y],
        ]).unsqueeze(0), requires_grad=requires_grad)

    def forward(self, x):
        grid = torch.nn.functional.affine_grid(self.matrix, x.size(), align_corners=False)
        return torch.nn.functional.grid_sample(x, grid, align_corners=False)

def align_masks(mask, ref, return_transform=True, return_losses=False, lr=1e-2, max_iter=200):
    # resize to 2d
    if len(mask.shape) == 3: mask = mask[...,0]
    if len(ref.shape) == 3: ref = ref[..., 0]


    # align mask sizes
    mask_resized = cv2.resize(mask, (ref.shape[1],ref.shape[0]), interpolation=cv2.INTER_AREA)

    # Convert masks to tensors
    mask = torch.from_numpy(mask).float()[None,None,...]
    ref = torch.from_numpy(ref).float()[None,None,...]
    mask_resized = torch.from_numpy(mask_resized).float()[None,None,...]

    # Correct for differences in max values
    if ref.max() > 1.0: ref /= mask.max()
    if mask.max() > 1.0: mask /= mask.max()
    if mask_resized.max() > 1.0: mask_resized /= mask_resized.max()

    # define four transforms to start from
    transform_allrot = torch.nn.ModuleList([
        DifferentiableAffine(scale_w=1, scale_h=1, rot_1=0, rot_2=-0), # no rotation
        DifferentiableAffine(scale_w=0, scale_h=0, rot_1=-1, rot_2=1), #  90 deg clockwise
        DifferentiableAffine(scale_w=-1, scale_h=-1, rot_1=0, rot_2=0), # 180 degrees
        DifferentiableAffine(scale_w=0, scale_h=0, rot_1=1, rot_2=-1)]) # 270 degrees clockwise

    # define optimisers
    optim = torch.optim.Adam(transform_allrot.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, threshold=1e-4, patience=50)
    loss_fn = torch.nn.MSELoss()

    # find ideal transform via gradient descent
    losses = []
    for iter in range(max_iter):
        optim.zero_grad()
        loss_iter = []
        for transform in transform_allrot:
            # Transform mask
            warped_mask = transform(mask_resized)

            # Compare similarity of transformed mask and reference mask
            loss = loss_fn(warped_mask, ref)
            loss.backward()
            loss_iter.append(loss.item())

        losses.append(loss_iter)
        scheduler.step(np.sum(loss_iter))
        optim.step()
    losses = np.array(losses)

    # select best transform
    transform = transform_allrot[losses[-1].argmin()]
    losses = losses[:, losses[-1].argmin()]
    transform.matrix.requires_grad = False
    matrix = np.concatenate([transform.matrix.squeeze().numpy(),[[0,0,1],]],axis=0)

    # transform original image
    aligned_mask = transform.forward(mask).squeeze().numpy()

    # pack outputs
    ret = [aligned_mask,]
    if return_transform: ret.append(matrix)
    if return_losses: ret.append(losses)
    if len(ret) == 0: ret = ret[0]

    return ret

def affine_warp(image, matrix):
    image = torch.from_numpy(image).float()
    if image.ndim == 2: image = image.unsqueeze(-1)
    image = image.unsqueeze(0).unsqueeze(0)

    affine = DifferentiableAffine(
        scale_w = matrix[0,0],
        scale_h = matrix[1,1],
        trans_x = matrix[0,2],
        trans_y = matrix[1,2],
        rot_1 = matrix[0,1],
        rot_2 = matrix[1,0],
        requires_grad=False
    )

    warped = torch.stack(
        [affine.forward(image[...,channel]) for channel in range(image.shape[-1])],
        dim=-1).squeeze().numpy()

    return warped