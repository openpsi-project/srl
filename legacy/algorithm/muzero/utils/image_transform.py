import torch
import torch.nn as nn

from torchvision.transforms import RandomCrop, RandomAffine, RandomResizedCrop, GaussianBlur


class Transforms(object):
    """ Reference : Data-Efficient Reinforcement Learning with Self-Predictive Representations
    Thanks to Repo: https://github.com/mila-iqia/spr.git
    """

    def __init__(self, augmentation, shift_delta=4, image_shape=(96, 96)):
        self.augmentation = augmentation

        self.transforms = []
        for aug in self.augmentation:
            if aug == "affine":
                transformation = RandomAffine(5, (.14, .14), (.9, 1.1), (-5, 5))
            elif aug == "crop":
                transformation = RandomCrop(image_shape)
            elif aug == "rrc":
                transformation = RandomResizedCrop((100, 100), (0.8, 1))
            elif aug == "blur":
                transformation = GaussianBlur((5, 5), (1.5, 1.5))
            elif aug == "shift":
                transformation = nn.Sequential(nn.ReplicationPad2d(shift_delta), RandomCrop(image_shape))
            elif aug == "intensity":
                transformation = intensity_func
            elif aug == "none":
                transformation = nn.Identity()
            else:
                raise NotImplementedError()
            self.transforms.append(transformation)

    def apply_transforms(self, transforms, image):
        for transform in transforms:
            image = transform(image)
        return image

    @torch.no_grad()
    def transform(self, images):
        # images = images.float() / 255. if images.dtype == torch.uint8 else images
        flat_images = images.reshape(-1, *images.shape[-3:])
        processed_images = self.apply_transforms(self.transforms, flat_images)

        processed_images = processed_images.view(*images.shape[:-3], *processed_images.shape[1:])
        return processed_images


@torch.jit.script
def intensity_func(x):
    r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
    noise = 1.0 + (0.05 * r.clamp(-2.0, 2.0))
    return x * noise
