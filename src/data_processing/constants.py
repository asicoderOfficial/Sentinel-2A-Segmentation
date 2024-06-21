from torchvision import transforms

PATCHES_DEFAULT_SIZES = [32, 64, 128]

# TODO: Add all possible augmentations from the PyTorch library
AUGMENTATIONS_DECODER = {
    'compose': transforms.Compose,
    'resize': transforms.Resize,
    'center_crop': transforms.CenterCrop,
    'to_tensor': transforms.ToTensor,
    'normalize': transforms.Normalize,
    'random_resized_crop': transforms.RandomResizedCrop,
    'grayscale': transforms.Grayscale,
    'random_horizontal_flip': transforms.RandomHorizontalFlip,
    'random_vertical_flip': transforms.RandomVerticalFlip,
    'random_rotation': transforms.RandomRotation,
    'color_jitter': transforms.ColorJitter,
    'random_affine': transforms.RandomAffine,
    'random_perspective': transforms.RandomPerspective,
    'random_crop': transforms.RandomCrop,
    'pad': transforms.Pad,
    'random_erasing': transforms.RandomErasing,
    'gaussian_blur': transforms.GaussianBlur,
    'random_apply': transforms.RandomApply,
    'random_choice': transforms.RandomChoice,
    'random_order': transforms.RandomOrder,
    'five_crop': transforms.FiveCrop,
    'ten_crop': transforms.TenCrop,
    'linear_transformation': transforms.LinearTransformation,
    'lambda': transforms.Lambda,
    'to_pil_image': transforms.ToPILImage,
    'to_tensor': transforms.ToTensor,
    'normalize': transforms.Normalize,
}
