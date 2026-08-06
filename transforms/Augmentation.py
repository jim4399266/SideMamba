from torch.utils.data._utils.collate import default_collate
from .transform import *
from .random_erasing import RandomErasing
# from RandAugment import RandAugment
from randaugment import RandAugment


class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return [self.worker(img) for img in img_group], label


class SplitLabel(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img):
        img_group, label = img
        return self.worker(img_group), label



def train_augmentation(input_size, flip=True):
    if flip:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            GroupRandomHorizontalFlip(is_flow=False)])
    else:
        return torchvision.transforms.Compose([
            GroupRandomSizedCrop(input_size),
            # GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
            GroupRandomHorizontalFlip_sth()])


def get_augmentation(training, input_size, dataset, rand_aug, rand_erase):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]
    scale_size = 256 if input_size == 224 else input_size

    normalize = GroupNormalize(input_mean, input_std)
    if 'something' in dataset:
        groupscale = GroupScale((256, 320))
    else:
        groupscale = GroupScale(int(scale_size))


    common = torchvision.transforms.Compose([
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize])

    if training:
        auto_transform = None
        erase_transform = None
        if rand_aug:
            auto_transform = create_random_augment(
                input_size=256,
                auto_augment="rand-m7-n4-mstd0.5-inc1",
                interpolation="bicubic"
            )
        if rand_erase:
            erase_transform = RandomErasing(
                0.25,
                mode='pixel',
                max_count=1,
                num_splits=1,
                device="cpu",
            )           

        train_aug = train_augmentation(
            input_size,
            flip=False if 'something' in dataset else True)

        unique = torchvision.transforms.Compose([
            groupscale,
            train_aug,
            GroupRandomGrayscale(p=0 if 'something' in dataset else 0.2),
        ])

        if auto_transform is not None:
            print('=> ########## Using RandAugment!')
            unique = torchvision.transforms.Compose([
                SplitLabel(auto_transform), unique])

        if erase_transform is not None:
            print('=> ########## Using RandErasing!')
            return torchvision.transforms.Compose([
                unique, common, SplitLabel(erase_transform)
            ])
            
        return torchvision.transforms.Compose([unique, common])

    else:
        unique = torchvision.transforms.Compose([
            groupscale,
            GroupCenterCrop(input_size)])
        return torchvision.transforms.Compose([unique, common])


def get_test_augmentation(input_size, dataset_name, test_crops):
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    #rescale size
    if 'something' in dataset_name or 'sth' in dataset_name:
        scale_size = (256, 320)
    else:
        scale_size = 256 if input_size == 224 else input_size

    # control the spatial crop
    if test_crops == 1:  # one crop
        cropping = torchvision.transforms.Compose([
            GroupScale(scale_size),
            GroupCenterCrop(input_size),
        ])
    elif test_crops == 3:  # do not flip, so only 3 crops (left right center)
        cropping = torchvision.transforms.Compose([
            GroupFullResSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif test_crops == 5:  # do not flip, so only 5 crops
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
                flip=False)
        ])
    elif test_crops == 10:
        cropping = torchvision.transforms.Compose([
            GroupOverSample(
                crop_size=input_size,
                scale_size=scale_size,
            )
        ])
    else:
        raise ValueError("Only 1, 3, 5, 10 crops are supported while we got {}".format(args.test_crops))

    return torchvision.transforms.Compose([
        cropping,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        GroupNormalize(input_mean,input_std),
    ])

def randAugment(transform_train, config):
    print('Using RandAugment!')
    transform_train.transforms.insert(0, GroupTransform(RandAugment(config.data.randaug.N, config.data.randaug.M)))
    return transform_train




def multiple_samples_collate(batch):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels = zip(*batch)
    # print(inputs, flush=True)
    # print(labels, flush=True)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    inputs, labels = (
        default_collate(inputs),
        default_collate(labels),
    )
    return inputs, labels