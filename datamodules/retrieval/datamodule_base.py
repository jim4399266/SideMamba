from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AutoTokenizer,
)
from pathlib import Path

from .tokenization_clip import SimpleTokenizer as ClipTokenizer


import sys
sys.path.append('../..')


class BaseDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dist = config['dist']
        # self.data_dir = config['data_root']
        # self.features_dir = config['features_dir']
        self.num_workers = config['num_workers']
        self.batch_size = config['per_gpu_batch_size']
        self.eval_batch_size = config['batch_size_val']
        self.shuffle = config['shuffle']
        self.pin_memory = config['pin_memory']


        self.data_dir = str(Path(config['data_root']) / config['dataset']['data_dir'])
        self.features_dir = str(Path(config['data_root']) / config['dataset']['features_dir'])

        self.image_resolution = config['encoder']['image_resolution']

        # self.max_text_len = config['encoder']['max_text_len']
        self.max_text_len = config['dataset']['max_text_len']

        # self.feature_framerate = config['encoder']['feature_framerate']
        # self.max_frames = config['encoder']['max_frames']
        # self.slice_frame_pos = config['encoder']['slice_frame_pos']
        # self.strategy = config['encoder']['strategy']

        self.feature_framerate = config['dataset']['feature_framerate']
        self.max_frames = config['dataset']['max_frames']
        self.slice_frame_pos = config['dataset']['slice_frame_pos']
        self.train_strategy = config['dataset']['train_strategy']
        self.eval_strategy = config['dataset']['eval_strategy']

        # 图片转换器，用于在dataset中将原始图片转换到到image_size大小
        self.frames_transform_keys = (
            ['clip']
            if len(config['encoder']['frames_transform_keys']) == 0
            else config['encoder']['frames_transform_keys']
        )
        self.video_transform_keys = (
            ['clip']
            if len(config['encoder']['video_transform_keys']) == 0
            else config['encoder']['video_transform_keys']
        )


        self.setup_flag = False


    def get_pretrained_tokenizer(self, tokenizer):
        # 获取分词器，考虑分布式情况
        if (tokenizer == 'simple' or 'simple' in tokenizer or
                tokenizer == 'clip' or 'clip' in tokenizer):
            print(f'ClipTokenizer')
            return ClipTokenizer()
        else:
            return AutoTokenizer.from_pretrained(tokenizer)


    def setup(self, stage: str) -> None:
        # if not self.setup_flag or manual:
        # # 加载分词器
        self.tokenizer = self.get_pretrained_tokenizer(
            str(Path(self.config['pretrained_model_dir']) / self.config['encoder']['tokenizer']))

        if stage == 'fit':
            self.set_train_dataset()
            self.set_val_dataset()
            # 设置采样器
            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.train_sampler = None
                self.val_sampler = None
        elif stage == 'validate':
            self.set_val_dataset()
            if self.dist:
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
            else:
                self.val_sampler = None
        elif stage=='test':
            self.set_test_dataset()
            if self.dist:
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.test_sampler = None
        # elif stage=='predict':
        #     self.set_predict_dataset()
        #
        else:
            raise NotImplementedError
        self.setup_flag = True

    def set_train_dataset(self):
        raise NotImplementedError("return tuple of train dataset class")

    def set_val_dataset(self):
        raise NotImplementedError("return tuple of validation dataset class")

    def set_test_dataset(self):
        raise NotImplementedError("return tuple of test dataset class")

    def set_predict_dataset(self):
        print('No prediction.')

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")


    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.shuffle if self.train_sampler == None else False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return dataloader


    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader


    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        return dataloader


