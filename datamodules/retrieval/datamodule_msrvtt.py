import sys

sys.path.append('../..')
from .datamodule_base import BaseDataModule
from src.datasets.retrieval.dataset_msrvtt import MSRVTT_EvalDataset, MSRVTT_TrainDataset


class MSRVTTDataModule(BaseDataModule):
    '''
    只是选择数据集，构建方法在 BaseDataModule 中
    '''
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)


    @property
    def dataset_name(self):
        return 'MSRVTT'

    def set_train_dataset(self):
        self.train_dataset = MSRVTT_TrainDataset(
            subset='train',
            data_dir=self.data_dir,
            # features_dir=self.features_dir,
            max_text_len=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_frame_pos=self.slice_frame_pos,
            strategy=3,
            tokenizer=self.tokenizer,
            unfold_sentences=True
        )

    def set_val_dataset(self):
        self.val_dataset = MSRVTT_EvalDataset(
            subset='test',
            data_dir=self.data_dir,
            features_dir=self.features_dir,
            max_text_len=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_frame_pos=self.slice_frame_pos,
            strategy=1,
            tokenizer=self.tokenizer,
        )

    def set_test_dataset(self):
        self.test_dataset = MSRVTT_EvalDataset(
            subset='test',
            data_dir=self.data_dir,
            features_dir=self.features_dir,
            max_text_len=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_frame_pos=self.slice_frame_pos,
            strategy=1,
            tokenizer=self.tokenizer,
        )

if __name__ == '__main__':
    ...