import sys
sys.path.append('..')
sys.path.append('../..')
from .datamodule_base import BaseDataModule
from src.datasets.retrieval.dataset_didemo import DiDeMo_Dataset

# from src.datamodules.retrieval.datamodule_base import BaseDataModule
# from src.datasets.retrieval.dataset_didemo import DiDeMo_Dataset


class DidemoDataModule(BaseDataModule):
    '''
    只是选择数据集，构建方法在 BaseDataModule 中
    '''
    def __init__(self, config, *args, **kwargs):
        super().__init__(config)


    @property
    def dataset_name(self):
        return 'Didemo'

    def set_train_dataset(self):
        self.train_dataset = DiDeMo_Dataset(
            subset='train',
            data_path=self.data_dir,
            features_path=self.features_dir,
            max_words=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_framepos=self.slice_frame_pos,
            strategy=self.train_strategy,
            tokenizer=self.tokenizer,
        )

    def set_val_dataset(self):
        self.val_dataset = DiDeMo_Dataset(
            subset='test',
            data_path=self.data_dir,
            features_path=self.features_dir,
            max_words=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_framepos=self.slice_frame_pos,
            strategy=self.eval_strategy,
            tokenizer=self.tokenizer,
        )

    def set_test_dataset(self):
        self.test_dataset = DiDeMo_Dataset(
            subset='test',
            data_path=self.data_dir,
            features_path=self.features_dir,
            max_words=self.max_text_len,
            feature_framerate=self.feature_framerate,
            max_frames=self.max_frames,
            image_resolution=self.image_resolution,
            frame_order=0,
            slice_framepos=self.slice_frame_pos,
            strategy=self.eval_strategy,
            tokenizer=self.tokenizer,
        )

if __name__ == '__main__':
    data_path =  '/home/tzj/datas/DiDeMo/text'
    video_path = '/home/tzj/datas/DiDeMo/videos'
    # frame_path = '/home/tzj/datas/MSVD-Frames/MSVD_frames'
    import sys
    sys.path.append('../..')
    from src.datamodules.recognition.tokenization_clip import SimpleTokenizer as ClipTokenizer
    tokenizer = ClipTokenizer()

    max_words = 32
    feature_framerate = 1
    max_frames = 12
    train_frame_order = 0
    slice_framepos = 2

    dataset = DiDeMo_Dataset(
        subset="train",
        data_dir=data_path,
        features_dir=video_path,
        max_text_len=max_words,
        feature_framerate=feature_framerate,
        tokenizer=tokenizer,
        max_frames=max_frames,
        frame_order=train_frame_order,
        slice_frame_pos=slice_framepos,
    )

    from torch.utils.data import DataLoader
    dataloader_didemo = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    for i, (pairs_text, pairs_mask, pairs_segment, video, video_mask, idx, v_idx) in enumerate(dataloader_didemo):
        print('='*80)
        print(i)
        print('[pairs_text]:', pairs_text.shape)
        print('[pairs_mask]:', pairs_mask.shape)
        print('[pairs_segment]:', pairs_segment.shape)
        print('[video]:', video.shape)
        print('[video_mask]:', video_mask.shape)
        if i == 20:
            break