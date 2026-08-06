from torch.utils.data import Dataset
import re
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Union
import json
import random
from collections import defaultdict
import sys
sys.path.append('../..')
# from ..transforms import keys_to_transforms
from src.datasets.rawframes_util import RawFramesExtractorCV2_Image
# from src.datasets.rawframes_util import RawFramesExtractorCV2
from src.datasets.rawvideo_util import RawVideoExtractor


class MSRVTT_TrainDataset(Dataset):
    def __init__(
            self,
            subset: str,
            data_dir: Union[str, Path],
            # features_dir: Union[str, Path],
            max_text_len:int = 40,
            feature_framerate: Union[int, float] = 1.0,
            max_frames: Union[int, float] = 100,
            image_resolution: int = 224,
            frame_order: int = 0,
            slice_frame_pos: int = 0,
            strategy: int = 1,
            load_video = None,
            tokenizer = None,
            unfold_sentences=True,
    ):
        super().__init__()
        assert subset in ["train"]
        self.subset = subset

        self.data_dir = data_dir
        # self.features_dir = features_dir

        self.feature_framerate = feature_framerate
        self.max_frames = max_frames

        self.max_text_len = max_text_len
        self.image_resolution = image_resolution

        # self.transforms = transforms
        self.tokenizer = tokenizer

        # 0: ordinary order; 1: reverse order; 2: random order.
        assert frame_order in [0, 1, 2]
        self.frame_order = frame_order
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        assert slice_frame_pos in [0, 1, 2]
        self.slice_frame_pos = slice_frame_pos

        self.strategy = strategy
        print(subset, ' strategy====', strategy)
        self.unfold_sentences = unfold_sentences
        self.sample_len = 0
        ############################# 读取数据文件 ##############################

        video_id_path = Path(self.data_dir) / 'msrvtt_data/MSRVTT_train.9k.csv'
        caption_file = Path(self.data_dir) / 'msrvtt_data/MSRVTT_data.json'
        self.features_path = Path(self.data_dir) / 'frames_30fps'

        # video_id_path_dict = {}
        # video_id_path_dict["train"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_train.9k.csv")
        # video_id_path_dict["val"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_JSFUSION_test.csv")
        # video_id_path_dict["test"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_JSFUSION_test.csv")
        # caption_file = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_data.json")

        self.csv = pd.read_csv(video_id_path)
        self.caption = json.load(open(caption_file, 'r'))


        if self.unfold_sentences:   # this
            train_video_ids = list(self.csv['video_id'].values)
            self.sentences_dict = {}
            for itm in self.caption['sentences']:
                if itm['video_id'] in train_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (
                        itm['video_id'], itm['caption'])
            self.sentence_num = self.sample_len = len(self.sentences_dict)
            self.video_num = len(train_video_ids)


        else:
            num_sentences = 0
            self.sentences = defaultdict(list)
            s_video_id_set = set()
            for itm in self.caption['sentences']:
                self.sentences[itm['video_id']].append(itm['caption'])
                num_sentences += 1
                s_video_id_set.add(itm['video_id'])

            # Use to find the clips in the same video
            self.parent_ids = {}
            self.children_video_ids = defaultdict(list)
            for itm in self.caption['videos']:
                vid = itm["video_id"]
                url_posfix = itm["url"].split("?v=")[-1]
                self.parent_ids[vid] = url_posfix
                self.children_video_ids[url_posfix].append(vid)
            self.sample_len = len(self.csv)

            self.sentence_num = num_sentences
            self.video_num = len(s_video_id_set)




        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractorCV2_Image(
            num_segments=max_frames, size=image_resolution, random_shift=True,
            strategy=self.strategy)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}
        self.load_video = load_video

    @property
    def dataset_name(self):
        return 'MSRVTT'

    def __len__(self):
        return self.sample_len


    def __getitem__(self, idx):
        if self.unfold_sentences:   # this
            video_id, caption = self.sentences_dict[idx]
        else:
            video_id, caption = self.csv['video_id'].values[idx], None

        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawframes(choice_video_ids)
        # video, video_mask = self._get_rawvideo(choice_video_ids)
        video_index = self._extract_number(video_id)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, idx, video_index

    def _extract_number(self, v_id):
        match = re.search(r'\d+$', v_id)
        return int(match.group()) if match else None

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_text_len), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            if caption is not None:
                words = self.tokenizer.tokenize(caption)
            else:
                words = self._get_single_text(video_id)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_text_len - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_text_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_text_len
            assert len(input_mask) == self.max_text_len
            assert len(segment_ids) == self.max_text_len

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_frame_pos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_frame_pos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice
                video_slice = self.rawVideoExtractor.process_frame_order(video_slice,
                                                                         frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawframes(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size),
                         dtype=np.float64)  # (1, 8, 1, 3, 224, 224)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)  #
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_frame_pos == 0:  # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_frame_pos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:  # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawFramesExtractor.process_frame_order(video_slice,
                                                                          frame_order=self.frame_order)  # (8, 1, 3, 224, 224)

                try:
                    assert video_slice.shape == (
                    self.max_frames, 1, 3, self.rawFramesExtractor.size, self.rawFramesExtractor.size)
                except:
                    print(f'wrong frame in {choice_video_ids}')
                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask



class MSRVTT_EvalDataset(Dataset):
    def __init__(
            self,
            subset: str,
            data_dir: Union[str, Path],
            features_dir: Union[str, Path],
            max_text_len:int = 40,
            feature_framerate: Union[int, float] = 1.0,
            max_frames: Union[int, float] = 100,
            image_resolution: int = 224,
            frame_order: int = 0,
            slice_frame_pos: int = 0,
            strategy: int = 1,
            load_video = None,
            tokenizer = None,
            return_images = False,
    ):
        super().__init__()
        assert subset in ["val", "test"]
        self.subset = subset

        self.data_dir = data_dir
        # self.features_dir = features_dir

        self.feature_framerate = feature_framerate
        self.max_frames = max_frames

        self.max_text_len = max_text_len
        self.image_resolution = image_resolution

        # self.transforms = transforms
        self.tokenizer = tokenizer
        self.return_images = return_images

        # 0: ordinary order; 1: reverse order; 2: random order.
        assert frame_order in [0, 1, 2]
        self.frame_order = frame_order
        # 0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.
        assert slice_frame_pos in [0, 1, 2]
        self.slice_frame_pos = slice_frame_pos

        self.strategy = strategy
        print(subset, ' strategy====', strategy)

        ############################# 读取数据文件 ##############################
        # video_id_path_dict = {}
        # video_id_path_dict["train"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_train.9k.csv")
        # video_id_path_dict["val"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_JSFUSION_test.csv")
        # video_id_path_dict["test"] = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_JSFUSION_test.csv")

        # caption_file = os.path.join(self.data_dir, "/msrvtt_data/MSRVTT_data.json")


        video_id_path = Path(self.data_dir) / 'msrvtt_data/MSRVTT_JSFUSION_test.csv'
        # caption_file = Path(self.data_dir) / 'msrvtt_data/MSRVTT_data.json'
        self.features_path = Path(self.data_dir) / 'frames_30fps'

        self.csv = pd.read_csv(video_id_path)
        # self.caption = json.load(open(caption_file, 'r'))

        self.sentence_num = self.video_num = len(self.csv)

        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractorCV2_Image(
            num_segments=max_frames, size=image_resolution, random_shift=False,
            strategy=self.strategy, return_images=return_images)  # return_images 返回原始图片做可视化！！

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}

    @property
    def dataset_name(self):
        return 'MSRVTT'

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        video_id = self.csv['video_id'].values[idx]
        sentence = self.csv['sentence'].values[idx]
        # video_id = list(self.data.keys())[idx]
        # sentence = self.data[video_id]['gt']
        # title = self.data[video_id]['titles']


        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, sentence)
        video, video_mask, raw_images = self._get_rawframes(choice_video_ids)
        video_index = self._extract_number(video_id)
        if self.return_images:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, idx, video_index, raw_images
        else:
            return pairs_text, pairs_mask, pairs_segment, video, video_mask, idx, video_index

    def _extract_number(self, v_id):
        match = re.search(r'\d+$', v_id)
        return int(match.group()) if match else None

    def _get_text(self, video_id, sentence):
        choice_video_ids = [video_id]
        n_caption = len(choice_video_ids)

        k = n_caption
        pairs_text = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_text_len), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(sentence)

            words = [self.SPECIAL_TOKEN["CLS_TOKEN"]] + words
            total_length_with_CLS = self.max_text_len - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + [self.SPECIAL_TOKEN["SEP_TOKEN"]]

            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            while len(input_ids) < self.max_text_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
            assert len(input_ids) == self.max_text_len
            assert len(input_mask) == self.max_text_len
            assert len(segment_ids) == self.max_text_len

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)

        return pairs_text, pairs_mask, pairs_segment, choice_video_ids

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}.mp4".format(video_id))
            if os.path.exists(video_path) is False:
                video_path = video_path.replace(".mp4", ".webm")

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawVideoExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_frame_pos == 0:
                        video_slice = raw_video_slice[:self.max_frames, ...]
                    elif self.slice_frame_pos == 1:
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                    else:
                        sample_indx = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_indx, ...]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawVideoExtractor.process_frame_order(video_slice, frame_order=self.frame_order)

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask

    def _get_rawframes(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x T x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            # Individual for YoucokII dataset, due to it video format
            video_path = os.path.join(self.features_path, "{}".format(video_id))  # folder

            raw_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_data['video']
            raw_images = raw_data['raw_images']
            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x T x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data_clip)
                if self.max_frames < raw_video_slice.shape[0]:
                    if self.slice_frame_pos == 0:  # cut from head
                        video_slice = raw_video_slice[:self.max_frames, ...]
                        if raw_images:
                            raw_images = raw_images[:self.max_frames]
                    elif self.slice_frame_pos == 1:  # cut from tail
                        video_slice = raw_video_slice[-self.max_frames:, ...]
                        if raw_images:
                            raw_images = raw_images[-self.max_frames:]
                    else:  # extract uniformly
                        sample_index = np.linspace(0, raw_video_slice.shape[0] - 1, num=self.max_frames, dtype=int)
                        video_slice = raw_video_slice[sample_index, ...]
                        if raw_images:
                            raw_images = [raw_images[i] for i in sample_index]
                else:
                    video_slice = raw_video_slice

                video_slice = self.rawFramesExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                try:
                    assert video_slice.shape == (
                    self.max_frames, 1, 3, self.rawFramesExtractor.size, self.rawFramesExtractor.size)
                except:
                    logger.info(f'wrong frame in {choice_video_ids}')

                slice_len = video_slice.shape[0]
                max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_len else slice_len
                if slice_len < 1:
                    pass
                else:
                    video[i][:slice_len, ...] = video_slice
            else:
                print("video path: {} error. video id: {}".format(video_path, video_id))

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        return video, video_mask, raw_images


    def vis_collate_fn(self, batch):
        import torch
        input_ids, input_mask, segment_ids, video, video_mask, index, v_index, raw_images = batch[0]
        return [torch.tensor(input_ids).unsqueeze(0),
                torch.tensor(input_mask).unsqueeze(0),
                torch.tensor(segment_ids).unsqueeze(0),
                torch.tensor(video).unsqueeze(0),
                torch.tensor(video_mask).unsqueeze(0),
                torch.tensor(index),
                torch.tensor(v_index),
                raw_images]