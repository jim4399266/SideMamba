from torch.utils.data import Dataset
import numpy as np
import pickle
import os
from pathlib import Path
from typing import Union
import sys
sys.path.append('../..')
# from ..transforms import keys_to_transforms
from src.datasets.rawframes_util import RawFramesExtractorCV2
from src.datasets.rawvideo_util import RawVideoExtractor

class MSVDDataset(Dataset):
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
    ):
        super().__init__()
        assert subset in ["train", "val", "test"]
        self.subset = subset

        self.data_dir = data_dir
        self.features_dir = features_dir

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

        ############################# 读取数据文件 ##############################
        video_id_path_dict = {}
        video_id_path_dict["train"] = os.path.join(self.data_dir, "train_list.txt")
        video_id_path_dict["val"] = os.path.join(self.data_dir, "val_list.txt")
        video_id_path_dict["test"] = os.path.join(self.data_dir, "test_list.txt")
        caption_file = os.path.join(self.data_dir, "raw-captions.pkl")

        with open(video_id_path_dict[self.subset], 'r') as fp:
            video_ids = [itm.strip() for itm in fp.readlines()]

        with open(caption_file, 'rb') as f:
            captions = pickle.load(f)

        video_dict = {}
        for root, dub_dir, video_files in os.walk(self.features_dir):
            # for video_file in video_files:
            for video_file in dub_dir:
                video_id_ = video_file

                if video_id_ not in video_ids:
                    continue
                file_path_ = os.path.join(root, video_file)
                video_dict[video_id_] = file_path_
        self.video_dict = video_dict

        self.sample_len = 0
        self.sentences_dict = {}
        self.cut_off_points = []
        for video_index, video_id in enumerate(video_ids):
            assert video_id in captions
            for cap in captions[video_id]:
                cap_txt = " ".join(cap)
                self.sentences_dict[len(self.sentences_dict)] = (video_index, video_id, cap_txt)
            self.cut_off_points.append(len(self.sentences_dict))

        ## below variables are used to multi-sentences retrieval
        # self.cut_off_points: used to tag the label when calculate the metric
        # self.sentence_num: used to cut the sentence representation
        # self.video_num: used to cut the video representation

        self.multi_sentence_per_video = True  # !!! important tag for eval
        if self.subset == "val" or self.subset == "test":
            self.sentence_num = len(self.sentences_dict)
            self.video_num = len(video_ids)
            assert len(self.cut_off_points) == self.video_num
            print("For {}, sentence number: {}".format(self.subset, self.sentence_num))
            print("For {}, video number: {}".format(self.subset, self.video_num))

        print("Video number: {}".format(len(self.video_dict)))
        print("Total Paire: {}".format(len(self.sentences_dict)))

        self.sample_len = len(self.sentences_dict)
        self.rawVideoExtractor = RawVideoExtractor(framerate=feature_framerate, size=image_resolution)
        self.rawFramesExtractor = RawFramesExtractorCV2(
            num_segments=max_frames, size=image_resolution, random_shift=False,
            strategy=self.strategy if self.subset == 'train' else 1)

        self.SPECIAL_TOKEN = {"CLS_TOKEN": "<|startoftext|>", "SEP_TOKEN": "<|endoftext|>",
                              "MASK_TOKEN": "[MASK]", "UNK_TOKEN": "[UNK]", "PAD_TOKEN": "[PAD]"}


    def __len__(self):
        return self.sample_len


    def __getitem__(self, index):
        video_index, video_id, caption = self.sentences_dict[index]
        pairs_text, pairs_mask, pairs_segment, choice_video_ids = self._get_text(video_id, caption)
        video, video_mask = self._get_rawframes(choice_video_ids)
        return pairs_text, pairs_mask, pairs_segment, video, video_mask, index, video_index


    def _get_text(self, video_id, caption):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_text_len), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_text_len), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):
            words = self.tokenizer.tokenize(caption)

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

    def _get_rawframes(self, choice_video_ids):

        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x max_frames x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawFramesExtractor.size, self.rawFramesExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawFramesExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                # L x max_frames x 3 x H x W
                raw_video_slice = self.rawFramesExtractor.process_raw_data(raw_video_data)
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

                video_slice = self.rawFramesExtractor.process_frame_order(video_slice, frame_order=self.frame_order)
                try:
                    assert video_slice.shape == (
                    self.max_frames, 1, 3, self.rawFramesExtractor.size, self.rawFramesExtractor.size)

                except Exception as e:
                    print(f'wrong frame in {choice_video_ids} -> {e}')


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

    def _get_rawvideo(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        # Pair x L x max_frames x 3 x H x W
        video = np.zeros((len(choice_video_ids), self.max_frames, 1, 3,
                          self.rawVideoExtractor.size, self.rawVideoExtractor.size), dtype=np.float64)

        for i, video_id in enumerate(choice_video_ids):
            video_path = self.video_dict[video_id]

            raw_video_data = self.rawVideoExtractor.get_video_data(video_path)
            raw_video_data = raw_video_data['video']

            if len(raw_video_data.shape) > 3:
                raw_video_data_clip = raw_video_data
                # L x max_frames x 3 x H x W
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



    # def collate(self, batch, mlm_collator=None):
    #     keys = set([key for b in batch for key in b.keys()])
    #     # 将batch中属于同一个keys的信息放到一起
    #     dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}
    #     # ==================================== 整理图片 ====================================
    #     # 取出与image相关的keys
    #     img_keys = ['image']
    #     # 并且将dict_batch中的list转换为tensor
    #     for img_key in img_keys:
    #         imgs = [img[0] for img in dict_batch[img_key]]
    #         new_images = torch.stack(imgs, dim=0)
    #         dict_batch[img_key] = new_images
    #     dict_batch['image_index'] = torch.tensor(dict_batch['image_index'], dtype=torch.long)
    #
    #     # ==================================== 整理文本 ====================================
    #     encodings = {}
    #     e_keys = set([key for b in dict_batch['text_encodings'] for key in b.keys()])
    #     for k in e_keys:
    #         encodings[k] = torch.cat([dic[k] if k in dic else None for dic in dict_batch['text_encodings']], dim=0)
    #     dict_batch['text_encodings'] = encodings
    #     text_list_index = [i for index in dict_batch['text_list_index'] for i in index]
    #     dict_batch['text_list_index'] = torch.tensor(text_list_index, dtype=torch.long)
    #     return dict_batch


