from torch import nn
import torch
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Union
import torch.nn.functional as F

from .retrieval_utils import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim

from .module_base import BaseModule
from .module_retrieval_clip import CLIPRetrievalModule


from src.models.eva_clip.model import CustomCLIP
from src.models.eva_clip.factory import create_model, get_model_config

from src.models.module_utils import EncoderOutput, EvalCacheOutput, EvalReorderedOutput, DynamicTanh
from src.models.optimization import BertAdam

from src.models.module_vmamba import SideVMamba


class EVACLIPRetrievalModule(CLIPRetrievalModule, BaseModule):
    def __init__(self, config):
        BaseModule.__init__(self)

        self.save_hyperparameters()
        self.config = config

        encoder_config = config['encoder']

        # if encoder_config['freeze_text_encoder']:
        #     side_net_t = self.create_side_net(config['side_t'], modal='t')
        # else:
        #     side_net_t = None

        side_net_t = None
        side_net_v = self.create_side_net(config['side'])

        if config['side']['use_dyt']:
            side_net_v = DynamicTanh.convert_ln_to_dyt(side_net_v)


        self.clip = self.create_clip(config, side_net_v)

        self.loose_type = True

        self.use_queue = False


        self.text_weight_fc = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']), nn.ReLU(inplace=True),
            nn.Linear(encoder_config['embed_dim'], 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']), nn.ReLU(inplace=True),
            nn.Linear(encoder_config['embed_dim'], 1))

        self.loss_fct = nn.CrossEntropyLoss()
        # self.loss_fct = CrossEn()
        # self.mse_loss = nn.MSELoss()

        self.apply(self.init_weights)


    def create_clip(self, config, side_network_v, side_network_t=None):


        clip = create_model(config['encoder']['pretrained_clip_name'], str(config['encoder']['pretrained_clip']),
            force_custom_clip=True, T=config['dataset']['max_frames'] , side_network_v=side_network_v)

        return clip.float()

    @classmethod
    def from_pretrained(cls, config, state_dict=None):
        pretrained_clip_name = config['encoder']["pretrained_clip_name"]
        model_path = config['encoder']["pretrained_clip"]

        model_name = pretrained_clip_name.replace('/', '-')  # for callers using old naming with / in ViT names
        model_cfg = get_model_config(model_name)

        config['encoder']['embed_dim'] = model_cfg['embed_dim']
        config['side'].update({'trans_dim': model_cfg['vision_cfg']['width']})
        config['side_t'].update({'trans_dim': model_cfg['text_cfg']['width']})

        model = cls(config)

        # if config['encoder']['freeze_text_encoder'] == True:
        #     for name, param in model.named_parameters():
        #         if 'clip.transformer.' in name and 'side' not in name:
        #             param.requires_grad_(False)

        if config['encoder']['freeze_text_encoder'] == True:
            freeze_text_layer_num = config['encoder']['freeze_text_layer_num']
            if isinstance(freeze_text_layer_num, float):
                freeze_text_layer_num = int(config['encoder']['transformer_layers'] * freeze_text_layer_num)
            for name, param in model.named_parameters():
                if 'token_embedding' in name:
                    param.requires_grad_(False)

                if 'clip.transformer.' in name and 'side' not in name:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num < freeze_text_layer_num:
                        param.requires_grad_(False)


        if config['encoder']['freeze_vit_encoder'] == True:
            for name, param in model.named_parameters():
                if 'visual' in name and 'side' not in name and 'visual.head' not in name and 'visual.norm' not in name:
                    param.requires_grad_(False)

        assert config['encoder']['freeze_layer_num'] <= 12 and config['encoder']['freeze_layer_num'] >= -1
        if hasattr(model, "clip") and config['encoder']['freeze_layer_num'] > -1:
            for name, param in model.clip.named_parameters():
                # top layers always need to train
                if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                        or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0 or name.find(
                    'visual.head') == 0 or name.find('visual.norm') == 0 or name.find('text.ln_final') == 0:
                    param.requires_grad = True
                    continue  # need to train
                elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num >= config['encoder']['freeze_layer_num']:
                        continue  # need to train

        return model


    #   ============================  模型基础编码方法  ===================================
    def get_sequence_output(self, input_ids, input_mask, segment_ids, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1]).contiguous()
            input_mask = input_mask.view(-1, input_mask.shape[-1]).contiguous()
            segment_ids = segment_ids.view(-1, segment_ids.shape[-1]).contiguous()

        # with torch.amp.autocast('cuda', dtype=torch.float16):
        text_output = self.clip.encode_text(input_ids, return_all_features=True)

        return EncoderOutput(
            pooler_output=text_output[0].to(self.head_dtype),
            last_hidden_state=text_output[1].to(self.head_dtype)
        )

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1]).contiguous()
            video = torch.as_tensor(video).float().contiguous()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w).contiguous()
            video_frame = bs * ts

        # with torch.amp.autocast('cuda', dtype=torch.float16):
        visual_output = self.clip.encode_image(video)

        return EncoderOutput(
            pooler_output=visual_output.to(self.head_dtype),
            # last_hidden_state=visual_output.last_hidden_state.to(self.head_dtype),
            # hidden_states=visual_output.hidden_states.to(self.head_dtype),
        )


    def get_sequence_visual_output(self, input_ids, input_mask, segment_ids, video, video_mask, shaped=False,
                                   video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1]).contiguous()
            input_mask = input_mask.view(-1, input_mask.shape[-1]).contiguous()
            segment_ids = segment_ids.view(-1, segment_ids.shape[-1]).contiguous()
            video_mask = video_mask.view(-1, video_mask.shape[-1]).contiguous()

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, input_mask, segment_ids, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)
        # visual_output = self.side_vim(video, visual_output)

        return sequence_output, visual_output