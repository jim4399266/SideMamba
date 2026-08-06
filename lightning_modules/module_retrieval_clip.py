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


from src.models.module_clip import CLIP
from src.models.module_utils import EncoderOutput, EvalCacheOutput, EvalReorderedOutput, DynamicTanh
from src.models.optimization import BertAdam

from src.models.module_vmamba import SideVMamba

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

class CLIPRetrievalModule(BaseModule):
    '''
    In this class, we just build the model structure for retrieval with CLIP. The training and evaluating steps will be set in lightning_modules.
    '''
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.ignore_video_index = -1
        self.config = config
        encoder_config = config['encoder']

        self._stage_one = True
        self._stage_two = False
        self.queue_size = config['batch_size'] * config['queue_size_ratio']
        self.use_queue = config['use_queue']
        self.logit_scale = nn.Parameter(torch.tensor([np.log(150)]))

        self.visual_hidden = config['visual_hidden']
        self.visual_all_hidden = config['visual_all_hidden']

        self.loose_type = False
        if self._stage_one and encoder_config.get('loose_type', None):
            self.loose_type = True
            print("Test retrieval by loose type.")


        side_net_t = None
        side_net_v = self.create_side_net(config['side'])

        if config['side']['use_dyt']:
            side_net_v = DynamicTanh.convert_ln_to_dyt(side_net_v)

        self.clip = self.create_clip(encoder_config, side_net_v, side_net_t)

        self.sim_header = config['sim_header'].lower()
        # show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        print("\t sim_header: {}".format(self.sim_header))


        self.text_weight_fc = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']), nn.ReLU(inplace=True),
            nn.Linear(encoder_config['embed_dim'], 1))
        self.video_weight_fc = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']), nn.ReLU(inplace=True),
            nn.Linear(encoder_config['embed_dim'], 1))
        # # Temperature for query-aware soft frame/token selection. A smaller value is closer to hard max.
        # self.query_ware_temperature = config.get('query_ware_temperature', 0.07)
        # # WTI-TopK keeps the original hard match and adds a small amount of multi-frame/token evidence.
        # self.wti_topk = config.get('wti_topk', 3)
        # self.wti_topk_ratio = config.get('wti_topk_ratio', 0.2)
        # # Learnable monotonic rank gaps; softmax of the resulting logits starts near [0.5, 0.3, 0.2].
        # self.wti_topk_weight_deltas = nn.Parameter(
        #     torch.tensor([-0.4055, -0.6931], dtype=torch.float32)
        # )
        # # Temporal segment kernel: center frame contributes most, adjacent frames provide context.
        # self.wti_segment_kernel = config.get('wti_segment_kernel', [0.25, 0.5, 0.25])

        self.loss_fct = CrossEn()
        self.apply(self.init_weights)


    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    @property
    def head_dtype(self):
        return self.text_weight_fc[0].weight.dtype

    def create_clip(self, encoder_config, side_network_v, side_network_t):
        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        embed_dim = encoder_config['embed_dim']
        image_resolution = encoder_config['image_resolution']
        vision_layers = encoder_config['vision_layers']
        vision_width = encoder_config['vision_width']
        vision_patch_size = encoder_config['vision_patch_size']
        context_length = encoder_config['context_length']  # clip transformer中的文本长度
        # context_length = encoder_config['max_text_len']       # 自定义数据集文本长度
        vocab_size = encoder_config['vocab_size']
        transformer_width = encoder_config['transformer_width']
        transformer_heads = encoder_config['transformer_heads']
        transformer_layers = encoder_config['transformer_layers']

        # cut_top_layer = encoder_config['cut_top_layer']
        linear_patch = encoder_config['linear_patch']

        # max_frames = encoder_config['max_frames']

        # self.transformer_width = transformer_width
        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        return CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
            linear_patch=linear_patch, side_network_v=side_network_v, side_network_t=side_network_t,
        ).float()



    def create_side_net(self, side_config, modal='v'):
        if modal == 'v':
            if side_config['network'] in ['vmamba', 'vmamba2']:
                return SideVMamba(
                    **side_config
                )
            else:
                return None
        else:
            raise NotImplementedError

    #   ============================  加载模型方法  ===================================
    @classmethod
    def from_pretrained(cls, config, state_dict=None):
        # pretrained_clip_name = config['encoder']["pretrained_clip_name"]
        # model_path = Path(config['pretrained_model_dir']) / config['encoder']["pretrained_clip"]

        pretrained_clip_name = config['encoder']["pretrained_clip_name"]
        model_path = config['encoder']["pretrained_clip"]

        clip_state_dict = CLIP.load_clip_state_dict(pretrained_clip_name, model_path)

        config['encoder'] = CLIP.get_clip_config(clip_state_dict, config['encoder'])

        config['side'].update({'trans_dim': config['encoder']['vision_width']})
        config['side_t'].update({'trans_dim': config['encoder']['transformer_width']})


        model = cls(config)

        if state_dict is None: state_dict = {}
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict)

        ## ####################################
        # freeze testing, freeze some layers
        ## ####################################

        if config['encoder']['freeze_text_encoder'] == True:
            freeze_text_layer_num = config['encoder']['freeze_text_layer_num']
            if isinstance(freeze_text_layer_num, float):
                freeze_text_layer_num = int(config['encoder']['transformer_layers'] * freeze_text_layer_num)
            for name, param in model.named_parameters():
                if 'clip.transformer.' in name and 'side' not in name:
                    layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                    if layer_num < freeze_text_layer_num:
                        param.requires_grad_(False)
                if 'token_embedding' in name:
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

    def from_config(cls, config, state_dict=None):
        pretrained_clip_name = config['encoder']["pretrained_clip_name"]
        model_path = config['encoder']["pretrained_clip"]

        clip_state_dict = CLIP.load_clip_state_dict(pretrained_clip_name, model_path)

        config['encoder'] = CLIP.get_clip_config(clip_state_dict, config['encoder'])

        config['side'].update({'trans_dim': config['encoder']['vision_width']})
        config['side_t'].update({'trans_dim': config['encoder']['transformer_width']})

        model = cls(config)

        return model

    @classmethod
    def from_checkpoint(cls, config, strict=True):
        pretrained_clip_name = config['encoder']["pretrained_clip_name"]
        model_path = config['encoder']["pretrained_clip"]
        clip_state_dict = CLIP.load_clip_state_dict(pretrained_clip_name, model_path)
        config['encoder'] = CLIP.get_clip_config(clip_state_dict, config['encoder'])
        config['side'].update({'trans_dim': config['encoder']['vision_width']})
        config['side_t'].update({'trans_dim': config['encoder']['transformer_width']})

        model = cls(config)

        checkpoint = torch.load(str(config['checkpoint']), map_location='cpu')
        state_dict = checkpoint['state_dict']
        msg = model.load_state_dict(state_dict, strict=False)
        print("missing keys:")
        print(msg.missing_keys)
        return model

    #   ============================  初始化优化器方法  ===================================

    def configure_optimizers(self):
        opt_config = self.config['optimizer']
        max_steps, warmup_steps = self.cal_steps()


        # Original no-weight-decay parameter list:
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # NEW: Do not decay fusion logits toward zero, which would force their gates back to 0.5.
        no_decay = [
            'bias', 'LayerNorm.bias', 'LayerNorm.weight', 'side_fusion_logits',
            'wti_topk_weight_deltas',
        ]
        decay_param_tp = [(n, p) for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        no_decay_param_tp = [(n, p) for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]

        # # NEW: Collect fusion gates explicitly so their learning rate and decay are unambiguous.
        # fusion_gate_params = [(n, p) for n, p in self.named_parameters() if 'side_fusion_logits' in n]
        # wti_topk_weight_deltas = [(n, p) for n, p in self.named_parameters() if 'wti_topk_weight_deltas' in n]
        hierarchical_weights_tp = [(n, p) for n, p in decay_param_tp if 'hierarchical' in n]

        decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if
                               "clip." in n and 'side' not in n and 'hierarchical' not in n
                               and 'side_fusion_logits' not in n and 'wti_topk_weight_deltas' not in n]
        decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if
                                 ("clip." not in n or 'side' in n and 'hierarchical' not in n)
                                 and 'side_fusion_logits' not in n and 'wti_topk_weight_deltas' not in n]

        no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if
                                  "clip." in n and 'side' not in n and 'hierarchical' not in n
                                  and 'side_fusion_logits' not in n and 'wti_topk_weight_deltas' not in n]
        no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if
                                    ("clip." not in n or 'side' in n and 'hierarchical' not in n)
                                    and 'side_fusion_logits' not in n and 'wti_topk_weight_deltas' not in n]



        optimizer_grouped_parameters = [
            {   # backbone的参数
                "name": 'backbone',
                "params": [p for n, p in decay_clip_param_tp],
                "lr": opt_config['init_lr'] * opt_config['coef_lr'],
            },
            {   # backbone的参数
                "name": 'backbone_no_decay',
                "params": [p for n, p in no_decay_clip_param_tp],
                "lr": opt_config['init_lr'] * opt_config['coef_lr'],
                "weight_decay": 0.0,
            },

            {   # 除了backbone之外的参数，使用默认学习率
                "name": 'no_backbone',
                "params": [p for n, p in decay_noclip_param_tp],
            },
            {  # 除了backbone之外的参数，使用默认学习率
                "name": 'no_backbone_no_decay',
                "params": [p for n, p in no_decay_noclip_param_tp],
                "weight_decay": 0.0,
            },
            {
                "name": 'hierarchical_weights',
                "params": [p for n, p in hierarchical_weights_tp],
                "lr": 1e-2,
                "weight_decay": 0.0,
            },
            # {  # NEW: Train Side/CLIP fusion gates with the normal Side learning rate.
            #     "name": 'fusion_gate_params',
            #     "params": [p for n, p in fusion_gate_params],
            #     # "lr": opt_config['init_lr'],
            #     "lr": 1e-2,
            #     "weight_decay": 0.0,
            # },
            # {  # NEW: Train Side/CLIP fusion gates with the normal Side learning rate.
            #     "name": 'wti_topk_weight_deltas',
            #     "params": [p for n, p in wti_topk_weight_deltas],
            #     # "lr": opt_config['init_lr'],
            #     "lr": 1e-2,
            #     "weight_decay": 0.0,
            # },

        ]


        if self.config['optimizer']['name'] == 'adamw':
            optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                          lr=opt_config['init_lr'],
                                          weight_decay=opt_config['weight_decay'],
                                          eps=opt_config['eps'],
                                          betas=opt_config['betas'])

        else:
            optimizer = BertAdam(optimizer_grouped_parameters,
                                 lr=opt_config['init_lr'],
                                 # warmup=args.warmup_proportion,
                                 # schedule='warmup_cosine',
                                 b1=0.9, b2=0.98, e=1e-6,
                                 # t_total=num_train_optimization_steps,
                                 weight_decay=opt_config['weight_decay'],
                                 max_grad_norm=1.0)
        # print(f'----------- Optimizer: {optimizer} ------------')

        sched = self.get_scheduler(optimizer, warmup_steps, max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': sched,
        }

#   ============================  模型基础编码方法  ===================================

    def get_sequence_output(self, input_ids, input_mask, segment_ids, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1]).contiguous()
            input_mask = input_mask.view(-1, input_mask.shape[-1]).contiguous()
            segment_ids = segment_ids.view(-1, segment_ids.shape[-1]).contiguous()

        # with torch.amp.autocast('cuda', dtype=torch.float16):
        text_output = self.clip.encode_text(input_ids, return_hidden=True)

        return EncoderOutput(
            pooler_output=text_output.pooler_output.to(self.head_dtype),
            last_hidden_state=text_output.last_hidden_state.to(self.head_dtype)
        )

    def get_visual_output(self, video, video_mask, shaped=False, return_hidden=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1]).contiguous()
            video = torch.as_tensor(video).float().contiguous()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w).contiguous()
            video_frame = bs * ts

        # with torch.amp.autocast('cuda', dtype=torch.float16):
        visual_output = self.clip.encode_image(video, return_hidden=return_hidden, video_frame=video_frame)

        return EncoderOutput(
            pooler_output=visual_output.pooler_output.to(self.head_dtype),
            last_hidden_state=None if not return_hidden else visual_output.last_hidden_state.to(self.head_dtype),
        )


    def get_sequence_visual_output(self, input_ids, input_mask, segment_ids, video, video_mask, shaped=False,
                                   visual_return_hidden=False, video_frame=-1):
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
        visual_output = self.get_visual_output(video, video_mask, shaped=True, return_hidden=visual_return_hidden, video_frame=video_frame)
        # visual_output = self.side_vim(video, visual_output)
        return sequence_output, visual_output

    # def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
    #     attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
    #     attention_mask_un[:, 0, :] = 0.
    #     sequence_output = sequence_output * attention_mask_un
    #     text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
    #     return text_out
    #
    # def _mean_pooling_for_similarity_visual(self, visual_output, video_mask, ):
    #     video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
    #     visual_output = visual_output * video_mask_un
    #     video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
    #     video_mask_un_sum[video_mask_un_sum == 0.] = 1.
    #     video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
    #     return video_out

    # @torch.no_grad()
    # def _dequeue_and_enqueue(self, text_feat, image_feat, text_mask, video_mask, idxs):
    #     # gather keys before updating queue
    #     text_feats = self.concat_all_gather(text_feat)  # btd
    #     image_feats = self.concat_all_gather(image_feat)  # btd
    #     text_mask = self.concat_all_gather(text_mask)  # bt
    #     video_mask = self.concat_all_gather(video_mask)  # bt
    #     idxs = self.concat_all_gather(idxs)  # 1b
    #     self.synchronize()  # force sync
    #
    #     batch_size = image_feats.shape[0]
    #
    #     ptr = int(self.queue_ptr)
    #     # assert self.queue_size % batch_size == 0  # for simplicity
    #
    #     step = batch_size if (ptr + batch_size <= self.queue_size) else (self.queue_size - ptr)
    #
    #     # replace the keys at ptr (dequeue and enqueue)
    #
    #     self.text_queue[ptr:ptr + step, ...] = text_feats[:step]
    #     self.visual_queue[ptr:ptr + step, ...] = image_feats[:step]
    #
    #     self.text_mask_queue[ptr:ptr + step, ...] = text_mask[:step]
    #     self.visual_mask_queue[ptr:ptr + step, ...] = video_mask[:step]
    #
    #     self.idx_queue[..., ptr:ptr + step] = idxs[:, :step]
    #     # ptr = (ptr + batch_size) % self.queue_size  # move pointer
    #     self.queue_ptr[0] = (ptr + step) % self.queue_size
    #

    #   ============================  模型交互方法  ===================================
    # def _get_logits_cls(self, sequence_pooler, visual_pooler, text_mask, video_mask):
    #     ### sequence_output [B N D]        visual_output [B T L D]
    #
    #     sequence_pooler = sequence_pooler / sequence_pooler.norm(dim=-1, keepdim=True)  # [B N D]
    #     visual_pooler = visual_pooler / visual_pooler.norm(dim=-1, keepdim=True)  # [B T L D]
    #
    #     # 利用CLS token 获取每条文本和每张图片帧的相似度
    #     retrieve_logits = torch.einsum('ad, bvd -> abv', sequence_pooler, visual_pooler)
    #     # 用最相似的frame代表视频，获得与文本的相似度
    #     retrieve_logits, _ = retrieve_logits.max(dim=-1)  # a b
    #
    #
    #     logit_scale = self.clip.logit_scale.exp()
    #     # logit_scale = self.logit_scale.exp()
    #     retrieve_logits = logit_scale * retrieve_logits
    #     return retrieve_logits, retrieve_logits.T


    # def _get_logits_wti_v2(self, sequence_pooler, sequence_hidden, visual_hidden, text_mask, video_mask):
    #     ### sequence_output [B N D]        visual_output [B T L D]
    #
    #     text_weight = self.text_weight_fc(sequence_hidden).squeeze()  # B N D -> B N
    #     text_weight.masked_fill_((1 - text_mask).clone().detach().to(torch.bool), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)  # B N
    #
    #
    #     video_weight = self.video_weight_fc(visual_hidden).squeeze()  # B T L D -> B T L
    #     video_weight.masked_fill_((1 - video_mask.unsqueeze(-1)).clone().detach().to(torch.bool), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)  # B T L
    #
    #
    #
    #     sequence_hidden = sequence_hidden / sequence_hidden.norm(dim=-1, keepdim=True)  # [B N D]
    #     sequence_pooler = sequence_pooler / sequence_pooler.norm(dim=-1, keepdim=True)
    #
    #     visual_hidden = visual_hidden / visual_hidden.norm(dim=-1, keepdim=True)  # [B T L D]
    #     visual_pooler = visual_hidden[..., 0, :]
    #
    #
    #     # 利用CLS token 获取每条文本和每张图片帧的相似度
    #     cls_logits = torch.einsum('ad, bvd -> abv', sequence_pooler, visual_pooler)
    #     # 用最相似的frame代表视频，获得与文本的相似度
    #     cls_logits, _ = cls_logits.max(dim=-1)  # a b
    #
    #     # 利用 hidden state 获取每句话中每个token与每个图片帧中每个patch的相似度
    #     retrieve_logits = torch.einsum('atd,bvld->abtvl', [sequence_hidden, visual_hidden])
    #     retrieve_logits = torch.einsum('abtvl,at->abtvl', [retrieve_logits, text_mask])
    #     retrieve_logits = torch.einsum('abtvl,bv->abtvl', [retrieve_logits, video_mask])
    #
    #     # max for video token
    #     # #为每句话中每个token找出最相似的一个patch，再根据每个token的weight计算出文本与视频frame的相似度
    #     t2v_logits, _ = retrieve_logits.max(dim=-1)  # abtvl -> abtv
    #     t2v_logits = torch.einsum('abtv,at->abv', [t2v_logits, text_weight])
    #     # 用最相似的frame代表视频，获得与文本的相似度
    #     t2v_logits, _ = t2v_logits.max(dim=-1)  # a b
    #
    #     # #为每个图片帧中每个patch找出最相似的一个文本token，再根据每个patch的weight计算出视频与文本的相似度
    #     v2t_logits, _ = retrieve_logits.max(dim=-3)  # abtvl -> abvl
    #     v2t_logits = torch.einsum('abvl,bvl->abv', [v2t_logits, video_weight])
    #     # 用最相似的frame代表视频，获得与文本的相似度
    #     v2t_logits, _ = v2t_logits.max(dim=-1)  # a b
    #
    #     # retrieve_logits = torch.einsum('atd,bvd->abtv', [sequence_output, visual_output])
    #     # retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
    #     # retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
    #     # text_sum = text_mask.sum(-1)
    #     # video_sum = video_mask.sum(-1)
    #
    #     # # max for video token   #为每句话中每个token找出最相似的一个frame，再根据每个token的weight计算出文本与视频的相似度
    #     # t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
    #     # t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
    #     #
    #     # v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
    #     # v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
    #
    #     retrieve_logits = (cls_logits + t2v_logits + v2t_logits) / 3.0
    #
    #     logit_scale = self.clip.logit_scale.exp()
    #     # logit_scale = self.logit_scale.exp()
    #     retrieve_logits = logit_scale * retrieve_logits
    #     return retrieve_logits, retrieve_logits.T

    def _get_logits_wti(self, sequence_output, visual_output, text_mask, video_mask):
        if self.trainer.world_size > 1 :
            if self.training and torch.cuda.is_available():  # batch merge here
                # print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size before gather: {sequence_output.size()}')
                sequence_output = self.all_gather_with_grad(sequence_output)
                visual_output = self.all_gather_with_grad(visual_output)
                text_mask = self.all_gather_with_grad(text_mask)
                video_mask = self.all_gather_with_grad(video_mask)
                self.synchronize()  # force sync
                # torch.distributed.barrier()  # force sync
                # print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size after gather: {sequence_output.size()}')

        text_weight = self.text_weight_fc(sequence_output).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight.masked_fill_((1 - text_mask).clone().detach().to(torch.bool), float("-inf"))
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc(visual_output).squeeze(2)  # B x N_v x D -> B x N_v
        video_weight.masked_fill_((1 - video_mask).clone().detach().to(torch.bool), float("-inf"))
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        retrieve_logits = torch.einsum('atd,bvd->abtv', [sequence_output, visual_output])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        # text_sum = text_mask.sum(-1)
        # video_sum = video_mask.sum(-1)

        # max for video token   #为每句话中每个token找出最相似的一个frame，再根据每个token的weight计算出文本与视频的相似度
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
        retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        logit_scale = self.clip.logit_scale.exp()
        # logit_scale = self.logit_scale.exp()
        retrieve_logits = logit_scale * retrieve_logits
        return retrieve_logits, retrieve_logits.T

    # def _temporal_segment_scores(self, scores, video_mask):
    #     """Aggregate adjacent frames with boundary- and padding-aware normalization."""
    #     kernel = scores.new_tensor(self.wti_segment_kernel)
    #     if kernel.ndim != 1 or kernel.numel() % 2 == 0 or kernel.numel() == 0:
    #         raise ValueError("wti_segment_kernel must be a non-empty odd-length sequence")
    #     if (kernel < 0).any() or kernel.sum() <= 0:
    #         raise ValueError("wti_segment_kernel must contain non-negative values with a positive sum")
    #     kernel = (kernel / kernel.sum()).view(1, 1, -1)
    #
    #     # scores: [num_text, num_video, num_token, num_frame]
    #     score_shape = scores.shape
    #     frame_mask = video_mask.bool()[None, :, None, :].expand(score_shape)
    #     flat_scores = scores.reshape(-1, 1, score_shape[-1])
    #     flat_mask = frame_mask.reshape(-1, 1, score_shape[-1]).to(scores.dtype)
    #     padding = kernel.shape[-1] // 2
    #
    #     # Convolving the mask supplies the real weight sum at boundaries and near padding.
    #     weighted_sum = F.conv1d(flat_scores * flat_mask, kernel, padding=padding)
    #     valid_weight = F.conv1d(flat_mask, kernel, padding=padding)
    #     segment_scores = weighted_sum / valid_weight.clamp_min(1e-6)
    #
    #     # A padded frame may receive context from a valid neighbor, but it must never become a max center.
    #     segment_scores = segment_scores.masked_fill(~flat_mask.bool(), float("-inf"))
    #     return segment_scores.reshape(score_shape)
    #
    # def _get_logits_wti_segment(self, sequence_output, visual_output, text_mask, video_mask):
    #     """WTI matching over short continuous temporal segments instead of isolated frames."""
    #     if self.trainer.world_size > 1:
    #         if self.training and torch.cuda.is_available():
    #             sequence_output = self.all_gather_with_grad(sequence_output)
    #             visual_output = self.all_gather_with_grad(visual_output)
    #             text_mask = self.all_gather_with_grad(text_mask)
    #             video_mask = self.all_gather_with_grad(video_mask)
    #             self.synchronize()
    #
    #     text_weight = self.text_weight_fc(sequence_output).squeeze(-1)
    #     text_weight = text_weight.masked_fill(~text_mask.bool(), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)
    #
    #     video_weight = self.video_weight_fc(visual_output).squeeze(-1)
    #     video_weight = video_weight.masked_fill(~video_mask.bool(), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)
    #
    #     sequence_output = F.normalize(sequence_output, dim=-1, eps=1e-6)
    #     visual_output = F.normalize(visual_output, dim=-1, eps=1e-6)
    #     token_frame_logits = torch.einsum('atd,bvd->abtv', sequence_output, visual_output)
    #     segment_logits = self._temporal_segment_scores(token_frame_logits, video_mask)
    #
    #     # Text-to-video selects the best continuous segment for every valid text token.
    #     t2v_logits = segment_logits.max(dim=-1).values
    #     t2v_logits = torch.einsum('abt,at->ab', t2v_logits, text_weight)
    #
    #     # Video-to-text matches every segment center to its best valid text token.
    #     token_mask = text_mask.bool()[:, None, :, None]
    #     v2t_logits = segment_logits.masked_fill(~token_mask, float("-inf")).max(dim=-2).values
    #     v2t_logits = v2t_logits.masked_fill(~video_mask.bool()[None, :, :], 0.0)
    #     v2t_logits = torch.einsum('abv,bv->ab', v2t_logits, video_weight)
    #
    #     retrieve_logits = (t2v_logits + v2t_logits) / 2.0
    #     retrieve_logits = self.clip.logit_scale.exp() * retrieve_logits
    #     return retrieve_logits, retrieve_logits.T



    #
    # def _get_logits_query_ware(self, sequence_output, visual_output, text_mask, video_mask):
    #     """Compute query-aware similarities with differentiable token/frame selection."""
    #     if self.trainer.world_size > 1:
    #         if self.training and torch.cuda.is_available():
    #             sequence_output = self.all_gather_with_grad(sequence_output)
    #             visual_output = self.all_gather_with_grad(visual_output)
    #             text_mask = self.all_gather_with_grad(text_mask)
    #             video_mask = self.all_gather_with_grad(video_mask)
    #             self.synchronize()
    #
    #     # Retain WTI's learned importance within each text and video.
    #     text_weight = self.text_weight_fc(sequence_output).squeeze(-1)
    #     text_weight = text_weight.masked_fill(~text_mask.bool(), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)
    #
    #     video_weight = self.video_weight_fc(visual_output).squeeze(-1)
    #     video_weight = video_weight.masked_fill(~video_mask.bool(), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)
    #
    #     sequence_output = F.normalize(sequence_output, dim=-1, eps=1e-6)
    #     visual_output = F.normalize(visual_output, dim=-1, eps=1e-6)
    #     token_frame_logits = torch.einsum('atd,bvd->abtv', sequence_output, visual_output)
    #
    #     temperature = max(float(self.query_ware_temperature), 1e-6)
    #
    #     # For every text token and candidate video, select frames conditioned on their similarity.
    #     frame_mask = video_mask.bool()[None, :, None, :]
    #     frame_attention = torch.softmax(
    #         token_frame_logits.masked_fill(~frame_mask, float("-inf")) / temperature,
    #         dim=-1,
    #     )
    #     token_scores = torch.sum(frame_attention * token_frame_logits, dim=-1)
    #     t2v_logits = torch.einsum('abt,at->ab', token_scores, text_weight)
    #
    #     # Symmetrically, let every video frame select the relevant tokens from each candidate text.
    #     token_mask = text_mask.bool()[:, None, :, None]
    #     token_attention = torch.softmax(
    #         token_frame_logits.masked_fill(~token_mask, float("-inf")) / temperature,
    #         dim=-2,
    #     )
    #     frame_scores = torch.sum(token_attention * token_frame_logits, dim=-2)
    #     v2t_logits = torch.einsum('abv,bv->ab', frame_scores, video_weight)
    #
    #     logit_scale = self.clip.logit_scale.exp()
    #     return t2v_logits * logit_scale, v2t_logits.T * logit_scale

    # def _masked_topk_mean(self, scores, dim, k):
    #     """Average the strongest valid matches without including padded positions."""
    #     k = min(max(int(k), 1), scores.shape[dim])
    #     topk_values = torch.topk(scores, k=k, dim=dim).values
    #     valid = torch.isfinite(topk_values)
    #     topk_sum = torch.where(valid, topk_values, torch.zeros_like(topk_values)).sum(dim=dim)
    #     return topk_sum / valid.sum(dim=dim).clamp_min(1)
    #
    # def _get_logits_wti_topk(self, sequence_output, visual_output, text_mask, video_mask):
    #     """Enhance WTI with conservative Top-k temporal/token evidence."""
    #     if self.trainer.world_size > 1:
    #         if self.training and torch.cuda.is_available():
    #             sequence_output = self.all_gather_with_grad(sequence_output)
    #             visual_output = self.all_gather_with_grad(visual_output)
    #             text_mask = self.all_gather_with_grad(text_mask)
    #             video_mask = self.all_gather_with_grad(video_mask)
    #             self.synchronize()
    #
    #     text_weight = self.text_weight_fc(sequence_output).squeeze(2)  # B x N_t x D -> B x N_t
    #     text_weight.masked_fill_((1 - text_mask).clone().detach().to(torch.bool), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
    #
    #     video_weight = self.video_weight_fc(visual_output).squeeze(2)  # B x N_v x D -> B x N_v
    #     video_weight.masked_fill_((1 - video_mask).clone().detach().to(torch.bool), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
    #
    #     sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    #
    #     retrieve_logits = torch.einsum('atd,bvd->abtv', [sequence_output, visual_output])
    #     retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
    #     retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
    #
    #     # Original WTI hard matches remain the main retrieval signal.
    #     # max for video token   #为每句话中每个token找出最相似的一个frame，再根据每个token的weight计算出文本与视频的相似度
    #     t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
    #     t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])
    #
    #     v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
    #     v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])
    #     hard_logits = (t2v_logits + v2t_logits) / 2.0
    #
    #     # Top-k aggregation rewards repeated evidence across several frames/tokens.
    #     t2v_topk = self._masked_topk_mean(
    #         retrieve_logits, dim=-1, k=self.wti_topk
    #     )
    #     t2v_topk = torch.einsum('abt,at->ab', t2v_topk, text_weight)
    #     v2t_topk = self._masked_topk_mean(
    #         retrieve_logits, dim=-2, k=self.wti_topk
    #     )
    #     v2t_topk = torch.einsum('abv,bv->ab', v2t_topk, video_weight)
    #     topk_logits = (t2v_topk + v2t_topk) / 2.0
    #
    #     # Residual interpolation makes ratio=0 exactly equivalent to masked hard WTI.
    #     ratio = min(max(float(self.wti_topk_ratio), 0.0), 1.0)
    #     retrieve_logits = hard_logits + ratio * (topk_logits - hard_logits)
    #     retrieve_logits = self.clip.logit_scale.exp() * retrieve_logits
    #     return retrieve_logits, retrieve_logits.T
    #
    # def _learnable_topk_weights(self, k, device, dtype):
    #     """Build monotonic rank weights while keeping them normalized and learnable."""
    #     if k != 3:
    #         raise ValueError("wti_topk_learnable currently expects wti_topk=3")
    #
    #     # Positive gaps guarantee rank1 >= rank2 >= rank3 after softmax.
    #     gaps = F.softplus(self.wti_topk_weight_deltas).to(device=device, dtype=dtype)
    #     rank_logits = torch.stack((gaps[0] + gaps[1], gaps[1], gaps.new_zeros(())))
    #     learned = torch.softmax(rank_logits, dim=0)
    #
    #
    #     # Keep a small uniform component so training cannot collapse to one rank only.
    #     uniform = torch.full_like(learned, 1.0 / k)
    #     return 0.9 * learned + 0.1 * uniform
    #
    # def _masked_topk_weighted(self, masked_scores, dim, weights):
    #     """Apply rank weights to valid Top-k entries and renormalize after padding removal."""
    #     k = min(weights.numel(), masked_scores.shape[dim])
    #     topk_values = torch.topk(masked_scores, k=k, dim=dim).values
    #     valid = torch.isfinite(topk_values)
    #
    #     rank_shape = [1] * topk_values.ndim
    #     rank_shape[dim if dim >= 0 else topk_values.ndim + dim] = k
    #     rank_weights = weights[:k].view(rank_shape)
    #     rank_weights = rank_weights * valid
    #     rank_weights = rank_weights / rank_weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    #
    #     safe_values = torch.where(valid, topk_values, torch.zeros_like(topk_values))
    #     return (safe_values * rank_weights).sum(dim=dim)
    #
    # def _get_logits_wti_topk_learnable(self, sequence_output, visual_output, text_mask, video_mask):
    #     """Use learnable monotonic weights for Top-1/Top-2/Top-3 evidence only."""
    #     if self.trainer.world_size > 1 :
    #         if self.training and torch.cuda.is_available():  # batch merge here
    #             # print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size before gather: {sequence_output.size()}')
    #             sequence_output = self.all_gather_with_grad(sequence_output)
    #             visual_output = self.all_gather_with_grad(visual_output)
    #             text_mask = self.all_gather_with_grad(text_mask)
    #             video_mask = self.all_gather_with_grad(video_mask)
    #             self.synchronize()  # force sync
    #             # torch.distributed.barrier()  # force sync
    #             # print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size after gather: {sequence_output.size()}')
    #
    #     text_weight = self.text_weight_fc(sequence_output).squeeze(2)  # B x N_t x D -> B x N_t
    #     text_weight.masked_fill_((1 - text_mask).clone().detach().to(torch.bool), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
    #
    #     video_weight = self.video_weight_fc(visual_output).squeeze(2)  # B x N_v x D -> B x N_v
    #     video_weight.masked_fill_((1 - video_mask).clone().detach().to(torch.bool), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
    #
    #     sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    #
    #     retrieve_logits = torch.einsum('atd,bvd->abtv', [sequence_output, visual_output])
    #     retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
    #     retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
    #
    #     rank_weights = self._learnable_topk_weights(
    #         self.wti_topk, retrieve_logits.device, retrieve_logits.dtype
    #     )
    #     # self.rank_weights = rank_weights  # Old transient cache; unsafe for non-Top-k interactions.
    #     # Text-to-video: each valid text token uses learned weights over its best frames.
    #     t2v_topk = self._masked_topk_weighted(
    #         retrieve_logits, dim=-1, weights=rank_weights
    #     )
    #     t2v_logits = torch.einsum('abt,at->ab', t2v_topk, text_weight)
    #
    #     # Video-to-text: each valid video frame uses learned weights over its best tokens.
    #     v2t_topk = self._masked_topk_weighted(
    #         retrieve_logits, dim=-2, weights=rank_weights
    #     )
    #     v2t_logits = torch.einsum('abv,bv->ab', v2t_topk, video_weight)
    #
    #     retrieve_logits = (t2v_logits + v2t_logits) / 2.0
    #     retrieve_logits = self.clip.logit_scale.exp() * retrieve_logits
    #     return retrieve_logits, retrieve_logits.T

    # def _get_train_logits_wti_queue(self, sequence_output, visual_output, text_mask, video_mask, index, v_index):
    #     text_weight = self.text_weight_fc(sequence_output).squeeze(2)  # B x N_t x D -> B x N_t
    #     text_weight.masked_fill_((1 - text_mask).clone().detach().to(torch.bool), float("-inf"))
    #     text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t
    #
    #     video_weight = self.video_weight_fc(visual_output).squeeze(2)  # B x N_v x D -> B x N_v
    #     video_weight.masked_fill_((1 - video_mask).clone().detach().to(torch.bool), float("-inf"))
    #     video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v
    #
    #     # if self.trainer.world_size > 1 :
    #     #     print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size before gather: {sequence_output.size()}')
    #     #     if self.training and torch.cuda.is_available():  # batch merge here
    #     #         sequence_output = self.all_gather_with_grad(sequence_output)
    #     #         visual_output = self.all_gather_with_grad(visual_output)
    #     #
    #     #         # text_mask = self.all_gather_with_grad(text_mask)
    #     #         # video_mask = self.all_gather_with_grad(video_mask)
    #     #         text_mask = self.concat_all_gather(text_mask)
    #     #         video_mask = self.concat_all_gather(video_mask)
    #     #         v_index = self.concat_all_gather(v_index)
    #     #         self.synchronize()  # force sync
    #     #     print(f'\n---------- Rank {self.trainer.global_rank}: sequence_output size after gather: {sequence_output.size()}')
    #
    #
    #     sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)
    #     visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
    #
    #     with torch.no_grad():
    #         sequence_output_q = sequence_output.clone().detach()
    #         visual_output_q = visual_output.clone().detach()
    #         text_mask_q = text_mask.clone().detach()
    #         video_mask_q = video_mask.clone().detach()
    #         sequence_output_all = torch.cat([sequence_output_q, self.text_queue.clone().detach()], dim=0)
    #         visual_output_all = torch.cat([visual_output_q, self.visual_queue.clone().detach()], dim=0)
    #         text_mask_all = torch.cat([text_mask_q, self.text_mask_queue.clone().detach()], dim=0)
    #         video_mask_all = torch.cat([video_mask_q, self.visual_mask_queue.clone().detach()], dim=0)
    #
    #     self._dequeue_and_enqueue(sequence_output_q, visual_output_q, text_mask_q, video_mask_q, v_index)
    #
    #     retrieve_logtis_t2v = torch.einsum('atd,qvd->aqtv', [sequence_output, visual_output_all])
    #     retrieve_logtis_t2v = torch.einsum('aqtv,at->aqtv', [retrieve_logtis_t2v, text_mask])
    #     retrieve_logtis_t2v = torch.einsum('aqtv,qv->aqtv', [retrieve_logtis_t2v, video_mask_all])
    #
    #
    #     retrieve_logtis_v2t = torch.einsum('bvd,qtd->bqvt', [visual_output, sequence_output_all])
    #     retrieve_logtis_v2t = torch.einsum('bqvt,bv->bqvt', [retrieve_logtis_v2t, video_mask])
    #     retrieve_logtis_v2t = torch.einsum('bqvt,qt->bqvt', [retrieve_logtis_v2t, text_mask_all])
    #
    #     # retrieve_logits = torch.einsum('atd,bvd->abtv', [sequence_output, visual_output])
    #     # retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
    #     # retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
    #     # text_sum = text_mask.sum(-1)
    #     # video_sum = video_mask.sum(-1)
    #
    #     # max for video token   #为每句话中每个token找出最相似的一个frame，再根据每个token的weight计算出文本与视频的相似度
    #     t2v_logits, max_idx1 = retrieve_logtis_t2v.max(dim=-1)  # aqtv -> aqt
    #     t2v_logits = torch.einsum('aqt,at->aq', [t2v_logits, text_weight])
    #
    #     v2t_logits, max_idx2 = retrieve_logtis_v2t.max(dim=-1)  # bqvt -> bqv
    #     v2t_logits = torch.einsum('bqv,bv->bq', [v2t_logits, video_weight])
    #     # retrieve_logits = (t2v_logits + v2t_logits) / 2.0
    #
    #     # logit_scale = self.clip.logit_scale.exp()
    #     logit_scale = self.logit_scale.exp()
    #     return t2v_logits * logit_scale, v2t_logits * logit_scale

    def get_similarity_logits(self, sequence_output:Union[EncoderOutput, EvalReorderedOutput], visual_output:Union[EncoderOutput, EvalReorderedOutput], text_mask, video_mask, index=None, v_index=None, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            v_index = v_index.view(1, -1) if v_index is not None else None

        if self.config['interaction'] == 'wti':
            sequence_output = sequence_output.last_hidden_state
            visual_output = visual_output.pooler_output
            t2v_logits, v2t_logits = self._get_logits_wti(sequence_output, visual_output, text_mask, video_mask)

        # elif self.config['interaction'] == 'wti_segment':
        #     sequence_output = sequence_output.last_hidden_state
        #     visual_output = visual_output.pooler_output
        #     t2v_logits, v2t_logits = self._get_logits_wti_segment(
        #         sequence_output, visual_output, text_mask, video_mask
        #     )
        #
        # elif self.config['interaction'] == 'query_ware':
        #     sequence_output = sequence_output.last_hidden_state
        #     visual_output = visual_output.pooler_output
        #     t2v_logits, v2t_logits = self._get_logits_query_ware(
        #         sequence_output, visual_output, text_mask, video_mask
        #     )
        #
        # elif self.config['interaction'] == 'wti_topk':
        #     sequence_output = sequence_output.last_hidden_state
        #     visual_output = visual_output.pooler_output
        #     t2v_logits, v2t_logits = self._get_logits_wti_topk(
        #         sequence_output, visual_output, text_mask, video_mask
        #     )
        #
        # elif self.config['interaction'] == 'wti_topk_learnable':
        #     sequence_output = sequence_output.last_hidden_state
        #     visual_output = visual_output.pooler_output
        #     t2v_logits, v2t_logits = self._get_logits_wti_topk_learnable(
        #         sequence_output, visual_output, text_mask, video_mask
        #     )
        #
        # elif self.config['interaction'] == 'wti_v2':
        #     sequence_pooler = sequence_output.pooler_output
        #     sequence_last_hidden = sequence_output.last_hidden_state
        #     visual_last_hidden = visual_output.last_hidden_state
        #
        #     t2v_logits, v2t_logits = self._get_logits_wti_v2(sequence_pooler, sequence_last_hidden, visual_last_hidden, text_mask, video_mask)
        #
        # elif self.config['interaction'] == 'cls':
        #     sequence_pooler = sequence_output.pooler_output
        #     visual_pooler = visual_output.pooler_output
        #     t2v_logits, v2t_logits = self._get_logits_cls(sequence_pooler, visual_pooler, text_mask, video_mask)

        else:
            raise NotImplementedError

        return t2v_logits, v2t_logits

    def get_similarity_logits_queue(self, sequence_feat, sequence_output, visual_output, visual_hidden, text_mask, video_mask, index=None, v_index=None, shaped=False):
        if shaped is False:
            text_mask = text_mask.view(-1, text_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            v_index = v_index.view(1, -1) if v_index is not None else None

        if self.training:
            t2v_logits, v2t_logits = self._get_train_logits_wti_queue(sequence_output, visual_output, text_mask, video_mask, index, v_index)
        else:
            t2v_logits, v2t_logits = self._get_logits_wti(sequence_output, visual_output, text_mask, video_mask)
        return t2v_logits, v2t_logits

    #   ============================  模型训练方法  ===================================
    def on_train_epoch_start(self):
        if self.trainer.world_size > 1:
            self.trainer.datamodule.train_sampler.set_epoch(self.trainer.current_epoch)
            print('Set epoch for train sampler.')


    def train_batch(self, batch):
        input_ids, input_mask, segment_ids, video, video_mask, index, v_index = batch

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        input_mask = input_mask.view(-1, input_mask.shape[-1])
        segment_ids = segment_ids.view(-1, segment_ids.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        v_index = v_index.view(1, -1)


        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape  # 16, 1, 12, 1, 3, 224, 224
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts


        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, input_mask, segment_ids, video, video_mask, shaped=True, video_frame=video_frame)  # float32

        #
        # sequence_feat, sequence_last_hidden = sequence_output.pooler_output, sequence_output.last_hidden_state
        # visual_feat, visual_last_hidden = visual_output.pooler_output, visual_output.last_hidden_state

        if self.use_queue:
            idx_all = torch.cat([v_index, self.idx_queue.clone().detach()], dim=1)
            pos_idx = torch.eq(v_index.T, idx_all).float()
            sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            t2v_logits, v2t_logits = self.get_similarity_logits_queue(
                sequence_output, visual_output,
                input_mask, video_mask, index, v_index, shaped=True)

        else:
            # v_index = self.concat_all_gather(v_index)
            # pos_idx = torch.eq(v_index.T, v_index).float()
            # sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)

            t2v_logits, v2t_logits = self.get_similarity_logits(
                sequence_output, visual_output,
                input_mask, video_mask,index, v_index, shaped=True)

            # sim_targets = torch.eye(t2v_logits.shape[0], device=self.device)


        # sim_loss1 = self.loss_fct(t2v_logits, sim_targets)
        # sim_loss2 = self.loss_fct(v2t_logits, sim_targets)
        #
        # sim_loss11 = self.loss_fct1(t2v_logits)
        # sim_loss22 = self.loss_fct1(v2t_logits)
        #
        # sim_loss = (sim_loss1 + sim_loss2) / 2.0
        # sim_loss2 = (sim_loss11 + sim_loss22) / 2.0
        #
        # print(f'============ \n sim_loss1: {sim_loss1}, \t sim_loss2: {sim_loss2} \n ======================')

        sim_loss1 = self.loss_fct(t2v_logits)
        sim_loss2 = self.loss_fct(v2t_logits)
        sim_loss = (sim_loss1 + sim_loss2) / 2.0

        return sim_loss

    def forward(self, batch):
        return self.train_batch(batch)

    def training_step(self, batch, batch_idx):
        if self.config['statistic'] and self.trainer.current_epoch == 0 and batch_idx == 0 and self.trainer.global_rank == 0:
            self.statistic(self, batch, net_name='side')

        irtr_loss = self.train_batch(batch)

        clip_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        side_lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[-1]
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0 and batch_idx % self.trainer.accumulate_grad_batches == 0:
            # https://github.com/openai/CLIP/issues/46
            # torch.clamp_(self.logit_scale.data, max=np.log(200))
            torch.clamp_(self.clip.logit_scale.data, max=np.log(100))

            self.print('Global step:{global_step}.\t'
                       'Train Loss: {loss:.4f}.\t'
                       'CLIP LR: {clip_lr:.3E}.\t'
                       'Side LR: {side_lr:.3E}.\t'
                       .format(global_step=self.trainer.global_step,
                               loss=irtr_loss,
                               clip_lr=clip_lr,
                               side_lr=side_lr))

            self.log(f"train/total_loss", irtr_loss)

            if self.clip.visual.side_mamba_v.hierarchical:
                norm_weights = F.softmax(self.clip.visual.side_mamba_v.hierarchical_weight, dim=-1)
                self.log(f"train/hierarchical_weight_1", norm_weights[0])
                self.log(f"train/hierarchical_weight_2", norm_weights[1])
                self.log(f"train/hierarchical_weight_3", norm_weights[2])
                self.log(f"train/hierarchical_weight_4", norm_weights[3])
            #
            # for i, j in enumerate(self.clip.visual.side_fusion_logits):
            #     self.log(f"train/side_fusion_{i}_layer_{self.clip.visual.side_layer_route_index[i]}", torch.sigmoid(j))

            # for i, j in enumerate(self.rank_weights):
            #     self.log(f"train/top_{i+1}_ratio", j)

            # self.log(f"train/side_ratio", self.clip.visual.side_ratio)
            # self.log(f'train/side_proj', self.clip.visual.side_proj)

            # self.log(f"train/temp", pl_module.temp)
            # self.log(f"train/alpha", pl_module.alpha)
        return irtr_loss

    def on_train_epoch_end(self) -> None:
        # self.epoch_wrapup(None, phase='train')
        # self.training_step_outputs.clear()  # free memory
        ...


    #   ============================  模型验证方法  ===================================
    def on_validation_start(self) -> None:
        super().on_validation_start()
        self._prepare_val('val')

    def validation_step(self, batch, batch_idx):
        self._cache_features(batch, batch_idx)


    def on_validation_epoch_end(self) -> None:
        # 不传入out了，直接从self属性获取每个val step的返回
        self.epoch_wrapup(phase='val')
        self._clean_cache_features()

    def on_test_start(self) -> None:
        super().on_test_start()
        self._prepare_val('test')


    def test_step(self, batch, batch_idx):
        self._cache_features(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        # 不传入out了，直接从self属性获取每个val step的返回
        self.epoch_wrapup(phase='test')
        self._clean_cache_features()


    #   ============================  模型验证方法  ===================================
    def _prepare_val(self, phase) -> None:
        if phase == 'val':
            dataset = self.trainer.datamodule.val_dataset
        elif phase == 'test':
            dataset = self.trainer.datamodule.test_dataset
        elif phase == 'predict':
            dataset = self.trainer.datamodule.pred_dataset
        else:
            raise NotImplementedError
        self.sentence_num = dataset.sentence_num
        self.video_num = dataset.video_num
        self.batch_visual_output, self.batch_text_output = [], []

        # #################################################################
        ## below variables are used to multi-sentences retrieval
        # multi_sentence_: important tag for eval
        # cut_off_points: used to tag the label when calculate the metric
        # sentence_num: used to cut the sentence representation
        # video_num: used to cut the video representation
        # #################################################################
        self.multi_sentence_, self.cut_off_points_ = False, []

        if getattr(dataset, 'multi_sentence_per_video', None):
            self.multi_sentence_ = True
            self.cut_off_points_ = [itm - 1 for itm in dataset.cut_off_points]
            # print(f'----- cut_off_points: {self.cut_off_points_} -----')

        if self.multi_sentence_:
            print(f"### {phase} ### under the multi-sentence per video clip setting.")
            print("sentence num: {}, video num: {}".format(dataset.sentence_num, dataset.video_num))

    def _cache_features(self, batch, batch_idx):
        input_ids, input_mask, segment_ids, video, video_mask, index, v_index = batch

        # print(f'batch_idx: {batch_idx}, index_t:{index}, index_v:{v_index}')

        if self.multi_sentence_:         # MSVD 走这边， MSVD的验证集是一段视频对应多条文本
            # multi-sentences retrieval means: one clip has two or more descriptions.
            # ---------------------------- (1)先对文本编码 ----------------------------
            sequence_output = self.get_sequence_output(input_ids, segment_ids, input_mask)

            if self.config['interaction'] in ['wti', 'wti_segment', 'query_ware', 'wti_topk', 'wti_topk_learnable']:
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        # pooler_output=sequence_output.pooler_output,
                        last_hidden_state=sequence_output.last_hidden_state,
                    )
                )
            elif self.config['interaction'] == 'wti_v2':
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        pooler_output=sequence_output.pooler_output,
                        last_hidden_state=sequence_output.last_hidden_state,
                    )
                )

            elif self.config['interaction'] == 'cls':
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        pooler_output=sequence_output.pooler_output,
                        # last_hidden_state=sequence_output.last_hidden_state,
                    )
                )

            # ---------------------------- (2)找出文本对应的视频 ----------------------------
            # cut_off_points_ 表示在整个数据集中，每个视频对应其所有描述文本的位置
            common_elements = np.intersect1d(np.array(self.cut_off_points_), index.cpu().numpy())
            # video_index 是该video在整个数据集中的位置
            video_index = torch.tensor([np.where(self.cut_off_points_ == x)[0][0] for x in common_elements],
                                       device=video.device, dtype=torch.int64)
            # video_map 是该批次中，需要编码的视频所对应文本的index
            video_map = [np.where(index.cpu().numpy() == x)[0][0] for x in common_elements]

            # ---------------------------- (3)进行视频编码 ----------------------------
            if len(video_map) > 0:
                video, video_mask = video[video_map, ...], video_mask[video_map, ...]
                visual_output = self.get_visual_output(video, video_mask)

                if self.config['interaction'] in ['wti', 'wti_segment', 'query_ware', 'wti_topk', 'wti_topk_learnable']:
                    self.batch_visual_output.append(
                        EvalCacheOutput(
                            index=video_index,
                            mask=video_mask,
                            pooler_output=visual_output.pooler_output,
                            # last_hidden_state=visual_output.last_hidden_state,
                        )
                    )
                elif self.config['interaction'] == 'wti_v2':
                    self.batch_visual_output.append(
                        EvalCacheOutput(
                            index=video_index,
                            mask=video_mask,
                            pooler_output=visual_output.pooler_output,
                            last_hidden_state=visual_output.last_hidden_state,
                        )
                    )

                elif self.config['interaction'] == 'cls':
                    self.batch_visual_output.append(
                        EvalCacheOutput(
                            index=video_index,
                            mask=video_mask,
                            pooler_output=visual_output.pooler_output,
                            # last_hidden_state=visual_output.last_hidden_state,
                        )
                    )

        else:   # MSRVTT 走这边，MSRVTT的验证集是一段视频对应一条文本，因此index和video_index一致
            sequence_output, visual_output = self.get_sequence_visual_output(input_ids, input_mask, segment_ids,
                                                                             video, video_mask)  # float32

            if self.config['interaction'] in ['wti', 'wti_segment', 'query_ware', 'wti_topk', 'wti_topk_learnable']:
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        # pooler_output=sequence_output.pooler_output,
                        last_hidden_state=sequence_output.last_hidden_state,
                    )
                )
                self.batch_visual_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=video_mask,
                        pooler_output=visual_output.pooler_output,
                        # last_hidden_state=visual_output.last_hidden_state,
                    )
                )
            elif self.config['interaction'] == 'wti_v2':
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        pooler_output=sequence_output.pooler_output,
                        last_hidden_state=sequence_output.last_hidden_state,
                    )
                )
                self.batch_visual_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=video_mask,
                        pooler_output=visual_output.pooler_output,
                        last_hidden_state=visual_output.last_hidden_state,
                    )
                )

            elif self.config['interaction'] == 'cls':
                self.batch_text_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=input_mask,
                        pooler_output=sequence_output.pooler_output,
                        # last_hidden_state=sequence_output.last_hidden_state,
                    )
                )
                self.batch_visual_output.append(
                    EvalCacheOutput(
                        index=index,
                        mask=video_mask,
                        pooler_output=visual_output.pooler_output,
                        # last_hidden_state=visual_output.last_hidden_state,
                    )
                )

            # self.batch_text_output.append(sequence_output)
            # self.batch_visual_output.append(visual_output)


    def _clean_cache_features(self):
        # free memory
        self.batch_visual_output.clear()
        self.batch_text_output.clear()


    def _run_on_single_gpu(self, reordered_sequence_list:List[EvalReorderedOutput], reordered_visual_list:List[EvalReorderedOutput]):
        sim_matrix = []
        for idx1, sequence_output in tqdm(enumerate(reordered_sequence_list), total=len(reordered_sequence_list)):
            each_row = []

            for idx2, visual_output in enumerate(reordered_visual_list):
                b1b2_logits, *_tmp = self.get_similarity_logits(sequence_output, visual_output, sequence_output.mask, visual_output.mask)
                # b1b2_logits = b1b2_logits.cpu().detach().numpy()
                each_row.append(b1b2_logits)
            # each_row = np.concatenate(tuple(each_row), axis=-1)
            each_row = torch.cat(tuple(each_row), dim=-1)
            sim_matrix.append(each_row)
        sim_matrix = torch.cat(tuple(sim_matrix), dim=0)
        # sim_matrix = np.concatenate(tuple(sim_matrix), axis=0)
        return sim_matrix

    def epoch_wrapup(self, phase):
        # ----------------------------------
        # 1. If distributed, use index to reorder.
        reordered_list = self._distributed_reorder(self.batch_text_output, self.batch_visual_output)

        # ----------------------------------
        # 2. calculate the similarity
        sim_matrix = self._calculate_similarity(*reordered_list)

        # ----------------------------------
        # 3. calculate the metrics
        self._calculate_metrics(sim_matrix, phase)

    def _distributed_reorder(self, batch_text_output: List[EvalCacheOutput], batch_visual_output: List[EvalCacheOutput]):

        batch_index_t = [item.index for item in batch_text_output]
        batch_index_v = [item.index for item in batch_visual_output]
        batch_mask_t = [item.mask for item in batch_text_output]
        batch_mask_v = [item.mask for item in batch_visual_output]

        batch_pooler_t = None
        batch_pooler_v = None
        batch_last_hidden_t = None
        batch_last_hidden_v = None
        batch_hidden_t = None
        batch_hidden_v = None

        if self.config['interaction'] in ['wti', 'wti_segment', 'query_ware', 'wti_topk', 'wti_topk_learnable']:
            batch_last_hidden_t = [item.last_hidden_state for item in batch_text_output]
            batch_pooler_v = [item.pooler_output for item in batch_visual_output]

        elif self.config['interaction'] == 'wti_v2':
            batch_pooler_t = [item.pooler_output for item in batch_text_output]
            batch_last_hidden_t = [item.last_hidden_state for item in batch_text_output]
            batch_last_hidden_v = [item.last_hidden_state for item in batch_visual_output]



        elif self.config['interaction'] == 'wti_v3':
            ...
        elif self.config['interaction'] == 'cls':
            batch_pooler_t = [item.pooler_output for item in batch_text_output]
            batch_pooler_v = [item.pooler_output for item in batch_visual_output]
        else:
            raise NotImplementedError

        rank = self.trainer.global_rank
        # print(f'\n---------- Rank {rank}: before gather --------------\n'
              # f'text index: {index_list}\n'
              # f'video index: {a.size()}\n'
              # )

        index_t = torch.cat(batch_index_t, dim=0)
        index_v = torch.cat(batch_index_v, dim=0)
        mask_t = torch.cat(batch_mask_t, dim=0)
        mask_v = torch.cat(batch_mask_v, dim=0)

        pooler_t = None if batch_pooler_t is None else torch.cat(batch_pooler_t, dim=0)
        pooler_v = None if batch_pooler_v is None else torch.cat(batch_pooler_v, dim=0)
        last_hidden_t = None if batch_last_hidden_t is None else torch.cat(batch_last_hidden_t, dim=0)
        last_hidden_v = None if batch_last_hidden_v is None else torch.cat(batch_last_hidden_v, dim=0)
        hidden_t = None if batch_hidden_t is None else torch.cat(batch_hidden_t, dim=0)
        hidden_v = None if batch_hidden_v is None else torch.cat(batch_hidden_v, dim=0)

        # print(f'\n Before reorder111, in rank{self.trainer.local_rank}, total text:{len(index_t)}, total video:{len(index_t)}')
        # print(f'\n index_t:{index_t}')
        # print(f'\n index_v:{index_v}')

        if self.trainer.world_size > 1:
            # 将所有卡上的数据拼接并按照index排序
            mask_t = self.concat_all_gather_diff_size(mask_t, index_t, self.sentence_num)
            mask_v = self.concat_all_gather_diff_size(mask_v, index_v, self.video_num)

            pooler_t = None if pooler_t is None else self.concat_all_gather_diff_size(pooler_t, index_t, self.sentence_num)
            pooler_v = None if pooler_v is None else self.concat_all_gather_diff_size(pooler_v, index_v, self.video_num)
            last_hidden_t = None if last_hidden_t is None else self.concat_all_gather_diff_size(last_hidden_t, index_t, self.sentence_num)
            last_hidden_v = None if last_hidden_v is None else self.concat_all_gather_diff_size(last_hidden_v, index_v, self.video_num)
            hidden_t = None if hidden_t is None else self.concat_all_gather_diff_size(hidden_t, index_t, self.sentence_num)
            hidden_v = None if hidden_v is None else self.concat_all_gather_diff_size(hidden_v, index_v, self.video_num)

            index_t = self.concat_all_gather_diff_size(index_t, index_t, self.sentence_num)
            index_v = self.concat_all_gather_diff_size(index_v, index_v, self.video_num)

            torch.distributed.barrier()

            # print(f'\n---------- Rank {rank}: start reorder  --------------\n')
            index_t, sorted_order = index_t[:self.sentence_num].sort(dim=0)
            index_v, sorted_order_v = index_v[:self.video_num].sort(dim=0)

            print(
                f'\n Before reorder, in rank{self.trainer.local_rank}, total text:{len(index_t)}, total video:{len(index_t)}')
            # print(f'\n index_t:{index_t}')
            # print(f'\n index_v:{index_v}')

            mask_t = mask_t[index_t]
            mask_v = mask_v[index_v]

            pooler_t = None if pooler_t is None else pooler_t[index_t]
            pooler_v = None if pooler_v is None else pooler_v[index_v]
            last_hidden_t = None if last_hidden_t is None else last_hidden_t[index_t]
            last_hidden_v = None if last_hidden_v is None else last_hidden_v[index_v]
            hidden_t = None if hidden_t is None else hidden_t[index_t]
            hidden_v = None if hidden_v is None else hidden_v[index_v]
            # index_t, sorted_order = index_t.sort(dim=0)
            # index_v, sorted_order_v = index_v.sort(dim=0)
            # mask_t = mask_t[index_t]
            # mask_v = mask_v[index_v]
            #
            # pooler_t = None if pooler_t is None else pooler_t[index_t]
            # pooler_v = None if pooler_v is None else pooler_v[index_v]
            # last_hidden_t = None if last_hidden_t is None else last_hidden_t[index_t]
            # last_hidden_v = None if last_hidden_v is None else last_hidden_v[index_v]
            # hidden_t = None if hidden_t is None else hidden_t[index_t]
            # hidden_v = None if hidden_v is None else hidden_v[index_v]

            # sequence_output = sequence_output[index_t]
            # visual_output = visual_output[index_v]
            # print(f'\n---------- Rank {rank}: after reorder --------------\n '
            #       f'text index: {index_t.size()}\n'
            #       f'video index: {index_v.size()}\n'
            #       )

        batch_mask_t = list(torch.split(mask_t, self.config['batch_size_val'], dim=0))
        batch_mask_v = list(torch.split(mask_v, self.config['batch_size_val'], dim=0))

        batch_pooler_t = None if pooler_t is None else list(torch.split(pooler_t, self.config['batch_size_val'], dim=0))
        batch_pooler_v = None if pooler_v is None else list(torch.split(pooler_v, self.config['batch_size_val'], dim=0))

        batch_last_hidden_t = None if last_hidden_t is None else list(torch.split(last_hidden_t, self.config['batch_size_val'], dim=0))
        batch_last_hidden_v = None if last_hidden_v is None else list(torch.split(last_hidden_v, self.config['batch_size_val'], dim=0))

        batch_hidden_t = None if hidden_t is None else list(torch.split(hidden_t, self.config['batch_size_val'], dim=0))
        batch_hidden_v = None if hidden_v is None else list(torch.split(hidden_v, self.config['batch_size_val'], dim=0))

        if self.trainer.local_rank == 0:
            print(f'After reorder, total text:{len(index_t)}, total video:{len(index_t)}')

        return [batch_mask_t, batch_mask_v,
                batch_pooler_t, batch_pooler_v[:self.video_num],
                batch_last_hidden_t, batch_last_hidden_v,
                batch_hidden_t, batch_hidden_v]

    def _calculate_similarity(self, batch_mask_t, batch_mask_v, batch_pooler_t, batch_pooler_v, batch_last_hidden_t, batch_last_hidden_v,  batch_hidden_t, batch_hidden_v):
        rank = self.trainer.global_rank
        n_gpu = self.trainer.world_size
        device = batch_mask_t[0].device

        n_batches_t, n_batches_v = len(batch_mask_t), len(batch_mask_v)
        batch_t_output_len = batch_mask_t[0].size(0)
        matrix_size = [sum(len(batch) for batch in batch_mask_t),
                       sum(len(batch) for batch in batch_mask_v)]
        matrix_size1 = [self.sentence_num, self.video_num]

        assert matrix_size1 == matrix_size

        sim_matrix = torch.full(matrix_size, 0.).to(device)

        if rank == 0:
            print('---------- Start calculating the similarity -----------')
            print(f'Calculate similarity, sim_matrix size:{sim_matrix.shape}')

        split_len = (n_batches_t + n_gpu - 1) // n_gpu
        s_, e_ = rank * split_len, (rank + 1) * split_len

        reordered_sequence_list = [
            EvalReorderedOutput(
                mask = batch_mask_t[i],
                pooler_output = None if batch_pooler_t is None else batch_pooler_t[i],
                last_hidden_state = None if batch_last_hidden_t is None else batch_last_hidden_t[i],
                hidden_states = None if batch_hidden_t is None else batch_hidden_t[i],
            )
            for i in range(len(batch_mask_t))
        ]

        reordered_visual_list = [
            EvalReorderedOutput(
                mask=batch_mask_v[i],
                pooler_output=None if batch_pooler_v is None else batch_pooler_v[i],
                last_hidden_state=None if batch_last_hidden_v is None else batch_last_hidden_v[i],
                hidden_states=None if batch_hidden_v is None else batch_hidden_v[i],
            )
            for i in range(len(batch_mask_v))
        ]

        parameters_tuple_list = [reordered_sequence_list[s_:e_], reordered_visual_list]
        matrix_block = self._run_on_single_gpu(*parameters_tuple_list)
        matrix_index_interval = min(sim_matrix.size(0), e_ * batch_t_output_len) - s_ * batch_t_output_len
        # print(f'\n---------- Rank {rank}: Complete sim matrix size: {sim_matrix.size()}')
        # print(f'\n---------- Rank {rank}: matrix block size: {matrix_block.size()}')
        # print(
        #     f'\n---------- Rank {rank}: from batch {s_} to {e_} (sample {s_ * batch_t_output_len} to {s_ * batch_t_output_len + matrix_index_interval}) ----------')

        sim_matrix[s_ * batch_t_output_len: s_ * batch_t_output_len + matrix_index_interval] = matrix_block[
                                                                                               :matrix_index_interval]

        # 多卡情况下，同步进度
        if n_gpu > 1:
            torch.distributed.all_reduce(sim_matrix, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.barrier()

        return sim_matrix.cpu().detach().numpy()

    def _calculate_metrics(self, sim_matrix, phase):
        if self.multi_sentence_:
            print("before reshape, sim matrix size: {} x {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            cut_off_points2len_ = [itm + 1 for itm in self.cut_off_points_[:sim_matrix.shape[1]]]
            max_length = max([e_ - s_ for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_)])
            sim_matrix_new = []
            for s_, e_ in zip([0] + cut_off_points2len_[:-1], cut_off_points2len_):
                sim_matrix_new.append(np.concatenate((sim_matrix[s_:e_],
                                                      np.full((max_length - e_ + s_, sim_matrix.shape[1]), -np.inf)),
                                                     axis=0))
            sim_matrix = np.stack(tuple(sim_matrix_new), axis=0)
            print("after reshape, sim matrix size: {} x {} x {}".
                  format(sim_matrix.shape[0], sim_matrix.shape[1], sim_matrix.shape[2]))

            tv_metrics = tensor_text_to_video_metrics(sim_matrix)
            vt_metrics = compute_metrics(tensor_video_to_text_sim(sim_matrix))
        else:
            print("sim matrix size: {}, {}".format(sim_matrix.shape[0], sim_matrix.shape[1]))
            tv_metrics = compute_metrics(sim_matrix)
            vt_metrics = compute_metrics(sim_matrix.T)
            # print('\t Length-T: {}, Length-V:{}'.format(len(sim_matrix), len(sim_matrix[0])))
        rsum = tv_metrics['R1'] + tv_metrics['R5'] + tv_metrics['R10'] + vt_metrics['R1'] + vt_metrics['R5'] + \
               vt_metrics['R10']

        dataset_name = self.config['dataset']['name']

        for k, v in tv_metrics.items():
            self.log(f"{phase}_{dataset_name}/T2V_{k}", v)
        for k, v in vt_metrics.items():
            self.log(f"{phase}_{dataset_name}/V2T_{k}", v)
        self.log(f"{phase}_{dataset_name}/R@SUM", rsum)

        if self.trainer.local_rank == 0:
            print("Text-to-Video:")
            print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                  format(tv_metrics['R1'], tv_metrics['R5'], tv_metrics['R10'], tv_metrics['MedianR'],
                         tv_metrics['MeanR']))
            print("Video-to-Text:")
            print('\t>>>  R@1: {:.1f} - R@5: {:.1f} - R@10: {:.1f} - Median R: {:.1f} - Mean R: {:.1f}'.
                  format(vt_metrics['R1'], vt_metrics['R5'], vt_metrics['R10'], vt_metrics['MedianR'],
                         vt_metrics['MeanR']))
            print('\t>>>  R@SUM: {:.1f}'.format(rsum))
