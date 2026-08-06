'''
BaseModule: 提供模型需要的一些基础方法
'''
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,)

import torch.distributed as dist

from thop import profile
from torchinfo import summary

_LOCAL_PROCESS_GROUP = None


from src.models.module_utils import LayerNorm
from .dist_utils import GatherLayer



class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

    #   ============================  Lightning Module 基础方法  ===================================

    def on_fit_start(self) -> None:
        print('=' * 30,  'FIT LOOP', '=' * 30)
        self.training_step_outputs = []

    def on_validation_start(self) -> None:
        print('=' * 30, 'VALIDATION LOOP', '=' * 30)
        self.validation_step_outputs = []

    def on_test_start(self) -> None:
        print('=' * 30, 'TEST LOOP', '=' * 30)
        self.test_step_outputs = []

    def create_visual_encoder(self, config):
        raise NotImplementedError("create custom visual encoder")

    def create_text_encoder(self, config):
        raise NotImplementedError("create custom text encoder")

    #   ============================  加载模型方法  ===================================

    @classmethod
    def from_pretrained(cls, config):
        raise NotImplementedError("load from pretrained model")

    @classmethod
    def from_checkpoint(cls, config):
        raise NotImplementedError("load from personal checkpoint")


    def freeze_module(self, module):
        """
        Freezes module's parameters.
        """
        for parameter in module.parameters():
            parameter.requires_grad = False

    def unfreeze_module(self, module):
        """
        Unfreezes module's parameters.
        """
        for parameter in module.parameters():
            parameter.requires_grad = True

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @staticmethod
    def count_params(model):
        # The unit is M (million)
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        params = round(params/(1024**2), 2)
        return params


    #   ============================  初始化方法  ===================================

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def init_preweight(cls, model, state_dict, prefix=None):
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        if prefix is not None:
            old_keys = []
            new_keys = []
            for key in state_dict.keys():
                old_keys.append(key)
                new_keys.append(prefix + key)
            for old_key, new_key in zip(old_keys, new_keys):
                state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix='')

        # if prefix is None and (cls.get_rank() == 0):
        #     print("-" * 20)
        #     if len(missing_keys) > 0:
        #         print("Weights of {} not initialized from pretrained model: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)))
        #     if len(unexpected_keys) > 0:
        #         print("Weights from pretrained model not used in {}: {}"
        #                     .format(model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)))
        #     if len(error_msgs) > 0:
        #         print("Weights from pretrained model cause errors in {}: {}"
        #                      .format(model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)))

        return model

    #   ============================  优化器初始化方法  ===================================


    def cal_steps(self):
        # 计算 max_steps 和 warmup_steps
        if self.trainer.max_steps == None or self.trainer.max_epochs != None:
            max_steps = (len(self.trainer.datamodule.train_dataloader()) * self.trainer.max_epochs
                         // self.config['gradient_accumulation_steps'])
        else:
            max_steps = self.trainer.max_steps
        # 当 warmup_steps=-1 时不启用warm up
        warmup_steps = max(0, self.config['warmup_steps'])
        if isinstance(warmup_steps, float):
            warmup_steps = int(warmup_steps * max_steps)
        print(f'====== Max steps: {max_steps},\t Warm up steps: {warmup_steps} =========')
        return max_steps, warmup_steps

    def get_scheduler(self, optimizer, warmup_steps, max_steps):
        # 设置scheduler
        scheduler = self.config['optimizer']['scheduler']
        num_cycles = self.config['optimizer'].get('num_cycles', None)
        if scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
            )
        elif scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=num_cycles
            )
        elif scheduler == 'cosine_hard':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps,
                num_cycles=num_cycles
            )
        else:
            scheduler = None
        sched = {
            'scheduler': scheduler, 'interval': 'step'
        }
        return sched


    #   ============================  分布式方法  ===================================

    @staticmethod
    def get_world_size() -> int:
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size()

    @staticmethod
    def get_local_size() -> int:
        """
        Returns:
            The size of the per-machine process group,
            i.e. the number of processes per machine.
        """
        if not dist.is_available():
            return 1
        if not dist.is_initialized():
            return 1
        return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)

    @staticmethod
    def get_rank() -> int:
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    @staticmethod
    def get_local_rank() -> int:
        """
        Returns:
            The rank of the current process within the local (per-machine) process group.
        """
        if not dist.is_available():
            return 0
        if not dist.is_initialized():
            return 0
        assert _LOCAL_PROCESS_GROUP is not None
        return dist.get_rank(group=_LOCAL_PROCESS_GROUP)


    def synchronize(self):
        """
        Helper function to synchronize (barrier) among all processes when
        using distributed training
        """
        if not dist.is_available():
            return
        if not dist.is_initialized():
            return
        # world_size = dist.get_world_size()
        if self.trainer.world_size == 1:
            return
        dist.barrier()

    @torch.no_grad()
    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        if self.trainer.world_size > 1:
            tensors_gather = [torch.ones_like(tensor)
                              for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

            output = torch.cat(tensors_gather, dim=0)
            return output
        else:
            return tensor


    @torch.no_grad()
    def concat_all_gather_diff_size(self, tensor: torch.Tensor, index, total_size):
        """
        Performs all_gather operation on the provided tensors with different size.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        rank = self.trainer.global_rank

        device = tensor.device

        # print(f'\n---------- Rank {rank}: tensor size {tensor.shape}  --------------\n')
        tensor_size = [total_size] + list(tensor.shape[1:])
        # print(f'\n---------- Rank {rank}: total tensor size {tensor_size}  --------------\n')
        tensors_gather = torch.zeros(tensor_size, device=device, dtype=tensor.dtype)

        tensors_gather[index, ...] = tensor

        torch.distributed.barrier()
        torch.distributed.all_reduce(tensors_gather, op=torch.distributed.ReduceOp.SUM)


        return tensors_gather

    def all_gather_with_grad(self, tensors):
        """
        Performs all_gather operation on the provided tensors.
        Graph remains connected for backward grad computation.
        """
        # Queue the gathered tensors
        # world_size = torch.distributed.get_world_size()
        # There is no need for reduction in the single-proc case
        if self.trainer.world_size == 1:
            return tensors

        tensor_all = GatherLayer.apply(tensors, self.trainer.world_size, self.trainer.global_rank)

        return tensor_all

    #   ============================  统计方法  ===================================

    def statistic(self, model, *args, net_name='side'):
        # from thop import profile
        # from torchinfo import summary

        # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #     model = model.module

        # 统计计算量的时候，最好 batch size 统一设置为 1，因为 batch size 会影响计算量
        # 1. 使用 torchinfo 计算 MACs (Multiply-Accumulate Operations) 和 参数量

        print("========================= Torchinfo for Params and FLOPs ================================")
        # 直接打印详细报告，包含每层的参数和 Mult-Adds
        model_stats = summary(model, input_data=(*args,), verbose=1)

        # 通常 1 MAC ≈ 2 FLOPs
        flops = model_stats.total_mult_adds * 2
        print(f"参数量 (Params): {model_stats.total_params / 1e6:.2f} M (百万)")
        print(f"可训练参数量 (Native): {model_stats.trainable_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据

        print(f"计算量 (GFLOPs): {flops / 1e9:.2f} G (十亿)")
        print(f"计算量 (TFLOPs): {flops / 1e12:.4f} T")

        # 2. 使用 thop 计算 MACs (Multiply-Accumulate Operations) 和 参数量
        print("========================= Thop for Params and FLOPs =========================")
        macs, params = profile(model, inputs=(*args,), verbose=False)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_side_params = sum(p.numel() for n,p in model.named_parameters() if (p.requires_grad and net_name in n))

        # 通常 1 MAC ≈ 2 FLOPs
        flops = macs * 2
        print(f"参数量 (Params): {params / 1e6:.2f} M (百万)")
        print(f"可训练参数量 (Native): {trainable_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据
        print(f"可训练侧网络参数量 ({net_name}): {trainable_side_params / 1e6:.2f} M (百万)")  # <--- 你想要的数据

        print(f"计算量 (GFLOPs): {flops / 1e9:.2f} G (十亿)")
        print(f"计算量 (TFLOPs): {flops / 1e12:.4f} T")

