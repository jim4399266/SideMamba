import copy
import pytorch_lightning as pl
import os
import sys
import torch
from pathlib import Path
from typing import Dict


sys.path.append('../')
os.environ["NCCL_DEBUG"] = "INFO"
from config.retrieval.config_main import ex
from datamodules import build_datamodule
from lightning_modules import build_retrieval_module, build_retrieval_module_ckpt
# from lightning_modules.dist_utils import init_distributed_print

torch.set_float32_matmul_precision('medium')


def rectify_config(loaded_config, current_config):
    rectify_keys = ['encoder', 'side']
    for key in rectify_keys:
        current_config[key] = loaded_config[key]
    return current_config


def preprocess_config(config: Dict):

    config['num_device'] = config['devices'] if isinstance(config['devices'], int) else len(config['devices'])
    config['dist'] = True if config['num_device'] > 1 else False

    grad_steps = max(
        config['batch_size'] //
        (config['per_gpu_batch_size'] * max(1, config['num_device']) * config['num_nodes'])
        , 1)
    config['gradient_accumulation_steps'] = grad_steps


    prefix_dict = {
        '': config['param'],
        # 'heads': config['num_heads'],
        'bs': config["batch_size"],
        'pbs': config["per_gpu_batch_size"],
        # 'queue_size': config['queue_size'],
        # 'topk': config['top_k'],
        'epoch': config["max_epoch"],
        'lr': config["optimizer"]["init_lr"],
        'from_': '',
    }


    log_name = Path(config['dataset']['name']) / '_'.join([f'{k}{v}' if (k != 'task' and k != 'pretrained_arch' and k != 'dataset') else f'{v}'
                         for k, v in prefix_dict.items()])
    config['log_name'] = log_name

    # config['encoder']['pretrained_clip_name'] = Path(config['pretrained_model_dir']) / config['encoder']['pretrained_clip_name']
    config['encoder']["pretrained_clip"] = Path(config['pretrained_model_dir']) / config['encoder']["pretrained_clip"]
    config['encoder']['tokenizer'] = Path(config['pretrained_model_dir']) / config['encoder']['tokenizer']
    config['encoder']['vit'] = Path(config['pretrained_model_dir']) / config['encoder']['vit']
    config['side']['T'] = config['dataset']['max_frames']


def test_after_train(trainer, model, dm):
    weight_paths = list(trainer.checkpoint_callback.best_k_models.keys())

    # weight_paths = list(Path(checkpoint_callback.dirpath).rglob('*.[pc][tk][hp]*'))
    # weight_paths = list(Path('/home/tzj/codes/my_clip/outputs/'
    #                          'irtr_bs200_pbs50_epoch6_lr1e-05_is224_from_model_base/version_19/').rglob(
    #     '*.[pc][tk][hp]*'))

    print(weight_paths)
    for ckpt in weight_paths:
        trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))



@ex.automain
def main(_config):
    config = copy.deepcopy(_config)
    # config = Map(_config)

    config['param'] = ' '.join(sys.argv[2:])

    preprocess_config(config)
    # init_distributed_print()

    pl.seed_everything(config["seed"])
    strategy = 'ddp_find_unused_parameters_true' if config['num_device'] > 1 else 'auto'
    # strategy = 'ddp' if config['num_device'] > 1 else 'auto'

    log_dir = config['log_dir']
    output_dir = config['output_dir']
    if output_dir != None or "" or '':
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_name = config['log_name']
    save_dir = Path(output_dir) / log_name


    logger = pl.loggers.TensorBoardLogger(
        save_dir=log_dir,
        name=log_name,
        # default_hp_metric=False,    # 禁用 PyTorch Lightning 默认的 hparams 评估指标, 启用 TensorboardX
    )
    print('-------------\n',log_name, '\n----------------------')
    modelsummary_callback = pl.callbacks.RichModelSummary(max_depth=3)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir / f"version_{logger.version}",
        filename='step{step}-val_score{val_' + config["dataset"]["name"] + '/R@SUM:.4f}',
        auto_insert_metric_name=False,
        save_top_k=3,
        monitor=f'val_{config["dataset"]["name"]}/R@SUM',
        mode='max',
        save_last=False,
        verbose=True,
        save_weights_only=False,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [modelsummary_callback, checkpoint_callback, lr_callback]

    # early_stop_callback = pl.callbacks.EarlyStopping(
    #     monitor=f'val_{config["dataset"]["name"]}/R@SUM',
    #     patience=3,
    #     verbose=True,
    #     mode='max'
    # )
    # callbacks = [modelsummary_callback, checkpoint_callback, early_stop_callback, lr_callback]


    trainer = pl.Trainer(
        # resume_from_checkpoint=config['load_path'],
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        precision=config['precision'],
        # amp_backend='apex' if config['apex'] else "native",
        # amp_level=config['amp_level'] if config['apex'] else None,

        accelerator=config['accelerator'],
        devices=config['devices'],
        # gpus=config['gpus'],
        strategy=strategy,
        # strategy='ddp_find_unused_parameters_true',
        use_distributed_sampler=False,
        # enable_model_summary=True,
        # benchmark=True,
        max_epochs=config['max_epoch'],
        callbacks=callbacks,
        # gradient_clip_val=None if config['manual_optimization'] else config['max_grad_norm'],
        # accumulate_grad_batches=None if config['manual_optimization'] else grad_steps,
        gradient_clip_val=config['max_grad_norm'],
        accumulate_grad_batches=config['gradient_accumulation_steps'],

        fast_dev_run=config.get('fast_dev_run', False),

        limit_train_batches=config.get('limit_train_batches', None),
        limit_val_batches=config.get('limit_val_batches', None),
        limit_test_batches=config.get('limit_test_batches', None),
        limit_predict_batches=config.get('limit_predict_batches', None),

        num_sanity_val_steps=config['num_sanity_val_steps'],
        val_check_interval=config.get('val_check_interval', None),
        check_val_every_n_epoch=config.get('check_val_every_n_epoch', None),
    )

    if config['test_only']:
        weight_path = Path(config['checkpoint'])
        print('---------------------------------------------')
        print(weight_path)
        checkpoint = torch.load(str(weight_path), map_location='cpu', weights_only=False)
        checkpoint_config = rectify_config(checkpoint['hyper_parameters']['config'], config)

        dm = build_datamodule(checkpoint_config)
        # model = build_retrieval_module(checkpoint_config)
        model = build_retrieval_module_ckpt(checkpoint_config)
        # trainer.test(model, datamodule=dm, ckpt_path=str(weight_path))
        trainer.test(model, datamodule=dm)

    # if config['test_only']:
    #     weight_paths = list(sorted(Path(config['test_checkpoints_dir']).rglob('*.[pc][tk][hp]*')))
    #     for ckpt in weight_paths:
    #         print('---------------------------------------------')
    #         print(weight_paths)
    # #         checkpoint = torch.load(str(ckpt), map_location='cpu')
    # #         checkpoint_config = rectify_config(checkpoint['hyper_parameters']['config'], config)
    # #         # checkpoint_config = checkpoint['hyper_parameters']['config']
    # #         # checkpoint_config['coco_scale'] = config['coco_scale']
    # #         # checkpoint_config['checkpoint'] = str(ckpt)
    #         dm = build_datamodule(checkpoint_config)
    #         model = build_module(checkpoint_config)
    #         trainer.test(model, datamodule=dm, ckpt_path=str(ckpt))


    elif config['eval_only']:
        dm = build_datamodule(config)
        model = build_retrieval_module(config)

        # dm.setup('val')
        trainer.validate(model, datamodule=dm)

    else:
        dm = build_datamodule(config)
        model = build_retrieval_module(config)
        # model = init_model(config)

        if config['checkpoint'] != '':
            trainer.fit(model, datamodule=dm, ckpt_path=config['checkpoint'])
        else:
            trainer.fit(model, datamodule=dm)

            # test_after_train(trainer, model, dm)

