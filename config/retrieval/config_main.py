from sacred import Experiment
from sacred.config.custom_containers import ReadOnlyDict
import sys
sys.path.append('..')
from .dataset_ingredient import dataset_ingredient
from src.config.retrieval.encoder_ingredient import encoder_ingredient
from src.config.retrieval.side_ingredient import side_ingredient
from src.config.retrieval.optimizer_ingerdient import optimizer_ingredient
from src.config.retrieval.side_ingredient_text import side_ingredient_t

ex = Experiment('video-text retrieval', save_git_info=False, ingredients=[dataset_ingredient, side_ingredient, side_ingredient_t, encoder_ingredient, optimizer_ingredient])
'''
视频-文本检索配置文件
main: 文件路径，训练总体设置，Lightning 框架设置
dataset_ingredient: 数据集设置
side_ingredient: 视觉侧网络配置
side_ingredient_t: 文本侧网络配置（不用）
encoder_ingredient: CLIP 主干配置
optimizer_ingredient: 优化器配置
'''

# with dataset.didemo encoder.clip_vit_L_14 side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos devices=2 batch_size=128 per_gpu_batch_size=8 batch_size_val=20 autodl_path


@ex.config
def config():
    # ----------------------  Path Setting  ----------------------
    data_root = "/home/tzj/datas"               # 存放所有数据集的路径
    output_dir = "../output_retrieval"          # 保存checkpoints的路径
    log_dir = "../log_retrieval"                # 存放log的路径
    test_checkpoints_dir = ""                   # 加载测试模型的路径
    # checkpoint = '/home/tzj/codes/my_video_new/output_retrieval/MSRVTT/dataset.msrvtt encoder.clip_vit_L_14 side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos devices=8 batch_size=320 per_gpu_batch_size=40 autodl_path max_epoch=10_bs320_pbs40_epoch10_lr0.0001_from_/version_0/step4776-val_score427.4064.ckpt'                             # 加载checkpoints的路径
    # checkpoint = '/home/tzj/codes/my_video_new/output_retrieval/MSVD/dataset.msvd encoder.clip_vit_L_14 side.vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos devices=8 batch_size=320 per_gpu_batch_size=40 autodl_path_bs320_pbs40_epoch5_lr0.0001_from_/version_1/step722-val_score495.9356.ckpt'                             # 加载checkpoints的路径
    # checkpoint = '/home/tzj/codes/my_video_new/output_retrieval/step8436-val_score422.1399.ckpt'                             # 加载checkpoints的路径
    checkpoint = ''
    pretrained_model_dir = '/home/tzj/pretrained_models' # 保存所有预训练模型的路径


    # ----------------------  Experiment Setting  ----------------------
    param = ''         # 参数备注
    desc = ''          # 结构备注

    statistic = False
    seed = 42
    log_every_n_steps = 50
    # train_dataset_len = -1
    # val_dataset_len = -1
    # test_dataset_len = -1

    batch_size = 128  # this is a desired batch size; pl trainer will accumulate gradients when per step batch is smaller.
    per_gpu_batch_size = 16  # you should define this manually with per_gpu_batch_size=#
    batch_size_val = 32
    queue_size_ratio = 2
    use_queue = False
    sim_header = 'meanP'
    interaction = 'wti'  # Original hard-max weighted token interaction.
    # interaction = 'wti_segment'  # Match short continuous temporal segments instead of isolated frames.
    # interaction = 'query_ware'  # Query-conditioned soft token/frame aggregation.
    # interaction = 'wti_topk'  # Fixed-ratio hard WTI + Top-k baseline.
    # interaction = 'wti_topk_learnable'  # Learnable monotonic Top-k rank weighting.
    # query_ware_temperature = 0.07
    # wti_topk = 3
    # wti_topk_ratio = 0.2
    # wti_segment_kernel = [0.25, 0.5, 0.25]

    visual_hidden = False
    visual_all_hidden = False

    # get_recall_metric = True
    # top_k = 64
    # queue_size = 1000
    # distill = False
    # momentum = 0.995
    # alpha = 0.4
    # negative_all_rank = True
    # coco_scale = [''] #测试集使用1k，5k还是都用


    # ----------------------  Lightning Trainer Setting  ------------------------------
    num_sanity_val_steps = 0  # 在开始前取 n 个val batches
    fast_dev_run = False  # 快速检验，取 n 个train, val, test batches
    val_check_interval = 0.25 # 验证间隔（浮点数为每X个epoch验证一次，整数为每X个step验证一次）
    check_val_every_n_epoch = 1  # 每几个epoch验证一次
    accelerator = 'gpu'
    devices = [0]
    num_nodes = 1
    pin_memory = True
    num_workers = 8
    precision = '16-mixed'
    # precision = '16-mixed'
    max_grad_norm = 1.0
    max_epoch = 5
    max_steps = -1
    warmup_steps = 0.1
    shuffle = True # 训练集是否打乱
    limit_train_batches = 1.0
    limit_val_batches = 1.0
    limit_test_batches = 1.0
    limit_predict_batches = 1.0

    debug = False
    eval_only = False
    test_only = False

    # ----------------------  Path Setting  ------------------------------
@ex.named_config
def cad04_path():
    data_root = "/mnt/ssd0"
    output_dir = "../output_retrieval"
    log_dir = "../log_retrieval"

@ex.named_config
def local_path():
    data_root = "/home/tzj/datas"
    output_dir = "../output_retrieval"
    log_dir = "../log_retrieval"
    test_checkpoints_dir = ""
    # checkpoint = ''
    pretrained_model_dir = '/home/tzj/pretrained_models'

@ex.named_config
def autodl_path():
    data_root = "/root/autodl-tmp/datas"
    output_dir = "../output_retrieval"
    log_dir = "/root/tf-logs"
    test_checkpoints_dir = ""
    # checkpoint = ''
    pretrained_model_dir = '/root/autodl-tmp/pretrained_models'
    # pretrained_model_dir = '/root/autodl-fs/pretrained_models'


    # ----------------------  Experiment Setting  ------------------------------

@ex.named_config
def debug():
    statistic = True
    # per_gpu_batch_size, batch_size_val = 10, 20
    # batch_size = per_gpu_batch_size


    # limit_train_batches = int(160)
    # limit_val_batches =  int(30)
    # limit_test_batches = int(50)
    # limit_predict_batches = int(10)

    log_every_n_steps = 1
    check_val_every_n_epoch = 1

    # num_sanity_val_steps = 5
    fast_dev_run = 5
    # eval_only = True
    shuffle = False
    num_workers = 0

    debug = True

@ex.named_config
def eval_only():
    eval_only = True


@ex.named_config
def test_only():
    test_only = True
    checkpoint = "/root/autodl-tmp/step722-val_score495.9356.ckpt"

@ex.capture
class Map(ReadOnlyDict):
    '''
    _config 默认返回的是一个字典, 调用参数时需要大量的 [""] 符号, 因此实现了一个映射的功能, 把字典的键值对映射为了成员变量, 可以直接通过 . 来调用. 这个映射支持字典的嵌套映射。
    '''
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, obj, **kwargs):
        new_dict = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_dict[k] = Map(v)
                else:
                    new_dict[k] = v
        else:
            raise TypeError(f"`obj` must be a dict, got {type(obj)}")
        super().__init__(new_dict, **kwargs)

# config = Map(_config)
