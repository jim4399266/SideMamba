from sacred import Ingredient

dataset_ingredient = Ingredient('dataset', save_git_info=False)

@dataset_ingredient.config
def base_config():
    name = ''  # 数据集信息
    data_dir = ''
    features_dir = ''
    max_text_len = -1
    feature_framerate = 1
    max_frames = 12

    slice_frame_pos = 2
    train_strategy = 3
    eval_strategy = 1


    # ----------------------  Dataset Setting  ------------------------------


@dataset_ingredient.named_config
def msvd():
    name = 'MSVD'
    data_dir = 'MSVD-Frames'
    features_dir = 'MSVD-Frames/MSVD_frames'
    max_text_len = 32
    feature_framerate = 1
    max_frames = 12

    slice_frame_pos = 2
    train_strategy = 3
    eval_strategy = 1


@dataset_ingredient.named_config
def msrvtt():
    name = 'MSRVTT'
    data_dir = 'MSRVTT'
    features_dir = 'MSRVTT/frames_30fps'
    max_text_len = 32
    feature_framerate = 1
    # max_frames = 100
    max_frames = 12

    slice_frame_pos = 2
    train_strategy = 3
    eval_strategy = 1



@dataset_ingredient.named_config
def didemo():
    name = 'Didemo'
    data_dir = 'DiDeMo/text'
    features_dir = 'DiDeMo/DiDeMo_Frames'
    max_text_len = 64
    feature_framerate = 1
    max_frames = 64

    slice_frame_pos = 2
    train_strategy = 3
    eval_strategy = 1

