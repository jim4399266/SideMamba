from sacred import Ingredient

encoder_ingredient = Ingredient('encoder', save_git_info=False)


@encoder_ingredient.config
def base_config():
    pretrained_clip_name = ''
    pretrained_clip = ''  # The folder where the specific pretrained model weight is stored.
    cut_top_layer = 0
    embed_dim = -1

    tokenizer_name = ''
    tokenizer = ''   # The folder where the specific pretrained model weight is stored.
    # max_text_len = -1
    whole_word_masking = False  # note that whole_word_masking does not work for RoBERTa
    mlm_prob = 0.  # mlm遮罩比例
    text_width = -1
    vocab_size = -1

    vit_name = ""
    vit = ''  # The folder where the specific pretrained model weight is stored.
    patch_size = -1
    image_resolution = -1
    linear_patch = ''
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    visual_width = -1

    # feature_framerate = 1.0
    # max_frames = 12
    # train_frame_order = 0
    # eval_frame_order = 0
    # slice_frame_pos = 0
    # strategy = 1

    loose_type = True
    freeze_vit_encoder = True
    freeze_text_encoder = False
    freeze_layer_num = 1.0
    freeze_text_layer_num = 1.0


# ---------------  clip encoder ---------------------------------



@encoder_ingredient.named_config
def clip_vit_B_32():
    pretrained_clip_name = 'ViT-B/32'
    pretrained_clip = 'clip/ViT-B-32.pt'
    image_resolution = 224
    patch_size = 32
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    # visual_width = 768
    # feature_framerate = 1.0
    # max_frames = 12

    # slice_frame_pos = 0
    # strategy = 1

    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.
    # max_text_len = 32

@encoder_ingredient.named_config
def clip_vit_B_16():
    # embed_dim = 512
    pretrained_clip_name = 'ViT-B/16'
    pretrained_clip = 'clip/ViT-B-16.pt'
    image_resolution = 224
    patch_size = 16
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]

    # feature_framerate = 1.0
    # max_frames = 12

    # slice_frame_pos = 0
    # strategy = 1

    tokenizer = 'clip'   # The folder where the specific pretrained model weight is stored.
    # max_text_len = 32



@encoder_ingredient.named_config
def clip_vit_B_16_freeze():
    # embed_dim = 512
    pretrained_clip_name = 'ViT-B/16'
    pretrained_clip = 'clip/ViT-B-16.pt'
    image_resolution = 224
    patch_size = 16
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    # visual_width = 768
    # feature_framerate = 1.0
    # max_frames = 12

    # slice_frame_pos = 0
    # strategy = 1

    tokenizer = 'clip'   # The folder where the specific pretrained model weight is stored.
    # max_text_len = 32

    freeze_vit_encoder = True
    freeze_text_encoder = True

@encoder_ingredient.named_config
def clip_vit_L_14():
    pretrained_clip_name = 'ViT-L/14'
    pretrained_clip = 'clip/ViT-L-14.pt'
    image_resolution = 224
    patch_size = 14
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    # visual_width = 768
    # feature_framerate = 1.0
    # max_frames = 12

    # slice_frame_pos = 0
    # strategy = 1

    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.
    # max_text_len = 32


@encoder_ingredient.named_config
def clip_vit_L_14_freeze():
    pretrained_clip_name = 'ViT-L/14'
    pretrained_clip = 'clip/ViT-L-14.pt'
    image_resolution = 224
    patch_size = 14
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]

    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.
    freeze_text_encoder = True
    # freeze_text_layer_num = 0



@encoder_ingredient.named_config
def eva_clip_bigE_14():
    pretrained_clip_name = 'EVA02-CLIP-bigE-14'
    pretrained_clip = 'clip/EVA02_CLIP_E_psz14_s4B.pt'
    image_resolution = 224
    patch_size = 14
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.

@encoder_ingredient.named_config
def eva_clip_bigE_14_freeze():
    pretrained_clip_name = 'EVA02-CLIP-bigE-14'
    pretrained_clip = 'clip/EVA02_CLIP_E_psz14_s4B.pt'
    image_resolution = 224
    patch_size = 14
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.
    freeze_text_encoder = True
    # freeze_text_layer_num = 0.5


@encoder_ingredient.named_config
def eva_clip_bigE_14_p():
    pretrained_clip_name = 'EVA02-CLIP-bigE-14-plus'
    pretrained_clip = 'clip/EVA02_CLIP_E_psz14_plus_s9B.pt'
    image_resolution = 224
    patch_size = 14
    linear_patch = '2d'
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    tokenizer = 'clip'  # The folder where the specific pretrained model weight is stored.

# ---------------  text encoder ---------------------------------
@encoder_ingredient.named_config
def text_roberta():
    tokenizer_name = "roberta-base"
    tokenizer = 'en-roberta-base'
    vocab_size = 50265
    input_text_embed_size = 768

@encoder_ingredient.named_config
def text_roberta_large():
    tokenizer_name = "roberta-large"
    tokenizer = 'en-roberta-large'
    vocab_size = 50265
    input_text_embed_size = 1024


# ---------------  visual encoder ---------------------------------
@encoder_ingredient.config
def swin32_large384():
    vit_name = ""
    vit = ''  # The folder where the specific pretrained model weight is stored.
    patch_size = 32
    image_resolution = 384
    frames_transform_keys = ["imagenet"]
    video_transform_keys = ["imagenet"]
    input_image_embed_size = 1536

@encoder_ingredient.named_config
def swin32_large384():
    vit_name = "swin_large_patch4_window12_384_in22k"
    vit = ''

    patch_size = 32
    image_resolution = 384
    frames_transform_keys = ["imagenet"]
    video_transform_keys = ["imagenet"]
    input_image_embed_size = 1536

@encoder_ingredient.named_config
def vit_B_32():
    vit_name = 'ViT-B/32'
    vit = ''

    image_resolution = 224
    patch_size = 32
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    input_image_embed_size = 768

@encoder_ingredient.named_config
def vit_B_16():
    vit_name = 'ViT-B/16'
    vit = ''

    image_resolution = 224
    patch_size = 16
    frames_transform_keys = ["clip"]
    video_transform_keys = ["clip"]
    input_image_embed_size = 768

# random augmentation
# @encoder_ingredient.named_config
# def imagenet_randaug():
#     train_transform_keys = ["imagenet_randaug"]
#     val_transform_keys = ["imagenet_randaug"]
#
# @encoder_ingredient.named_config
# def clip_randaug():
#     train_transform_keys = ["clip_randaug"]
#     val_transform_keys = ["clip_randaug"]