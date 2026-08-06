from sacred import Ingredient

side_ingredient = Ingredient('side', save_git_info=False)



@side_ingredient.config
def base_config():
    network = ''
    channel_first = False
    img_size = 224
    patch_size = 16
    # ====================
    side_layers_mode = 'all'  # all, top, interval
    mamba_depth = 4
    layer_depth = -1
    hierarchical = False
    side_dim = -1
    # trans_dim = 768
    block_pipeline = ''  # ['sequential', 'parallel', 'bidirectional', '']

    #


    if_abs_pos_embed = True
    if_cls_token = True
    if_rope = False
    if_rope_residual = False
    use_middle_cls_token = False
    # ====================
    ssm_d_state = -1
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = ""
    forward_type = ""
    bimamba_type = ""
    select_type = ""  # 扫描方式    's' 'sequential' 连续扫描;   'e' 'efficient'  跳跃扫描
    route_type = ""   # 选择扫描路径  's' 'spatial';      't' 'temporal';   'st' 'spatiotemporal';   'sy' 'synthetic'
    step_size = -1
    if_bidirectional = None
    if_divide_out = None
    if_noz = None
    pos_type = 'learnable_2d'  # learnable_2d / 3d: 可学习的位置编码;  cos: 固定余弦编码，类似transformer;  mrope：旋转位置编码
    # ====================
    if_mlp = False
    rms_norm = True
    norm_epsilon = 1e-5
    mlp_ratio = 4  # 中间层的维度：hidden_size * mlp_ratio
    mlp_drop_rate = 0.0  # dropout
    # ====================
    drop_path_rate = 0.1
    norm_layer = "LN"  # "BN", "LN2D"
    use_dyt = False
    fused_add_norm = True
    residual_in_fp32 = True

    if_diff_features = False
    scaling_type = ''
    patch_type = 'none'
    if_patch_bidirectional = False
    cls_interaction = ''

@side_ingredient.named_config
def trans():
    network = 'trans'
    side_layers_mode = 'all'

    mamba_depth = 4
    side_dim = 320


@side_ingredient.named_config
def vmamba_12_i_patch16_224_v0_3d_e_r1_h2_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-B/16 :
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 16
    channel_first = False
    # =========================
    side_layers_mode = 'all'
    mamba_depth = 12
    layer_depth = 1
    hierarchical = True
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'


@side_ingredient.named_config
def vmamba_4_i_patch16_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-B/16 :
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 16
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'



@side_ingredient.named_config
def vmamba_12_i_patch16_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-B/16 : MSVD 472.1
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 16
    channel_first = False
    # =========================
    side_layers_mode = 'all'
    mamba_depth = 12
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'

@side_ingredient.named_config
def vmamba_12_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 12
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'

@side_ingredient.named_config
def vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'

@side_ingredient.named_config
def vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv8_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s", "t", "st", "sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN"  # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v8'



@side_ingredient.named_config
def vmamba_4_i_patch14_224_v0_3d_e_sy_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'



@side_ingredient.named_config
def vmamba_4_i_patch14_224_v05_s_sy_mm_noz_clsv7_dyt_3dpos():
    '''
    Vmamba ss2d 消融
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v05_noz"
    select_type = "s"
    route_type = ["sy"]
    # step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'


@side_ingredient.named_config
def vmamba_8_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 8
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'

@side_ingredient.named_config
def vmamba_8_i_patch14_224_v0_3d_e_sy_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 8
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'



@side_ingredient.named_config
def vmamba_12_i_patch14_224_v0_3d_e_sy_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 12
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'


@side_ingredient.named_config
def vmamba_small_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 8
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'


@side_ingredient.named_config
def vmamba_base_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  490.5
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 12
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'

@side_ingredient.named_config
def vmamba_base_t_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos():
    '''
    ViT-L/14 : MSVD  488.4
    '''
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'top'
    mamba_depth = 12
    layer_depth = 1
    side_dim = 192
    # trans_dim = 768
    # =========================
    ssm_d_state = 16
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'



@side_ingredient.named_config
def vmamba_4_i_patch14_224_v0_3d_e_r1_mm_noz_clsv7_dyt_3dpos_test():
    network = 'vmamba'
    img_size = 224
    patch_size = 14
    channel_first = False
    # =========================
    side_layers_mode = 'interval'
    mamba_depth = 4
    layer_depth = 1
    side_dim = 12

    # trans_dim = 768
    # =========================
    ssm_d_state = 4
    ssm_ratio = 1.0
    ssm_drop_rate = 0.0
    ssm_conv = 3
    ssm_conv_bias = True
    ssm_init = "v0"
    forward_type = "v0_3d"
    select_type = "e"
    route_type = ["s","t","st","sy"]
    step_size = 2
    if_noz = True
    if_bidirectional = True
    if_divide_out = True
    pos_type = 'learnable_3d'  #
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 1.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    use_dyt = True
    fused_add_norm = True
    residual_in_fp32 = True
    if_abs_pos_embed = True
    if_cls_token = True
    cls_interaction = 'v7'
