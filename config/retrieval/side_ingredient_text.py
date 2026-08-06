from sacred import Ingredient

side_ingredient_t = Ingredient('side_t', save_git_info=False)

@side_ingredient_t.config
def base_config():
    network = ''
    side_layers_mode = 'all'
    seq_len = 32
    # ====================
    mamba_depth = 4
    layer_depth = -1
    side_dim = -1
    # trans_dim = 768
    block_pipeline = ''  # ['sequential', 'parallel', 'bidirectional', '']

    if_abs_pos_embed = True
    if_cls_token = True

    # ====================
    ssm_d_state = -1
    ssm_ratio = 2.0
    ssm_drop_rate = 0.0

    ssm_init = ""
    forward_type = ""
    bimamba_type = ""
    if_divide_out = True
    # ====================
    if_mlp = False
    rms_norm = True
    norm_epsilon = 1e-5
    mlp_ratio = 4  # 中间层的维度：hidden_size * mlp_ratio
    mlp_drop_rate = 0.0  # dropout
    # ====================
    drop_path_rate = 0.1
    norm_layer = "LN"  # "BN", "LN2D"
    fused_add_norm = True
    residual_in_fp32 = True


@side_ingredient_t.named_config
def bimamba2_v2_tiny_seq32_firstcls_mlp():
    network = 'mamba2'
    seq_len = 32
    ## 在Mamba2中， embed_dim * expend / headdim 要是 8 的倍数
    ## 在Mamba2中 expend 一般为 2 ，  headdim 一般为 64 ，因此 embed_dim 要为 256 的倍数

    side_dim = 256  # Model dimension d_model
    ssm_d_state = 64  # SSM state expansion factor, typically 64 or 128
    mamba_depth = 12
    layer_depth = 2
    rms_norm = True
    residual_in_fp32 = True
    fused_add_norm = True
    if_abs_pos_embed = True

    bimamba_type = "v2"
    if_cls_token = True
    if_divide_out = True
    if_mlp = True
    block_pipeline = 'sequential' # ['sequential', 'parallel', 'bidirectional', '']

@side_ingredient_t.named_config
def bimamba2_v2_tiny_seq32_firstcls():
    # (b t) l dilation  sum:    T2V:  R1     R5    R10       V2T: R1    R5    R10
    network = 'mamba2'
    seq_len = 32
    ## 在Mamba2中， embed_dim * expend / headdim 要是 8 的倍数
    ## 在Mamba2中 expend 一般为 2 ，  headdim 一般为 64 ，因此 embed_dim 要为 256 的倍数

    side_dim = 256  # Model dimension d_model
    ssm_d_state = 64  # SSM state expansion factor, typically 64 or 128
    mamba_depth = 12
    layer_depth = 2
    rms_norm = True
    residual_in_fp32 = True
    fused_add_norm = True
    if_abs_pos_embed = True

    bimamba_type = "v2"
    if_cls_token = True
    if_divide_out = True

    if_mlp = False

    block_pipeline = 'sequential' # ['sequential', 'parallel', 'bidirectional', '']




@side_ingredient_t.named_config
def vmamba_tiny_patch16_224_v05_noz():
    network = 'vmamba'
    img_size = 224
    patch_size = 16
    channel_first = False
    # =========================
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
    forward_type = "v05_noz"
    # =========================
    norm_epsilon = 1e-5
    rms_norm = True
    mlp_ratio = 4.0
    mlp_drop_rate = 0.0
    # =========================
    drop_path_rate = 0.1
    norm_layer = "LN" # "BN", "LN2D"
    fused_add_norm = True
    residual_in_fp32 = True
    if_bidirectional = False
    if_abs_pos_embed = True
    bimamba_type = "v2"
    if_cls_token = True
