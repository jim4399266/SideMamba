from sacred import Ingredient

optimizer_ingredient = Ingredient('optimizer', save_git_info=False)


@optimizer_ingredient.config
def base_config():
    name = 'bert'
    init_lr = 1e-4         # lr for other networks outside the backbone (side transformer ...)
    coef_lr = 1e-2        # coefficient of lr for the backbone(visual encoder and text encoder)   init_lr * coef_lr
    clip_ratio = 1

    min_lr = 0
    eps = 1e-8
    betas = (0.9, 0.98)
    weight_decay = 0.2

    scheduler = 'cosine'
    num_cycles = 0.3


@optimizer_ingredient.named_config
def adamw():
    name = 'adamw'
    init_lr = 1e-4  # lr for the backbone(visual encoder and text encoder)
    coef_lr = 1e-2

    min_lr = 0
    eps = 1e-8
    betas = (0.9, 0.98)
    weight_decay = 0.2

    scheduler = 'cosine'
    num_cycles = 0.3


@optimizer_ingredient.named_config
def adamw_rec():
    name = 'adamw'
    init_lr = 1e-4  # lr for the backbone(visual encoder and text encoder)
    clip_ratio = 1e-2

    min_lr = 0
    eps = 1e-8
    betas = (0.9, 0.98)
    weight_decay = 0.2

    scheduler = 'cosine'
    num_cycles = 0.3