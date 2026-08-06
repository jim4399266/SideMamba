from .module_retrieval_eva_clip import EVACLIPRetrievalModule
from .module_retrieval_clip import CLIPRetrievalModule
# from .module_retrieval_clip_2 import CLIPRetrievalModule as CLIPRetrievalModule2

# _models = {
#     "meanp":CLIPRetrievalModule,
# }

def build_retrieval_module(config):
    print('### building retrieval model. ###')
    # arch = config['arch'].lower()
    # sim_header = config['sim_header'].lower()
    model_name = config['encoder']['pretrained_clip_name']

    if 'EVA' in model_name:
        return EVACLIPRetrievalModule.from_pretrained(config)
    elif 'ViT' in model_name:
        return CLIPRetrievalModule.from_pretrained(config)
    else:
        raise NotImplementedError

def build_retrieval_module_config(config):
    print('### building retrieval model. ###')

    model_name = config['encoder']['pretrained_clip_name']

    if 'EVA' in model_name:
        return EVACLIPRetrievalModule.from_pretrained(config)
    elif 'ViT' in model_name:
        return CLIPRetrievalModule.from_config(config)
    else:
        raise NotImplementedError

def build_retrieval_module_ckpt(config):
    print('### building retrieval model. ###')

    model_name = config['encoder']['pretrained_clip_name']

    if 'EVA' in model_name:
        pass
    elif 'ViT' in model_name:
        return CLIPRetrievalModule.from_checkpoint(config)
    else:
        raise NotImplementedError

