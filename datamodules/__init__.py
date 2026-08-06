from typing import List

from src.datamodules.retrieval.datamodule_msvd import MSVDDataModule
from src.datamodules.retrieval.datamodule_msrvtt import MSRVTTDataModule
from src.datamodules.retrieval.datamodule_didemo import DidemoDataModule

_datamodules = {
    "msvd": MSVDDataModule,
    "msrvtt": MSRVTTDataModule,
    "didemo": DidemoDataModule,
}

def build_datamodule(config):
    print('### building datamodule. ###')
    dataset = config['dataset']['name'].lower()
    if isinstance(dataset, List) and len(dataset) > 1:
        return None # 多数据集无监督训练
    elif isinstance(dataset, List) and len(dataset) == 1:
        return _datamodules[dataset[0]](config)
    elif isinstance(dataset, str):
        return _datamodules[dataset](config)
    else:
        raise ValueError('dataset must be either a list or a string')

