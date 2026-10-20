from torch.utils.data import DataLoader

from train_config import TrainConfig
from eval_config import EvalConfig

from .train_dataset import TrainDataset
from .eval_dataset import EvalDataset


def create_training_data_loader(config: TrainConfig, offset: int = 0) -> DataLoader:
    data_set = TrainDataset(config, offset)
    return DataLoader(data_set, batch_size=config.batch_size)


def create_eval_data_loader(config: EvalConfig, offset: int = 0) -> DataLoader:
    data_set = EvalDataset(config, offset)
    return DataLoader(data_set, batch_size=1)
