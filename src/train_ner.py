from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from utils_ner import *
from transformers import AutoTokenizer
import wandb
import os

@hydra.main(config_path="../conf", config_name="cfg")
def train_ner(cfg: DictConfig) -> None:
    # load data
    data = load_dataset(path = cfg.symptemist_bigbio_path, name = "symptemist_entities_bigbio_kb")

    # preprocess data
    data = tokenize_split(data["train"], label2id, cfg.model)
    
    # make splits
    train_set = data.select(range(595))
    eval_set = data.select(range(595,744))

    # train  model
    with wandb.init(project=f'symptemist_ner', tags=["dev"]):
        train(train_set, eval_set, cfg.model, cfg.arguments)

    pass


if __name__ == "__main__":
    train_ner()
