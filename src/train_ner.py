from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from utils_ner import *
from transformers import AutoTokenizer, TrainingArguments
import wandb
import os
from span_marker import Trainer
from span_marker import SpanMarkerModel


@hydra.main(config_path="../conf", config_name="cfg")
def train_spanmarker(cfg: DictConfig) -> None:
    data = load_dataset(path=cfg.symptemist_bigbio_path, name="symptemist_entities_bigbio_kb"    )
    data = data.sort("document_id")

    data_train = data["train"].select(range(600))
    data_eval = data["train"].select(range(600,744))

    data_train = bigbio2spanmarker(data_train)
    data_eval = bigbio2spanmarker(data_eval)

    labels = ["O", "SINTOMA"]
    model = SpanMarkerModel.from_pretrained(cfg.model, labels=labels, model_max_length=256, entity_max_length=15)

    args = TrainingArguments(
        output_dir = f"{cfg.arguments.output_dir}/v2",
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        fp16 = True,
        save_strategy = "epoch",
        overwrite_output_dir = True,
        evaluation_strategy = "epoch",
        save_total_limit = 2,
        num_train_epochs = 30,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_overall_f1",
        lr_scheduler_type = "linear",
        learning_rate = 0.00005,
        warmup_ratio = 0.0,
        label_smoothing_factor = 0.0,
        weight_decay = 0.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset = data_train,
        eval_dataset = data_eval,
    )

    with wandb.init(project=f'symptemist_ner',tags=["dev"]):
        trainer.train()

    pass


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
    train_spanmarker()
