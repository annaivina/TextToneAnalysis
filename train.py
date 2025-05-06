import comet_ml
import torch
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, List
from src.lr_shedule import warmupcosine
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import string
import math

import os
import re
import os
import shutil
from tqdm.auto import tqdm
from collections import Counter

import logging 
import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_process import load_data
from src.models import CustomLSTMClassifier, TransformerSarcasmModel, DebertaFineTuned
from src.trainer import engine
from src import utils
from src.callbacks import EarlyStopping, LRShedulerCallback
from src.metrics import compute_metrics

from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import transformers 
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", 
                    handlers=[logging.StreamHandler()])

@hydra.main(config_path="configs", config_name="main_config", version_base="1.3")
def pipeline(cfg: DictConfig) ->None:

    logging.info(f"Running {cfg.mode} experiment: {cfg.experiment_name}")
    print(OmegaConf.to_yaml(cfg))

    logging.info("Loading the data from the DataLoader")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # or just "cuda
    else:
        device = torch.device("cpu")

    logging.info(f"Using {device} device for training")


    logging.info("Loading and processing the data")
    if cfg.model.name != "deberta":
        train_data, valid_data,_,voc_size, len_train = load_data(file_name=cfg.file_name, 
                                       is_transformer=cfg.isTransformer, 
                                       is_parent=cfg.model.params.isParent, 
                                       max_len=cfg.model.params.max_len, 
                                       min_freq=cfg.model.params.min_freq, batch_size=cfg.batch_size,
                                       frac_train=cfg.frac_train,
                                       frac_val=cfg.frac_val,
                                       frac_test=cfg.frac_test)
    else:
        train_data, valid_data,test_data,tokenizer = load_data(file_name=cfg.file_name, 
                                       is_deberta=cfg.model.params.isDeberta,
                                       model_checkpoint=cfg.model.params.checkpoint,
                                       frac_train=cfg.frac_train,
                                       frac_val=cfg.frac_val,
                                       frac_test=cfg.frac_test)
    #Logging into comet : NMotice that Deberta has looging into the Weights and biases
    if cfg.model.name != 'deberta':
        experiment = comet_ml.Experiment(api_key = os.getenv("COMET_API_KEY"),
                                              project_name=cfg.experiment_name,
                                              auto_param_logging=True,
                                              auto_metric_logging=True)
    

    #Lets load the models
    if cfg.model.name == 'lstm':
        logging.info("Loading LSTM sarcasm classifier")
        model = CustomLSTMClassifier.SarcasmClassifier(voc_size=voc_size,
                                                       embed_dim=cfg.model.params.embed_dim,
                                                       hidden_size=cfg.model.params.hidden_size,
                                                       is_parent = cfg.model.params.isParent,
                                                       multiplier=cfg.model.params.multiplier,
                                                       dropout=cfg.model.params.dropout).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.start_lr, weight_decay=cfg.weight_decay)
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=2,verbose=True)
        lr_sheduler = LRShedulerCallback(plateau_scheduler, monitor='val_loss', mode='min', experiment=experiment)


    elif cfg.model.name == 'transformer':
        logging.info("Loading transformer model")
        model = TransformerSarcasmModel.SarcasmTransformer(voc_size = voc_size,
                                                           seq_length = cfg.model.params.max_len,
                                                           embed_dim = cfg.model.params.embed_dim,
                                                           dense_dim = cfg.model.params.dense_dim,
                                                           num_heads = cfg.model.params.num_heads,
                                                           num_layers = cfg.model.params.num_layers,
                                                           dropout= cfg.model.params.dropout).to(device)
        
        steps_per_epoch = (len_train // cfg.batch_size) + 1
        decay_steps = steps_per_epoch * cfg.epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.start_lr, weight_decay=cfg.weight_decay) 
        scheduler = warmupcosine.WarmupCosineDecay(optimizer, total_steps=decay_steps, warmup_steps=steps_per_epoch*3, hold_steps=steps_per_epoch*2, start_lr=cfg.start_lr, target_lr=cfg.target_lr, alpha=cfg.alpha)

    elif cfg.model.name == 'deberta':
        model = DebertaFineTuned.DebertaClassifier(model_checkpoint = cfg.model.params.checkpoint, 
                                                   num_labels = cfg.model.params.num_labels,
                                                   freeze_layers=cfg.model.params.freeze_layers,
                                                   freeze_layers_num=cfg.model.params.freeze_layers_num)
        
        #Make the training arguments to pass them to the Trainer 
        training_args = TrainingArguments(output_dir=cfg.experiment_name,
                                  eval_strategy="epoch",
                                  save_strategy="epoch",
                                  learning_rate=cfg.start_lr,
                                  num_train_epochs=cfg.epochs,
                                  fp16=cfg.model.params.use_mixed_prec,
                                  per_device_train_batch_size=cfg.batch_size,
                                  per_device_eval_batch_size=cfg.batch_size,
                                  remove_unused_columns=False,
                                  weight_decay=cfg.weight_decay,
                                  save_total_limit=1,
                                  load_best_model_at_end=True,
                                  metric_for_best_model="accuracy",
                                  push_to_hub=False)

        
    #Define the loss function
    loss = torch.nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=cfg.callback.patience, monitor=cfg.callback.monitor, mode=cfg.callback.mode, save_path=cfg.callback.path)


    #Lets also include start form some checkpoint if nessesary
    start_epoch = 0
    if cfg.resume_from_checkpoint and os.path.exists(cfg.checkpoint_path):
        start_epoch = utils.restart_from_ecpoch(cfg, model, optimizer, scheduler, logging, device)

    if start_epoch > 0:
        logging.info(f"✅ Resumed training from checkpoint at epoch {start_epoch}")
    else:
        logging.info(f"🆕 Starting training from scratch")


    #Deberta has its own Trainer so I have to impelement this thing....
    if cfg.model.name == 'deberta':
    
        def custom_collate(batch):
            return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}

        trainer = transformers.Trainer(model,
                  train_dataset = test_data,#fall back to ttrain_data in GPU... 
                  eval_dataset = valid_data,
                  args = training_args,
                  compute_metrics = compute_metrics,
                  tokenizer = tokenizer)
                  #data_collator=lambda data: {k: torch.stack([f[k] for f in data]) for k in data[0]})
        
        trainer.train()
    
    else: 
        trainer = engine.Trainer(model=model,
                                  loss = loss,
                                  optimizer = optimizer,
                                  device = device,
                                  clip_grad = cfg.model.params.clip_grad,
                                  is_parent = cfg.model.params.isParent,
                                  scheduler = scheduler if cfg.model.name == 'transformer' else None)
        
        callbacks = [early_stop]
        if cfg.model.name == 'lstm':
            callbacks.append(lr_sheduler)
            
        
        results = trainer.fit(train_data, valid_data, epochs = cfg.epochs, callbacks=callbacks, start_epoch=start_epoch)
        




if __name__ == "__main__":
    pipeline()
