import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch 

def train_val_test_split(df, frac_train=0.8, frac_val=0.1, frac_test=0.1):
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6, "Fractions must sum to 1"

    #For some reason the text has some NaN values, remove it
    df = df.dropna(subset=['comment', 'label'])

    train_ds = df.sample(frac=frac_train, random_state=42)
    temp_ds = df.drop(train_ds.index)

    val_ds = temp_ds.sample(frac=frac_val/(frac_val + frac_test), random_state=42)
    test_ds = temp_ds.drop(val_ds.index)

    return train_ds, val_ds, test_ds


def calc_result(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    results = {
              'accuracy': accuracy,
              'precision': precision,
              'recall': recall,
              'fscore': fscore
    }

    return results 

    

def pair_comments(input_1, input_2):
  new_input = []
  for i in range(len(input_1)):
    new_input.append(input_1[i] + " <SEP> " + input_2[i])
  return new_input


def restart_from_ecpoch(cfg, model, optimizer, scheduler, logging, device):
  logging.info(f"Resuming training from checkpoint: {cfg.checkpoint_path}")
  checkpoint = torch.load(cfg.checkpoint_path, map_location=device)
  
  model.load_state_dict(checkpoint["model_state_dict"])
  optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
  
  if "scheduler_state_dict" in checkpoint and scheduler:
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  
  start_epoch = checkpoint["epoch"] + 1
  logging.info(f"Resuming from epoch {start_epoch}")

  return start_epoch
    