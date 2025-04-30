import torch
from tqdm.auto import tqdm

class Trainer:
  def __init__(self, model, loss, optimizer, device, clip_grad=False, is_parent = False, scheduler = None):
    self.model = model
    self.loss = loss
    self.optimizer = optimizer
    self.device = device
    self.clip_grad = clip_grad
    self.scheduler = scheduler
    self.is_parent = is_parent
 
 
  def train_step(self, train_data):
    self.model.train()

    train_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(train_data, desc="Training", leave=False)

    for batch in progress_bar:
      X1, X2, y = batch 
      X1, X2, y = X1.to(self.device), X2.to(self.device), y.to(self.device)

      if self.is_parent:
        y_pred_logits = self.model(X1, X2)
      else:
        y_pred_logits = self.model(X1)
  

      loss_fn = self.loss(y_pred_logits, y)
      train_loss +=loss_fn.item()

      self.optimizer.zero_grad()
      loss_fn.backward()

      #Gradient clipping 
      if self.clip_grad:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

      self.optimizer.step()
      if self.scheduler is not None:
        self.scheduler.step()

      y_pred_class = torch.argmax(y_pred_logits, dim=1)
      total_correct += (y_pred_class == y).sum().item()
      total_samples += y.size(0)

      progress_bar.set_postfix({
                "loss": f"{loss_fn.item():.4f}",
                "accuracy": f"{(total_correct/total_samples):.4f}"
            })

    train_loss /= len(train_data)
    train_acc = total_correct / total_samples

    return train_loss, train_acc


  def test_step(self, test_data):
    self.model.eval()

    test_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(test_data, desc="Testing", leave=False)

    with torch.inference_mode():
      
      for batch in progress_bar:
        X1, X2, y = batch 
        X1, X2, y = X1.to(self.device), X2.to(self.device), y.to(self.device)
        
        if self.is_parent:
          y_pred_logits = self.model(X1, X2)
        else: 
          y_pred_logits = self.model(X1)
       
        loss_fn = self.loss(y_pred_logits, y)
        test_loss +=loss_fn.item()

        y_pred_class = torch.argmax(y_pred_logits, dim=1)
        total_correct += (y_pred_class == y).sum().item()
        total_samples += y.size(0)

    test_loss /= len(test_data)
    test_acc = total_correct / total_samples

    return test_loss, test_acc


  def fit(self, train_data, test_data, epochs, start_epoch = 0, callbacks=[]):
  
    for epoch in tqdm(range(start_epoch, start_epoch + epochs)):
      train_loss, train_acc = self.train_step(train_data)
      val_loss, val_acc = self.test_step(test_data)
      print(f"Epoch: {epoch} | loss: {train_loss:.4f} | accuracy: {train_acc:.4f} | val_loss: {val_loss:.4f} | val_accuracy: {val_acc:.4f}")

      logs = {
            'epoch': epoch,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'state_dict': {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'sheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
            }
        }

      for cb in callbacks:
        cb.on_epoch_end(epoch, logs)
        if hasattr(cb, 'early_stop') and cb.early_stop:
                break
    for cb in callbacks:
        cb.on_train_end(logs)


    return logs
