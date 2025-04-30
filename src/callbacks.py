import torch 

class Callback:
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStopping(Callback):
    def __init__(self, patience = 5, monitor='val_loss', mode='min', save_path='best_model.pth'):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.save_path = save_path
        self.best = float('inf') if mode== 'min' else -float('inf')
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        is_improvement = (current < self.best) if self.mode == 'min' else (current > self.best)
        if is_improvement:
            self.best = current
            self.counter = 0
            self.best_state = logs['state_dict']
            torch.save(self.best_state, self.save_path)
        else:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {epoch}. No improvement in {self.patience} epochs.")

    def on_train_end(self, logs=None):
        if self.best_state:
            print(f"Loading best model from {self.save_path}")

class LRShedulerCallback(Callback):
    def __init__(self, scheduler, monitor='val_loss', mode='min', experiment = None):
        self.sheduler = scheduler
        self.monitor = monitor
        self.mode = mode
        self.experiment = experiment 

    def on_epoch_end(self, epoch, logs = None):
        current = logs.get(self.monitor)
        self.scheduler.step(current)

        lr = self.sheduler.optimizer.param_groups[0]['lr']
        self.experiment.log_metric("lr", lr, step=epoch)
        print(f"[Epoch {epoch}] LR scheduler updated. Current LR: {lr:.6f}")