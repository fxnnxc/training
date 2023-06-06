
import os 
import torch 
import json 
from .abstract_trainer import AbstractTrainer

class ImageClassfierTrainer(AbstractTrainer):
    def __init__(self, model, 
                 device,
                 save_dir, 
                 save_freq,
                 epochs,
                 lr,
                 batch_size,
                 num_workers,
                 train_dataset,
                 optim_type='sgd', 
                 scheduler_type='cosine',
                 clip_grad=1.0,
                 valid_dataset=None,
                 eval_freq=1,
                 args=None,
                 **kwargs):
        
        super().__init__( model,
                         device,
                        save_dir, 
                        save_freq,
                        epochs,
                        lr,
                        batch_size,
                        num_workers,
                        train_dataset,
                        optim_type, 
                        scheduler_type,
                        args)
       
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.history_flags.clip_grad = clip_grad
        self.valid_dataset = valid_dataset
        self.eval_freq = eval_freq

        self.history_flags.best_performanace =None 
        self.postfix = {}
        self.epoch_results = {
            'accuracy' : [],
            'running_loss' : [0 for i in range(epochs)],
            'last_lr' : [],
        }

    
    def init_callback(self, **kwargs):
        super().init_callback()
        
    def batch_callback(self, epoch, step, batch, **kwargs):
        super().batch_callback(epoch, step, batch,)
        x,y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)                

        self.optimizer.zero_grad()                                
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.history_flags.clip_grad)
        self.optimizer.step()
        self.epoch_results['running_loss'][epoch] +=  loss.item() 
        
        if step % 100 ==0:
            self.pbar.set_postfix({'running_loss': f"{self.epoch_results['running_loss'][epoch]/(step+1):.2E}"})
        
        
    def epoch_callback(self, epoch, **kwargs):
        super().epoch_callback(epoch)
        self.epoch_results['running_loss'][epoch] = \
                self.epoch_results['running_loss'][epoch] / self.history_flags.num_epoch_steps
        
        if epoch % self.eval_freq ==0:
            if self.valid_dataset is not None:
                acc = self.__class__.evaluate(
                    self.model, 
                    self.valid_dataset,
                    self.batch_size,
                    self.device, 
                    num_workers=self.num_workers,
                    prefix=f"--👾 {self.model.__class__.__name__}|{self.save_dir}|[Epoch:{epoch+1}/{self.epochs}]"
                )
                self.history_flags
                self.epoch_results['accuracy'].append(acc)
            
                if self.history_flags.best_performanace is None or acc > self.history_flags.best_performanace:  
                    self.history_flags.best_performanace = acc 
                    torch.save(self.model.state_dict(), os.path.join(self.history_dir, f'model_best.pt'))
                    
            
        if self.lr_scheduler is not None:
            self.epoch_results['last_lr'].append(self.lr_scheduler.get_last_lr()[0])
        json.dump(self.epoch_results, open(os.path.join(self.history_dir, 'eopch_results.json'), 'w'), indent=1)
        
    
    def train_callback(self, **kwargs):
        super().train_callback()

    
    
    @staticmethod
    def evaluate(model, dataset, batch_size, device, num_workers, prefix=''):
        from tqdm import tqdm 
        from torch.utils.data import DataLoader
        
        data_loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        is_train = model.training 
        model.eval()
        eq = 0
        print()
        with tqdm(total=len(data_loader)) as pbar:
            pbar.set_description(prefix)
            for step, batch in enumerate(data_loader):
                    # ---- start of an minibatch
                pbar.update(1)
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                with torch.no_grad():
                    y_hat = model(x).argmax(dim=-1)                
                eq += (y == y_hat).sum()  
            # average
            acc = (eq/len(dataset)).item()
            pbar.set_postfix({"eval_accuracy":f"{acc:.3f}"})
        if is_train:
            model.train()
        return acc 