import os 
from tqdm import tqdm 
import datetime 
from omegaconf import OmegaConf
from .utils import make_lr_scheduler, make_optimizer
from torch.utils.data import DataLoader
import torch 

class AbstractTrainer():
    def __init__(self, 
                 model,
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
                 args,
                 **kwargs):
        self.device= device 
        self.model = model.to(device)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = train_dataset
        self.start_date = datetime.datetime.now().strftime("%m_%d_%H:%M:%S")
        self.history_dir = os.path.join(self.save_dir, 'weights', self.start_date)
        

        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.optimizer = make_optimizer(self.model.parameters(), optim_type, lr=lr, **kwargs)
        self.lr_scheduler = make_lr_scheduler(self.optimizer, scheduler_type, epochs=epochs, **kwargs)
        
        new_experiment = {
                        'lr' : lr,
                        'bach_size' : batch_size,
                        'num_training_steps' : epochs * len(self.train_dataloader),
                        'num_epoch_steps' : len(self.train_dataloader),
                        'epochs' : epochs,
                        'optim_type' : optim_type,
                        'scheduler_type' :scheduler_type,
        }
        
        if os.path.exists(os.path.join(self.save_dir, 'config.yaml')):
            self.flags = OmegaConf.load(os.path.join(self.save_dir, 'config.yaml'))         
            self.flags.history[self.start_date] = new_experiment
            
        else:
            self.flags =  OmegaConf.create({
                'history' : {self.start_date:new_experiment}
            })
        
        
        # --- History ---
        self.history_flags = self.flags.history[self.start_date]
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        if args is not None:
            for k,v in vars(args).items():
                self.history_flags[k] = v

        OmegaConf.save(self.flags, os.path.join(self.save_dir, 'config.yaml'))
        self.init_callback(**kwargs)
        
    def train(self):
        with tqdm(total=self.history_flags.num_training_steps) as pbar:
            # ---- start of the train
            self.pbar = pbar 
            self.model.train()
            for epoch in range(self.epochs):
                # ---- start of an epoch
                self.pbar.set_description(f"--👾 {self.model.__class__.__name__}|{self.save_dir}|[Epoch:{epoch+1}/{self.epochs}]")
                for step, batch in enumerate(self.train_dataloader):
                    # ---- start of an minibatch
                    pbar.update(1)
                    self.batch_callback(epoch, step, batch)
                    # ---- end of minibatch
                # ---- end of epoch 
                self.epoch_callback(epoch)
            # ---- end of train
            self.train_callback()
            
            
    def init_callback(self, **kwargs):
        pass
        
    def batch_callback(self, epoch, step, batch, **kwargs):
        if self.save_freq > 0 and self.pbar.n % (self.history_flags.num_training_steps // self.save_freq) == 0:
            ratio = self.pbar.n/self.history_flags.num_training_steps 
            torch.save(self.model.state_dict(), os.path.join(self.history_dir, f'model_{ratio:.2f}.pt'))
            torch.save(self.model.state_dict(), os.path.join(self.history_dir, f'model_last.pt'))
            
    
    def epoch_callback(self, epoch, **kwargs):
        if self.lr_scheduler is not None:
            self.flags.history[self.start_date]['last_lr'] = self.lr_scheduler.get_last_lr()[0]
            self.lr_scheduler.step()
        OmegaConf.save(self.flags, os.path.join(self.save_dir, 'config.yaml'))
    
    def train_callback(self, **kwargs):
        OmegaConf.save(self.flags, os.path.join(self.save_dir, 'config.yaml'))
        torch.save(self.model.state_dict(), os.path.join(self.history_dir, f'model_last.pt'))

    