import torch 
import random 
import argparse
import numpy as np


from training.models.cnn import JihyeonCNNClassifier
from training.models.lstm import JihyeonLSTMClassifier
from training.data import get_jihyeon_dataset
from training.trainers import ImageClassfierTrainer

seed=0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True 
# ------------ MNIST Training SCripts ------------------------------------------------

parser =  argparse.ArgumentParser()
parser.add_argument("--data-path", required=True, help='[Required] data path')
parser.add_argument("--epochs",    type=int, required=True, help='[Required] epochs')
parser.add_argument("--model", required=True, help='[Required] model class name')
parser.add_argument("--lr",        type=float, required=True, help='[Required] initial epochs')
parser.add_argument("--hidden-dim", type=int, required=True, help='hidden_dim')


parser.add_argument("--save-freq", type=int, default=-1, help='number of saving files, default:5')
parser.add_argument("--batch-size",    type=int, default=32, help='batch size, default=32')
parser.add_argument("--num-workers",    type=int, default=4, help='number of workers, default=4')
parser.add_argument('--root', default='results', help='root to the base path root/<data>/<task>/<model>')
parser.add_argument("--eval-freq", type=int, default=1, help='evaluation frequency of epoch, default:1')
parser.add_argument('--optim-type')
parser.add_argument('--scheduler-type')
args = parser.parse_args()

model = eval(args.model)(args.hidden_dim)
train_dataset, valid_dataset, info = get_jihyeon_dataset(args.data_path, None, None)

name = __file__.split("/")[-1]
data = name.split("_")[0]
task = name.split("_")[1].split(".")[0]
save_dir = f'{args.root}/{data}/{task}/{model.__class__.__name__}'

trainer = ImageClassfierTrainer(model, 
                                'cuda:0',
                                save_dir, 
                                args.save_freq,
                                args.epochs,
                                args.lr,
                                args.batch_size,
                                args.num_workers,
                                train_dataset,
                                optim_type=args.optim_type, 
                                scheduler_type=args.scheduler_type ,
                                clip_grad=1.0,
                                valid_dataset=valid_dataset,
                                eval_freq=args.eval_freq,
                                args=args
                                )


torch.set_num_threads(args.num_workers)
trainer.train()