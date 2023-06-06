
from .models import * 



import os 
from omegaconf import OmegaConf
import torch 

def load(hub_path, data, task, model_name, version, map_location='cpu'):
    if model_name in NAMES.keys():
        model_name= NAMES[model_name].__name__
    
    model = torch.load(os.path.join(
        hub_path, data, task, model_name, version, 'model_best.pt'
            ), map_location=map_location)
    
    flags = OmegaConf.load(os.path.join(
        hub_path, data, task, model_name, version, 'config.yaml'
    ))
    return model, flags



def get_model_cls(name):
    return NAMES[name]

if __name__ == "__main__":
    import os 

    hub_path = 'hub'

    print(" =======================================Your Hub Directory ===========================================")
    print(" ---------------------------------------------------------------------------------------------------")

    data =  "💽 DATA"
    task =  "🎮 TASK"
    model = "👾 MODEL"
    version =  "🔖 V"
    best_performance = "🔥 BEST PERFORMANCE"
    print(f"   | {data:9s} | {task:15s} | {model:25s} | {version:3s} | {best_performance:9s}")
    print("   -------------- ----------------- ---------------------------- ------- ---------------------------")

    for data in os.listdir(hub_path):
        d_path = os.path.join(hub_path, data)
        for task in sorted(os.listdir(d_path)):
            d_t_path = os.path.join(d_path, task)
            for model in sorted(os.listdir(d_t_path)):  
                d_t_m_path = os.path.join(d_t_path, model)
                for version in sorted(os.listdir(d_t_m_path)):
                    model, flags = load('hub', data, task, model, version, map_location='cpu')
                    print(f"   | {data:10s} | {task:16s} | {model.__class__.__name__:26s} | {version:4s} | {flags.best_performance: .3f}")
                    del model 
        print("   ----------------------------------------------------------------------------")
        
    print(" ========================================================================================")
