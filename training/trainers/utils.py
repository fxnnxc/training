import torch 

def make_optimizer(parameters, optim_type, lr, **kwargs):
    if optim_type == "sgd":
        optimizer = torch.optim.SGD(parameters,
                                    lr=lr, 
                                    momentum=kwargs.get("momentum", 0.9), 
                                    weight_decay=kwargs.get("weight_decay", 5e-4))
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(parameters, 
                                     lr=lr ,
                                     **kwargs)
    elif optim_type == "adamw":
        optimizer = torch.optim.AdamW(parameters, 
                                      lr=lr,
                                      **kwargs
                                     )
    else:
        raise ValueError(f"{optim_type} is undefind optimizer")
    
    return optimizer 

def make_lr_scheduler(optimizer, scheduler_type, **kwargs):
    if scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs.get("epochs"))
    elif scheduler_type == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get("step_size"), gamma=kwargs.get("gamma"))
    else:
        lr_scheduler = None
    return lr_scheduler