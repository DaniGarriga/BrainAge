import importlib
import torch.optim as optim
import torch
from model.sfcn import SFCN

def create_model(args):

    # ESTO SE PUEDE QUITAR?
    #lr = args.lr
    in_channels = args.inChannels
    num_classes = args.classes
    weight_decay = 0.0000000001

    print("Building Model . . . . . . . .")

    # model = VNetBased(in_channels=in_channels, elu=False, classes=num_classes)
    model = SFCN()
    
    print('Number of params: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    
    return model


def create_optimizer(args, model):
    
    optimizer_name = args.optimizer

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif optimizer_name == 'adam':
        #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas= args.betas )
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    return optimizer


# VER COMO AFECTA EL LR, QUIZAS NO HACE FALTA?
def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    #class_name = lr_config.pop('name')
    class_name = lr_config.lr_scheduler
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config.optimizer = optimizer
    scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50])
    #lambda1 = lambda epoch: 0.3 ** epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.3)
    #return clazz(**vars(lr_config))
    return scheduler