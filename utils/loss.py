import torch.nn as nn
from torch.nn import MSELoss, SmoothL1Loss, L1Loss

def my_KLDivLoss(x, y):
    """Returns K-L Divergence loss
    Different from the default PyTorch nn.KLDivLoss in that
    a) the result is averaged by the 0th dimension (Batch size)
    b) the y distribution is added with a small value (1e-16) to prevent log(0) problem
    """
    loss_func = nn.KLDivLoss(reduction='sum')
    y += 1e-16
    n = y.shape[0]
    loss = loss_func(x, y) / n
    #print(loss)
    return loss

def get_loss_criterion(name):
    '''
        Returns the loss function based on provided configuration
        :param name: str containing the name of the loss function
        :return: an instance of the loss function
        todo: add/modify if needed other loss functions
    '''
    if name == 'cross-entropy':
        return CrossEntropy()
    elif name == 'mse-loss':
        return MSELoss()
    elif name == 'mae':
        return L1Loss()
    
class CrossEntropy(nn.CrossEntropyLoss):
    #Example: loo = LOO()
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def split(self, X):
        t=2
        #return super(LOO, self).split(X)