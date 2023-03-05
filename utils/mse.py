import torch.nn as nn

class RMSE(nn.CrossEntropyLoss):
    """
    Computes MSE between input and target
    """

    def __init__(self):
        super(RMSE, self).__init__()
        self.metric = nn.MSELoss()

    def __call__(self, input, target):
        #input, target = convert_to_numpy(input, target)
        val = torch.sqrt(self.metric(input, target))
        return val