from torch import nn
from torch.nn.functional import cross_entropy

class ELBO(nn.Module):
    def __init__(self, beta=1):
        super(ELBO, self).__init__()
        self.beta = beta
    
    def forward(self, inputs, targets, beta=None):
        if beta is None:
            beta = self.beta
        preds, kl = inputs
        nll = cross_entropy(preds, targets, ignore_index=-1)
        loss = nll + kl * beta
        # print(f"nll: {nll}, kl: {kl}")
        return loss
    
class BetaGenerator():
    def __init__(self, full_dataset=True, task_adaptive=False, zero_on_first_task=False) -> None:
        self.full_dataset = full_dataset
        self.task_adaptive = task_adaptive
        self.zero_on_first_task = zero_on_first_task
        
        self.size = None
        self.task = None
            
    def get(self):
        if self.size is None:
            raise ValueError("Beta has not been initialised.")
        if self.task_adaptive and self.task is None:
            raise ValueError("Beta has not been initialised.")

        
        if self.task_adaptive:
            if self.task == 0:
                if self.zero_on_first_task:
                    return 0.
                else:
                    return 1. / self.size
            else:
                return self.task / self.size
        else:
            return 1. / self.size
    
    def feed_full_dataset(self, size):
        if self.full_dataset:
            self.size = size
    
    def feed_batch(self, size):
        if not self.full_dataset:
            self.size = size
    
    def feed_task(self, task):
        self.task = task