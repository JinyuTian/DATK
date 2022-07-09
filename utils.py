import torch.nn as nn
import torch

def rescaling(x,MAX,MIN):
    ## rescaling [Min_x Max_x] into [MAX,MIN]
    x = x.view(-1,1)
    Max_x = torch.max(x)
    Min_x = torch.min(x)
    z = (x-Min_x)/(Max_x-Min_x)
    y = z*(MAX-MIN)+MIN
    return y


# x = torch.normal(torch.tensor(0.1),torch.tensor(0.3),[20])
# y = rescaling(x,1,-1)
# print(max(y))
# print(min(y))