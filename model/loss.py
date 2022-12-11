import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)



def TripletMarginloss(anchor, positive, negative):
    loss_function = nn.TripletMarginLoss(margin = 0.2, p =2)
    return loss_function(anchor, positive, negative)
