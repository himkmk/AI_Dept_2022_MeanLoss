import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
import argparse


class MeanLoss(nn.Module):
    
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.get_mean = lambda x: x.mean(dim=(-1,-2))
    
    def forward(self, generated, gt):
        """
        :param generated: (torch.tensor) shape of b,c,h,w
        :param gt: (torch.tensor) shape of b,c,h,w
        :return: (torch.tensor) loss
        """
    
        mean_gen = self.get_mean(generated)
        mean_gt = self.get_mean(gt)
    
        return F.mse_loss(mean_gen, mean_gt)
     
if __name__=="__main__":
    
    args = argparse.ArgumentParser()
    args.add_argument("--img1")
    args.add_argument("--img2")
    args = args.parse_args()
    
    
    mean_loss_module = MeanLoss()
    img1 = Image.open(args.img1)
    img2 = Image.open(args.img2)
    
    img1 = to_tensor(img1)
    img2 = to_tensor(img2)
    
    loss = mean_loss_module(img1, img2)
    
    print(loss)
    
    