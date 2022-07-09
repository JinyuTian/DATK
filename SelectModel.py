import torchvision.models as models
import torch.nn as nn
from Selectlayer import Selectlayer_Linf,Selectlayer,Selectlayer_L0

class SelectModel(nn.Module):
    def __init__(self, model,ImageScale=244,prior_noise=0,magnitude=1,prior_qzloga=0,maxiter=1):
        super(SelectModel, self).__init__()
        self.Selectlayer = Selectlayer(in_channels = ImageScale, out_channels = ImageScale,prior_noise=prior_noise
                                       ,magnitude=magnitude,prior_qzloga=prior_qzloga,maxiter=maxiter)
        self.model = model


    def forward(self, x, Selection):
        if Selection == True:
            x = self.Selectlayer(x)
            y = x.clone().detach().cpu()
            x = self.model(x)
        else:
            x = self.model(x)
            y = x.clone().detach().cpu()
        return x,y

    def attention_map(self,x,rho,maxiter):
        Ori_z = self.Selectlayer.SelelctPixel(maxiter)
        z = (Ori_z>rho).float()
        return z*x,z,Ori_z

    def Select(self,input_var,rho,gradient,level=0.02):
        x = self.Selectlayer.sample_weights_Throshold(input_var,rho,gradient,level=0.02)
        return x


class SelectModel_Linf(nn.Module):
    def __init__(self, model,ImageScale=244,prior_noise=0,magnitude=1,maxiter=1):
        super(SelectModel_Linf, self).__init__()
        self.Selectlayer = Selectlayer_Linf(in_channels = ImageScale, out_channels = ImageScale,prior_noise=prior_noise
                                       ,magnitude=magnitude,maxiter=maxiter)
        self.model = model


    def forward(self, x, Selection):
        if Selection == True:
            x = self.Selectlayer(x)
            y = x.clone().detach().cpu()
            x = self.model(x)
        else:
            x = self.model(x)
            y = x.clone().detach().cpu()
        return x,y

    def attention_map(self,x,rho,maxiter):
        Ori_z = self.Selectlayer.SelelctPixel(maxiter)
        z = (Ori_z>rho).float()
        return z*x,z,Ori_z

    def Select(self,input_var,rho,gradient,level=0.02):
        x = self.Selectlayer.sample_weights_Throshold(input_var,rho,gradient,level=0.02)
        return x


class SelectModel_L0(nn.Module):
    def __init__(self, model,ImageScale=244,prior_noise=0,magnitude=1,maxiter=1):
        super(SelectModel_L0, self).__init__()
        self.Selectlayer = Selectlayer_L0(in_channels = ImageScale, out_channels = ImageScale,prior_noise=prior_noise
                                       ,magnitude=magnitude,maxiter=maxiter)
        self.model = model


    def forward(self, x, Selection):
        if Selection == True:
            x = self.Selectlayer(x)
            y = x.clone().detach().cpu()
            x = self.model(x)
        else:
            x = self.model(x)
            y = x.clone().detach().cpu()
        return x,y

    def attention_map(self,x,rho,maxiter):
        Ori_z = self.Selectlayer.SelelctPixel(maxiter)
        z = (Ori_z>rho).float()
        return z*x,z,Ori_z

    def Select(self,input_var,rho,gradient,level=0.02):
        x = self.Selectlayer.sample_weights_Throshold(input_var,rho,gradient,level=0.02)
        return x
