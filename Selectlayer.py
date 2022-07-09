import torch
import math
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair as pair
from torch.autograd import Variable
from torch.nn import init
import numpy as np
import torchvision.transforms as transforms
limit_a, limit_b, epsilon = -.5, 1.5, 1e-6

class Selectlayer(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""
    def __init__(self, in_channels, out_channels,temperature=1.0/20.0,temperature2=1.0/20.0, lamba=1.,
                 local_rep=False,P=10,prior_noise=0,magnitude=1,prior_qzloga=0,maxiter=1,**kwargs):

        super(Selectlayer, self).__init__()
        self.P = P
        self.maxiter = maxiter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = pair(0)
        self.lamba = lamba
        self.z = 0
        self.z0 = 0
        self.temperature = temperature
        self.temperature2 = temperature2
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        # self.qz_loga = Parameter(prior_qzloga.view(-1,1).squeeze())
        self.qz_loga = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.dim_z = in_channels*in_channels*3
        self.dim_z = in_channels*in_channels*3
        self.input_shape = None
        self.local_rep = local_rep
        self.prior_noise = prior_noise
        self.magnitude = magnitude
        # self.reset_parameters(prior_qzloga)

    def reset_parameters(self,mu_a,mu_b):
        # self.qz_loga.copy_(prior_qzloga)
        self.qz_loga.data.normal_(0.0, 1e-2)
        self.qz_loga0.data.normal_(-4.0, 1e-2)

    def sample_z(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
        z = z.view(1,1,self.in_channels,self.in_channels)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    def reset_qz_loga(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 4)

    def L2_Q(self):
        return torch.linalg.norm(self.qz_loga)

    def L2_Z(self):
        return torch.linalg.norm(self.z)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga0).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, qz_loga,temperature):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + qz_loga) / temperature)
        return y * (limit_b - limit_a) + limit_a

    def L0_reg(self,Lambdas):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw = torch.sum(Lambdas.cuda()*(1 - q0))
        # logpw = torch.sum(1 - q0)
        return logpw


    def constraint(self):
        return torch.sum((1 - self.cdf_qz(0)))

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def initial_qz(self,x):
        self.qz_loga.data.normal_(x,0.0001)

    def add_noise(self,x):
        for i in range(self.maxiter):
            if i == 0:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z0 = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga0,self.temperature2)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z0 = F.hardtanh(z0, min_val=0, max_val=1)
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude/255)
                z0 = z0.view(x[0].shape).unsqueeze(dim=0)
                noise = z*z0
            else:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z0 = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga0,self.temperature2)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z0 = F.hardtanh(z0, min_val=0, max_val=1)
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude/255)
                z0 = z0.view(x[0].shape).unsqueeze(dim=0)
                noise = torch.cat([noise,z*z0],dim=0)








        self.z = noise
        self.z0 = z0
        # print(torch.sum(z*z0))
        # return x + z*z0
        # return x + z*z0
        return torch.clip(x+noise,0.0,1.0)


    # def SelelctPixel(self,maxiter):
    #
    #     for i in range(maxiter):
    #         if i == 0:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             Z = z.repeat(1, 3, 1, 1)
    #         else:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             z = z.repeat(1, 3, 1, 1)
    #             Z += z
    #
    #     Z = Z/maxiter
    #     return Z

    def forward(self, input_):
        return self.add_noise(input_)

class Selectlayer_L0(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""
    def __init__(self, in_channels, out_channels,temperature=1.0/20.0,temperature2=1.0/20.0, lamba=1.,
                 local_rep=False,P=10,prior_noise=0,magnitude=1,maxiter=1,**kwargs):

        super(Selectlayer_L0, self).__init__()
        self.P = P
        self.maxiter = maxiter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = pair(0)
        self.lamba = lamba
        self.z = 0
        self.z0 = 0
        self.temperature = temperature
        self.temperature2 = temperature2
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        # self.qz_loga = Parameter(prior_qzloga.view(-1,1).squeeze())
        self.qz_loga = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.dim_z = in_channels*in_channels*3
        self.dim_z = in_channels*in_channels*3
        self.input_shape = None
        self.local_rep = local_rep
        self.prior_noise = prior_noise
        self.magnitude = magnitude
        # self.reset_parameters(prior_qzloga)

    def reset_parameters(self,mu_a,mu_b):
        # self.qz_loga.copy_(prior_qzloga)
        self.qz_loga.data.normal_(mu_a, 1e-2)
        self.qz_loga0.data.normal_(mu_b, 1e-2)

    def sample_z(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
        z = z.view(1,1,self.in_channels,self.in_channels)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    def reset_qz_loga(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 4)

    def L2_Q(self):
        return torch.linalg.norm(self.qz_loga)

    def L2_Z(self):
        return torch.linalg.norm(self.z)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga0).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, qz_loga,temperature):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + qz_loga) / temperature)
        return y * (limit_b - limit_a) + limit_a

    def L0_reg(self,Lambdas):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw = torch.sum(Lambdas.cuda()*(1 - q0))
        # logpw = torch.sum(1 - q0)
        return logpw


    def constraint(self):
        return torch.sum((1 - self.cdf_qz(0)))

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def initial_qz(self,x):
        self.qz_loga.data.normal_(x,0.0001)

    def add_noise(self,x):
        for i in range(self.maxiter):
            if i == 0:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z0 = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga0,self.temperature2)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z0 = F.hardtanh(z0, min_val=0, max_val=1)
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude)
                z0 = z0.view(x[0].shape).unsqueeze(dim=0)
                noise = z*z0
            else:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z0 = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga0,self.temperature2)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z0 = F.hardtanh(z0, min_val=0, max_val=1)
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude)
                z0 = z0.view(x[0].shape).unsqueeze(dim=0)
                noise = torch.cat([noise,z*z0],dim=0)








        self.z = noise
        self.z0 = z0
        # print(torch.sum(z*z0))
        # return x + z*z0
        # return x + z*z0
        return torch.clip(x+noise,0.0,1.0)


    # def SelelctPixel(self,maxiter):
    #
    #     for i in range(maxiter):
    #         if i == 0:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             Z = z.repeat(1, 3, 1, 1)
    #         else:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             z = z.repeat(1, 3, 1, 1)
    #             Z += z
    #
    #     Z = Z/maxiter
    #     return Z

    def forward(self, input_):
        return self.add_noise(input_)

class Selectlayer_Linf(Module):
    """Implementation of L0 regularization for the feature maps of a convolutional layer"""
    def __init__(self, in_channels, out_channels,temperature=1.0/20.0,temperature2=1.0/20.0, lamba=1.,
                 local_rep=False,P=10,prior_noise=0,magnitude=1,prior_qzloga=0,maxiter=1,**kwargs):

        super(Selectlayer_Linf, self).__init__()
        self.P = P
        self.maxiter = maxiter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = pair(0)
        self.lamba = lamba
        self.z = 0
        self.z0 = 0
        self.temperature = temperature
        self.temperature2 = temperature2
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        # self.qz_loga = Parameter(prior_qzloga.view(-1,1).squeeze())
        self.qz_loga = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        self.qz_loga0 = Parameter(torch.Tensor(in_channels*in_channels*3))
        # self.dim_z = in_channels*in_channels*3
        self.dim_z = in_channels*in_channels*3
        self.input_shape = None
        self.local_rep = local_rep
        self.prior_noise = prior_noise
        self.magnitude = magnitude
        # self.reset_parameters(prior_qzloga)

    def reset_parameters(self,mu_a,mu_b):
        # self.qz_loga.copy_(prior_qzloga)
        self.qz_loga.data.normal_(mu_a, 1e-2)
        self.qz_loga0.data.normal_(-4.0, 1e-2)

    def sample_z(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
        z = z.view(1,1,self.in_channels,self.in_channels)
        z = F.hardtanh(z, min_val=0, max_val=1)
        return z

    def reset_qz_loga(self):
        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 4)

    def L2_Q(self):
        return torch.linalg.norm(self.qz_loga)

    def L2_Z(self):
        return torch.linalg.norm(self.z)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga0).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x, qz_loga,temperature):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + qz_loga) / temperature)
        return y * (limit_b - limit_a) + limit_a

    def L0_reg(self,Lambdas):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        q0 = self.cdf_qz(0)
        logpw = torch.sum(Lambdas.cuda()*(1 - q0))
        # logpw = torch.sum(1 - q0)
        return logpw


    def constraint(self):
        return torch.sum((1 - self.cdf_qz(0)))

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps = Variable(eps)
        return eps

    def initial_qz(self,x):
        self.qz_loga.data.normal_(x,0.0001)

    def add_noise(self,x):
        for i in range(self.maxiter):
            if i == 0:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude)
                noise = z
            else:
                z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z)), self.qz_loga,self.temperature)
                z = (F.hardtanh(z, min_val=0, max_val=1) - 0.5) * 2
                z = z.view(x[0].shape).unsqueeze(dim=0)*(self.magnitude)
                noise = torch.cat([noise,z],dim=0)

        self.z = noise
        return torch.clip(x+noise,0.0,1.0)


    # def SelelctPixel(self,maxiter):
    #
    #     for i in range(maxiter):
    #         if i == 0:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             Z = z.repeat(1, 3, 1, 1)
    #         else:
    #             z = self.quantile_concrete(self.get_eps(self.floatTensor(self.dim_z))).view(self.dim_z, 1, 1, 1)
    #             z = z.view(1, 1, self.in_channels, self.in_channels)
    #             z = F.hardtanh(z, min_val=0, max_val=1)
    #             z = z.repeat(1, 3, 1, 1)
    #             Z += z
    #
    #     Z = Z/maxiter
    #     return Z

    def forward(self, input_):
        return self.add_noise(input_)






