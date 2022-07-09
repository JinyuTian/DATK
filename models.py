import os
import torchvision.models as models
import torch
from My_VGG import my_vgg19_bn
from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn


all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}

class Fast_AT(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model = models.__dict__['resnet50']()
        self.model = torch.nn.DataParallel(self.model).cuda()
        self._mean_torch = torch.tensor((0.0, 0.0, 0.0)).view(3,1,1).cuda()
        self._std_torch = torch.tensor((1, 1, 1)).view(3,1,1).cuda()

    def forward(self, x):
        # x = x.transpose(1, 2).transpose(1, 3).contiguous()
        input_var = (x.cuda() - self._mean_torch) / self._std_torch
        labels = self.model(input_var)

        return labels.cpu()

    def load(self,MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

class Net(torch.nn.Module):
    def __init__(self,model,args):
        torch.nn.Module.__init__(self)
        self.model = model
        if args.data_type == 'imagenet':
            self.mean = torch.tensor((0.485, 0.456, 0.406)).view(3,1,1).cuda()
            self.std = torch.tensor((0.229, 0.224, 0.225)).view(3,1,1).cuda()
        elif args.data_type == 'cifar10':
            self.mean = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1).cuda()
            self.std = torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1).cuda()
        elif args.data_type == 'sketches':
            self.mean = torch.tensor((0.0)).view(1, 1, 1).cuda()
            self.std = torch.tensor((1.0)).view(1, 1, 1).cuda()

    def forward(self, x):
        input_var = (x.cuda() - self.mean) / self.std
        DEVICE = torch.device('cuda:{}'.format(0))
        input_var.to(DEVICE)
        labels = self.model(input_var)
        return labels

def get_model(args):
    if args.data_type == 'imagenet':
        try:
            model = eval('models.{}(pretrained=True)'.format(args.net_type))
        except:
            if args.net_type == 'Fast_AT':
                for file in os.listdir(args.checkpoint):
                    if args.model in file:
                        check_point_path = os.path.join(args.checkpoint,file)
                model = Fast_AT()
                model.load(check_point_path)
            else:
                print('model name should be ``fast_at`` or the same as the torchvision models')
    elif args.data_type == 'cifar10':
        model = all_classifiers[args.net_type]
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", args.net_type + ".pt"
        )
        model.load_state_dict(torch.load(state_dict))
    elif args.data_type == 'sketches':
        checkpoint_path = 'checkpoints/checkpoint_{}.tar'.format(args.net_type)
        # checkpoint_path = '/home/jinyu/DATK/checkpoint_0.793749988079071.tar'
        checkpoint = torch.load(checkpoint_path)
        if args.net_type == 'vgg19_bn':
            model = my_vgg19_bn()
            model.load_state_dict(checkpoint['state_dict'])

    model = Net(model,args)

    return model