## This file launch attacks via iteratively adding one piexl perturbation on each pixel.
import argparse
import numpy as np

import lib.adversary as adversary
from utils import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import os
import torchvision.transforms as transforms
from database import database
import warnings
import sys
import datetime
import torchattacks
from SelectModel import SelectModel_Linf
from torch.optim import lr_scheduler
from models import get_model

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=400, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--save_dir', default='save_temp/checkpoint_108.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--gpuid', type=str, default='1')
parser.add_argument('--shuffle', type=bool, default=False)
parser.add_argument('--net_type', type=str, help='vgg19 | resnet34 | inception_v3 | densenet121'
                                                 'wide_resnet101_2 | squeezenet1_0 | googlenet | mobilenet_v2')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--adv_type', required=False, help='FGSM | BIM | DeepFool | CWL2 | CW_inf')
parser.add_argument('--data_type', required=False, help='cifar10 | imagenet')
parser.add_argument('--data_path', required=False,default='Imagenet_2000/examples.txt', help='cifar10 | Imagenet')
parser.add_argument('--image_size', required=False, default=(3,244,244))
parser.add_argument('--adv_para', required=False,default=0.001)
parser.add_argument('--magnitude', required=False,default=1.0)
parser.add_argument('--N_class', required=False,default=1000)
parser.add_argument('--Lambda_2', required=False,default=1.0)
parser.add_argument('--Lambda_0', required=False,default=1.0)
parser.add_argument('--Lambda_loss', required=False,default=1.0)
parser.add_argument('--mu_a', required=False,default=-0.0)
parser.add_argument('--mu_b', required=False,default=-4.0)
parser.add_argument('--prior', required=False,default=True)
parser.add_argument('--loss', type=str,default='betterCW')
parser.add_argument('--Tag', type=int,default=0)
parser.add_argument('--nowtime', type=int,default=0)
parser.add_argument('--name_space', type=str,default='')

parser.set_defaults(net_type='mobilenet_v2')
parser.set_defaults(mu_b=-0.0)
parser.set_defaults(mu_a=-0.0)
# parser.set_defaults(data_type='imagenet')
parser.set_defaults(data_type='cifar10')
parser.set_defaults(tensorboard=False)
parser.set_defaults(ImageSize=244)
parser.set_defaults(lr=0.05)
parser.set_defaults(gpuid='0')
# parser.set_defaults(data_path='Imagenet_2000/examples.txt')
parser.set_defaults(data_path='Imagenet_2000/Imagenet_2000.txt')
parser.set_defaults(name_space='ExperimentRecordOld/Imagenet/Vgg19/DiscreteAdv_(1, 3.4, 0.33, 28.81, 520)_log.txt')
# parser.set_defaults(data_path='Imagenet_2000/failed_resnet50.txt')
parser.set_defaults(Lambda_2=2)
parser.set_defaults(Lambda_0=1)
parser.set_defaults(Lambda_loss=1000)
parser.set_defaults(loss='CW')
parser.set_defaults(adv_type='DiscreteAdv')
parser.set_defaults(adv_para=2)
parser.set_defaults(magnitude=1.0/255.0)

best_prec1 = 100
writer = None
time_acc = [(0, 0, 0)]
total_steps = 0
exp_flops, exp_l0 = [], []
maxiter = 20


def main():
    global args, best_prec1, writer, time_acc, total_steps, exp_flops, exp_l0, maxiter, Exp_log
    args = parser.parse_args()
    # for magnitude in [1/255.0,2/255.0,3/255.0,4/255.0,5/255.0,6/255.0,7/255.0,8/255.0,9/255.0,10/255.0,11/255.0]:
    for net_type in ['mobilenet_v2']:
    # for net_type in ['resnet50']:
        args.net_type = net_type
        main3(args)

def main3(args):
    global Exp_log
    now_time = datetime.datetime.now()
    args.now_time = '{}_{}_{}-{}({}:{}:{})_log.txt'.format(args.adv_type,args.net_type,now_time.month,
                            now_time.day,now_time.hour,now_time.minute,now_time.second)
    args.save_dir = 'ExperimnetRecord/{}'.format(os.path.basename(sys.argv[0]).split('.py')[0])
    os.makedirs(args.save_dir,exist_ok=True)
    Exp_log_path = os.path.join(args.save_dir,args.now_time)
    open(Exp_log_path,'w').close()
    Exp_log = open(Exp_log_path,'a')
    for name in vars(args):
        log = "{}={}".format(name,getattr(args,name))
        print(log)
        Exp_log.write('{}'.format(log))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
    main2(args, Exp_log)

def main2(args,Exp_log):
    script_name = os.path.basename(sys.argv[0]).split('.py')[0]
    ## Load Imagenet
    args.save_dir = 'ExperimnetRecord/{}/Images'.format(script_name)
    os.makedirs(args.save_dir, exist_ok=True)
    TrainDataloader,ValDataloader,ImageSize,N_class = database(args)
    args.ImageSize = ImageSize
    args.N_class = N_class
    ## Load Model
    model = get_model(args)
    model.cuda()
    model.eval()
    # atk = torchattacks.FGSM(model, eps=args.magnitude)
    count = 0
    succ_count = 0
    magnitude = get_magnitude()
    adv_list = []
    data_list = []
    avgl2 = 0
    for N_count,(image,label) in enumerate(ValDataloader):
        N_count+=1
        if N_count>1000:
            break
        if args.data_type == 'imagenet':
            image_path = ValDataloader.dataset.current_path
        elif args.data_type == 'cifar10':
            image_path = N_count
        data, target = image.cuda(), label.cuda()
        data, target = Variable(data, requires_grad=True), Variable(target)
        model.zero_grad()
        output = model(data)
        if not torch.argmax(output) == target:
            continue
        else:
            count += 1
            Flag = 0
            args.magnitude = 1.0 / 255
            while 1-Flag or args.magnitude > 20.0/255:
                adv_data,Flag = Attack(args, model, data, target)
                args.magnitude += 1.0/255

            # adv_noise = adv_data.cpu() - data.cpu()
            # adv_data2 = quantization(data,adv_noise,args.magnitude)
            magnitude.update(adv_data.cpu(),data.cpu())
            if Flag:
                l2 = magnitude.L_2_T
                avgl2 += l2
                adv_list.append(adv_data.detach().cpu().numpy().squeeze())
                data_list.append(data.detach().cpu().numpy().squeeze())
                succ_count += 1
                if args.data_type == 'imagenet':
                    save_path = save_image(adv_data, data, image_path,args)
            else:
                Exp_log.write('\n Fail !: {}'.format(image_path))
                print(image_path)
                Exp_log.flush()

            ASR = (succ_count / count)*100
            if count == 1:
                mode = '\n'
            else:
                mode = '\n'

            q_error = torch.mean(torch.abs(torch.round(adv_data*255.0)-adv_data*255.0))
            log = mode + 'succ_atk/succ_pred/n_sample: [{}]/[{}]/[{}] Net_type: {}, Adv_type: {}, ASR ({:.2f})! Magnitude (L_inf,L_2,L_1) [{L_inf:.4f},{L_2:.4f},{L_1:.4f}]) ### Q_ERROR {q_error}'.format(
                str(succ_count),str(count),str(N_count),args.net_type,args.adv_type,ASR,L_inf = magnitude.L_inf_T, L_2 = magnitude.L_2_T,
                L_1 = magnitude.L_1_T,q_error=q_error)
            sys.stdout.write(log)
            Exp_log.write(log)
        torch.cuda.empty_cache()

    save_dir = './ExperimnetRecord/{}_AEsArray_{}_{}_{}'.format(args.adv_type, args.net_type, args.data_type,args.nowtime)
    os.makedirs(save_dir,exist_ok=True)
    np.save(os.path.join(save_dir, 'NE.npy'), np.array(data_list))
    np.save(os.path.join(save_dir, 'AE.npy'), np.array(adv_list))
    l2 = magnitude.L_2_T

    log = mode + 'Net_type: {}, Adv_type: {}, ASR ({:.2f})! Magnitude (L_inf,L_2,L_1) [{L_inf:.4f},{L_2:.4f}(avgl2:.4f),{L_1:.4f}])'.format(
        args.net_type, args.adv_type, ASR, L_inf=magnitude.L_inf,L_2=magnitude.L_2,L_1=magnitude.L_1,avgl2=avgl2)
    print(log)

def save_image(adv_data,data,path,args):
    name = path.split('/')[-1].split('.JPEG')[0]
    save_dir = './ExperimnetRecord/AEs_{}_{}'.format(args.adv_type,args.net_type)
    os.makedirs(save_dir,exist_ok=True)
    adv_save_name =  os.path.join(save_dir,name+'_AE.png')
    noise_save_name =  os.path.join(save_dir,name+'_noise.png')
    adv_image = adv_data.cpu().detach().squeeze()
    image = data.cpu().detach().squeeze()
    noise = adv_image-image
    adv_PIL_image = transforms.ToPILImage()(adv_image)
    noise_image = transforms.ToPILImage()(noise)
    adv_PIL_image.save(adv_save_name,'PNG')
    noise_image.save(noise_save_name,'PNG')
    return adv_save_name

def quantization(data,adv_noise,manigude):
    data = data.cpu()
    adv_noise = adv_noise.cpu()
    c_data = data-adv_noise
    adv_noise = torch.clamp(adv_noise, -manigude, manigude)
    adv_noise = torch.round(adv_noise*255.0)
    adv_noise = adv_noise/255.0
    adv_data = c_data + adv_noise
    adv_data = torch.round(adv_data*255.0)
    adv_data = torch.clamp(adv_data,0.0,255.0)
    adv_data = adv_data/255.0
    return adv_data

def quantization_image(data):
    Trans = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    I_Trans = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    data = I_Trans(data.squeeze())
    data = torch.round(data*255.0)
    data = torch.clamp(data,0.0,255.0)
    data = data/255.0
    data = Trans(data).unsqueeze(dim=0)
    return data

def get_MaxMinPixel(net_type):
    import torchvision.transforms as transforms
    if net_type == 'vgg19':
        max_pixel = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])(torch.ones(3,1,1))
        min_pixel = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(torch.ones(3, 1, 1) * 0.0)
    return min_pixel,max_pixel

def Attack(args,model,clean,target):

    for para in model.parameters():
        para.requires_grad = False

    adv, Flag = DiscreteAdv_Linf(clean,target,model,args,targeted=-1)

    return adv, Flag

class get_magnitude():
    def __init__(self):
        self.L_inf = 0.0
        self.L_2 = 0.0
        self.L_1 = 0.0
        self.count = 0
        self.L_inf_T = 0.0
        self.L_2_T = 0.0
        self.L_1_T = 0.0

    def update(self,adv_data,data):
        import numpy as np
        self.count += 1
        adv_data = adv_data.detach().cpu().numpy()
        data = data.detach().cpu().numpy()
        adv_noise = (adv_data-data).squeeze()
        adv_noise = adv_noise
        self.L_inf_T = np.max(adv_noise*255.0)
        self.L_inf_T = float(self.L_inf_T)
        self.L_2_T  = np.min(np.sum((adv_data - data) ** 2, axis=(1, 2, 3)) ** .5)
        self.L_2_T  = float(self.L_2_T)
        self.L_1_T = np.min(np.sum(np.abs(adv_data - data), axis=(1, 2, 3)) )
        self.L_1_T = float(self.L_1_T)
        self.L_inf += self.L_inf_T
        self.L_2  += self.L_2_T
        self.L_1 += self.L_1_T

def DiscreteAdv_Linf(img,label,model,args,targeted=-1):
    SelectNet = SelectModel_Linf(model, ImageScale=args.ImageSize,maxiter=maxiter,magnitude=args.magnitude)
    SelectNet.Selectlayer.reset_parameters(args.mu_a,args.mu_b)
    SelectNet.cuda()
    optimizer = torch.optim.SGD(SelectNet.parameters(), args.lr)
    lr_schedule = lr_scheduler.MultiStepLR(optimizer, milestones=range(0, args.epochs,10), gamma=1.1)
    total_loss = 0
    total_reg_Q = 0
    total_reg_Z = 0
    magnitude = get_magnitude()
    Lambdas_0 = args.Lambda_0*torch.ones_like(SelectNet.Selectlayer.qz_loga)
    best_l2 = 1e+10
    best_adv_data = img.squeeze().cuda()
    for epoch in range(args.start_epoch, args.epochs):
        imgs = img.repeat([maxiter,1,1,1])
        input_var = torch.zeros(imgs.shape)
        target_var = label.repeat(maxiter)
        for i in range(maxiter):
            input_var[i,:] = (imgs[i,:]).squeeze()
            target_var[i] = label
        input_var = torch.autograd.Variable(input_var).cuda()
        target_var = torch.autograd.Variable(target_var).cuda()
        T_loss = 0
        output,adv_input = SelectNet(input_var,Selection=True)
        if args.loss == 'CW':
            loss = CW_loss(output, target_var, targeted=targeted)
        elif args.loss == 'CE':
            loss = targeted * nn.CrossEntropyLoss()(output, target_var)
        elif args.loss == 'betterCW':
            loss = softmax_cross_entropy_better(output,target_var)
        # loss = CW_loss(output,target_var,targeted=targeted)
        reg_Q = SelectNet.Selectlayer.L2_Q()
        reg_Z = SelectNet.Selectlayer.L0_reg(Lambdas_0)
        T_loss += args.Lambda_loss*loss-args.Lambda_2*reg_Q+reg_Z
        torch.cuda.empty_cache()
        with torch.no_grad():
            # adv_input = SelectNet.Selectlayer(input_var)
            # adv_noise = adv_input - imgs
            # adv_data = quantization(adv_input,adv_noise,args.magnitude)
            adv_data = torch.round(adv_input*255.0)/255.0
            adv_output = model(adv_data)

        prec1 = torch.sum(torch.argmax(adv_output,dim=1) == target_var)/maxiter

        total_loss += loss.data
        total_reg_Q += reg_Q.data
        total_reg_Z += reg_Z.data
        SelectNet.Selectlayer.constrain_parameters()

        if epoch == 0:
            mode = '\n'
        else:
            mode = '\r'
        magnitude.update(adv_data.cpu() , imgs.cpu())

        log = mode + 'Epoch: [{0}] Loss {loss:.4f} ({Tloss:.4f}) Flag {top1:.3f} L_inf {linf:.4f} L_2 {l2:.4f} L_0 {l1:.4f}'.format(
                    epoch,loss=float(loss.data.cpu().numpy()),Tloss=float(total_loss.data/(epoch+1)),linf=magnitude.L_inf_T, l2=magnitude.L_2_T,l1=magnitude.L_1_T,
                    top1=prec1)
        import sys
        sys.stdout.write(log)
        # print(log)
        if epoch%20 == 0:
            Exp_log.write(log)
            Exp_log.flush()
        adv_succ_index = torch.argmax(adv_output,dim=1) != target_var

        if torch.sum(adv_succ_index):
            Flag = 1
        else:
            Flag = 0

        for index,flag in enumerate(adv_succ_index):
            if flag:
                adv_noise_succ = (adv_data[index]-imgs.cpu())*255.0
                adv_noise_l2 = torch.sum(torch.pow(adv_noise_succ, 2)) / adv_noise_succ.view(-1, 1).shape[0]
                if adv_noise_l2<best_l2:
                    best_adv_data = adv_data[index]
                    best_l2 = adv_noise_l2
        if  best_l2 < 10000000:
            return best_adv_data.unsqueeze(dim=0), Flag

        optimizer.zero_grad()
        T_loss.backward()
        grads = SelectNet.Selectlayer.qz_loga.grad.data.cpu().unsqueeze(dim=0)
        Lambdas_0 = adjust_lambda(grads,args.Lambda_0)
        optimizer.step()

        lr_schedule.step()
        torch.cuda.empty_cache()
    return best_adv_data.unsqueeze(dim=0), Flag

def CW_loss(outputs,labels,targeted=-1):
    kappa = 0
    one_hot_labels = torch.zeros(maxiter,len(outputs[0])).cuda()
    one_hot_labels[:,labels[0]] = 1.0
    # one_hot_labels = torch.eye(len(outputs[0]))[labels[0]].cuda()
    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())
    return torch.mean(torch.clamp(targeted*(i-j), min=kappa))

def softmax_cross_entropy_better(logits, y_hot):
    import torch.nn.functional as F
    y_hot = F.one_hot(y_hot, num_classes=logits.shape[1])
    tmp = y_hot * logits
    logits_1 = logits - tmp
    j_best = torch.max(logits_1,axis=1)[0]
    j_best_v = torch.reshape(j_best, (j_best.shape[0], 1)).repeat((1,y_hot.shape[1]))
    logits_2 = logits_1 - j_best_v + y_hot*j_best_v
    tmp_s = torch.max(tmp, axis=1)[0]
    up = tmp_s - j_best
    down = torch.log(torch.sum(torch.exp(logits_2)+1,axis=1))
    loss = up - down
    return torch.mean(loss)

def adjust_lambda(grads,Lambda_0):
    Lambdas_0 = Lambda_0*torch.sigmoid(grads/1e-4)
    return Lambdas_0

if __name__ == '__main__':
    main()
