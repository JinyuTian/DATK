from Imagenet import Imagenet
import torch
import torchvision
import torchvision.transforms as transforms
# from sketches import Sketches

def database(args):
    # ValDataSourceFile = 'ImageInfo/ValData_2000.txt'
    ValDataSourceFile = args.data_path
    # ValDataSourceFile = 'Imagenet_2000/Imagenet_2000.txt'
    if args.data_type == 'imagenet':
        ImageSize = 244
        N_class = 1000
        trainloader, testloader, TrainDataSourceFile, ValDataSourceFile = \
            Imagenet([ImageSize, ImageSize], args.batch_size, shuffle=args.shuffle, NEC=10, NC=10, NV=2000,
                     TrainDataSourceFile=ValDataSourceFile, ValDataSourceFile=ValDataSourceFile)
        print('Training source: {}'.format(TrainDataSourceFile))
        print('Testing source: {}'.format(ValDataSourceFile))
    elif args.data_type == 'cifar10':
        ImageSize = 32
        N_class = 10
        transform = transforms.Compose(
            [transforms.ToTensor()])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                  shuffle=True)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                 shuffle=args.shuffle)
    # elif args.data_type == 'sketches':
    #     ImageSize = 244
    #     N_class = 10
    #     TrainDataSourceFile = '/home/jinyu/DATK/ImageInfo/Sketch_TrainData_0.8_10.txt'
    #     ValDataSourceFile = '/home/jinyu/DATK/ImageInfo/Sketch_TestData_0.8_10.txt'
    #     trainloader, testloader, TrainDataSourceFile, ValDataSourceFile = \
    #         Sketches([ImageSize, ImageSize], 1, shuffle=False, ratio=0.8, NC=10, NV=2000,
    #                  TrainDataSourceFile=TrainDataSourceFile, ValDataSourceFile=ValDataSourceFile)
    return trainloader,testloader,ImageSize,N_class