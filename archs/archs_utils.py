

def getModel(args):
    if args.dataset == "mnist":
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar10":
        from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet

    elif args.dataset == "fashionmnist":
        from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet

    elif args.dataset == "cifar100":
        from archs.cifar100 import AlexNet, fc1, LeNet5, vgg, resnet

    if args.arch_type == "fc1":
        model = fc1.fc1().to(args.device)
    elif args.arch_type == "lenet5":
        model = LeNet5.LeNet5().to(args.device)
    elif args.arch_type == "alexnet":
        model = AlexNet.AlexNet().to(args.device)
    elif args.arch_type == "vgg16":
        model = vgg.vgg16().to(args.device)
    elif args.arch_type == "resnet18":
        model = resnet.resnet18().to(args.device)
    elif args.arch_type == "densenet121":
        model = densenet.densenet121().to(args.device)
    else:
        print("\nWrong Model choice\n")
        exit()

    return model