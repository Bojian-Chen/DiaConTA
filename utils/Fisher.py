import torch
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import torch.optim as optim

def Fisher(backbone, classifier, loader):

    backbone.train()
    classifier.eval()

    params = filter(lambda p: p.requires_grad, backbone.parameters())
    EWC_optimizer = optim.SGD(params, lr=0.001)
    fishers = {}
    for batch_idx, instance in enumerate(loader):
        inputs, labels = instance[0].cuda(), instance[1].cuda()
        features = backbone(inputs)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        loss = nn.CrossEntropyLoss()(outputs, predicted)
        loss.backward()
        for name, param in backbone.named_parameters():
            if param.grad is not None:
                if batch_idx > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if batch_idx == len(loader):
                    fisher = fisher / batch_idx
                fishers.update({name: [fisher, param.data.clone().detach()]})
        EWC_optimizer.zero_grad()

    backbone.eval()
    return fishers

def Fisher_BN(backbone, classifier, loader):

    backbone.train()
    classifier.eval()

    for m in backbone.modules():
        if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
        else:
            m.requires_grad_(False)

    params = filter(lambda p: p.requires_grad, backbone.parameters())
    EWC_optimizer = optim.SGD(params, lr=0.001)
    fishers = {}
    for batch_idx, instance in enumerate(loader):
        inputs, labels = instance[0].cuda(), instance[1].cuda()
        features = backbone(inputs)
        outputs = classifier(features)
        _, predicted = outputs.max(1)
        loss = nn.CrossEntropyLoss()(outputs, predicted)
        loss.backward()
        for name, param in backbone.named_parameters():
            if param.grad is not None:
                if batch_idx > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if batch_idx == len(loader):
                    fisher = fisher / batch_idx
                fishers.update({name: [fisher, param.data.clone().detach()]})
        EWC_optimizer.zero_grad()

    backbone.eval()
    return fishers