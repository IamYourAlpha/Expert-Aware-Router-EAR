import torch
import models.cifar as models
import torch.optim as optim
from ms_net_utils import *
from data_utils import *


def load_teacher_network():
    """ return the best teacher network with state_dict. """

    teacher = models.__dict__['resnext'](
                    cardinality=8,
                    num_classes=10,
                    depth=29,
                    widen_factor=4,
                    dropRate=0,
                )
    teacher = torch.nn.DataParallel(teacher).cuda()
    checkpoint = torch.load("./ck_backup/teachers/resnext_best.pth.tar")
    teacher.load_state_dict(checkpoint['state_dict'])
    return teacher


def load_expert_networks_and_optimizers(lois, 
                                        num_classes, 
                                        dataset, 
                                        arch, 
                                        depth, 
                                        block_name,
                                        initialize_with_router=True,
                                        finetune_experts=True
                                        ):
    experts = {}
    eoptimizers = {}
    chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
    for loi in lois:
        experts[loi] = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    block_name=block_name)
        
        experts[loi] = experts[loi].cuda()
        
        initialize_with_router = True
        if (initialize_with_router):
            experts[loi].load_state_dict(chk['state_dict'])

        finetune_experts = True
        if (finetune_experts):
            eoptimizers[loi] = optim.SGD([{'params': experts[loi].layer1.parameters(), 'lr': 0.0},
                                        {'params': experts[loi].layer2.parameters(), 'lr': 0.0},
                                         {'params': experts[loi].layer3.parameters(), 'lr': 0.01},
                                         {'params': experts[loi].fc.parameters()}],
                                         lr=0.01, momentum=0.9, weight_decay=5e-4)
            
        else:
            eoptimizers[loi] = optim.SGD(experts[loi].parameters(), lr=0.1, momentum=0.9,
                      weight_decay=5e-4)
            
        
    return experts, eoptimizers



def make_router_and_optimizer(num_classes,
                              dataset,
                              arch,
                              depth,
                              block_name,
                              learning_rate,
                              load_weights=False):
    model = models.__dict__[arch](
                    num_classes=num_classes,
                    depth=depth,
                    block_name=block_name)
    if (load_weights):
        #model = torch.nn.DataParallel(model).cuda()
        model = model.cuda()
        print ('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
        chk = torch.load('./ck_backup/%s/%s-depth-%s/checkpoint/model_best.pth.tar'%(dataset, arch, depth))
        model.load_state_dict(chk['state_dict'])
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,
                      weight_decay=5e-4)
    return model, optimizer
