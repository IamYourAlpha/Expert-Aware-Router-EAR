from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import router_dataloader
import models.cifar as models
import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as data
import copy
from ear_model import EAR
from torchvision import datasets
from torch.autograd import Variable
import os
best_correct_test = 0.0

# Use threshold to define predicted labels and invoke sklearn's metrics with different averaging strategies.
def calculate_metrics(pred, target, threshold=0.7):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro')}
            #'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            #'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            #'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro'),
            #'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            #'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            #'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'),
            #}

        

def test(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    now_correct = 0.0
    train_result = []
    ground_truth = []
    global best_correct_test
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_result.extend(output.detach().cpu().numpy())
            ground_truth.extend(target.detach().cpu().numpy())
    result = calculate_metrics(np.array(train_result), np.array(ground_truth))
    
    for metric in result:
        print ('{} --> {}'.format(metric, result[metric]))
    print (best_correct_test)
    if (result['micro/f1'] > best_correct_test):
        best_correct_test = result['micro/f1']
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, './router_wts/mlp.pth.tar')
        print ("Found New best, saving weights !!")

def test_(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for dta, target in test_loader:
        dta, target = dta.cuda(), target.cuda()
        dta, target = Variable(dta, volatile=True), Variable(target)
        output = model(dta)
        output = F.softmax(output, dim=1)
        test_loss += F.cross_entropy(output, target).item() # sum up batch loss
        pred = torch.argsort(output, dim=1, descending=True)[0:, 0]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
       
    test_loss /= len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
         100. * correct.double() / len(test_loader.dataset) ))
    

    
def test_whole_sytstem(whole_model, device, test_loader_):
    whole_model.eval()
    test_loss = 0.0
    correct = 0
    now_correct = 0.0
    count_ = 0
    train_result = []
    ground_truth = []
    count = 0
    global best_correct_test
    with torch.no_grad():
        for data, target in test_loader_:
            data, target = data.to(device), target.to(device)
            output = whole_model(data)
            output = F.softmax(output, dim=1)
            exp_conf, exp_pred = torch.sort(output, dim=1, descending=True)
            pred, conf_ = exp_pred[0:, 0], exp_conf[0:, 0]
            if (pred.cpu().numpy()[0] == target.cpu().numpy()[0]):
                correct += 1
            # if (count == 500):
            #    break
            # count = count + 1
    print (correct)


def get_list_of_confuising_classes(confusing_classes_dir):
    final_list = []
    for confusing_pair in confusing_classes_dir:
        name_ = confusing_pair[:-8]
        temp_list = name_.split('_')
        for i, num_ in enumerate(temp_list):
            temp_list[i] = int(temp_list[i])
        final_list.append(temp_list)
    return final_list

    

def main():
    
    # Set all params.
    torch.manual_seed(1)
    device = torch.device("cuda")
    confusing_classes_dir = os.listdir('F:/Research/PHD_AIZU/tiny_ai/ear/checkpoint_experts/ear_wts/r-110-subset/')
    confusing_classes = get_list_of_confuising_classes(confusing_classes_dir)
    #confusing_classes = [[35, 98], [55, 72], [47, 52], [11, 35], [11, 46], [70, 92], [13, 81], [47, 96], [2, 35], [81, 90], [52, 59], [62, 92], [78, 99], [5, 25], [30, 95], [50, 74], [30, 73], [10, 61], [33, 96], [44, 78], [67, 73], [23, 71], [46, 98], [52, 96], [2, 11], [35, 46], [13, 58], [18, 44], [26, 45], [4, 55]]
    dataloc = "F:/Research/PHD_AIZU/tiny_ai/ear/data/c100_combined"
    train_data = "train"
    test_data = "test"
    no_of_labels = len(confusing_classes)
    print (no_of_labels)
    batch_size = 32


    # STEP #1: Prepare the dataset.
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    #########  FOR THE ROUTER
    
    train_set  = router_dataloader.RouterDataLoader(dataloc, train_data, confusing_classes, transform_train)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    
    test_set  = router_dataloader.RouterDataLoader(dataloc, test_data, confusing_classes, transform_test)
    test_loader = DataLoader(test_set, batch_size, shuffle=True)
    
    ########################
    
    loc_ = 'F:/Research/PHD_AIZU/tiny_ai/ear/data/c100/test'
    test_set_ = datasets.ImageFolder(loc_, transform =transform_test)
    test_loader_ = torch.utils.data.DataLoader(test_set_, batch_size=1, shuffle=False)
    print ("Data loader completed")
  
    whole_model = EAR(confusing_classes)
    print (confusing_classes)
    whole_model.cuda()
    # Step # 3: Define the scheduler and optimizer.
    test_whole_sytstem(whole_model, device, test_loader_)
       

if __name__ == '__main__':
    main()