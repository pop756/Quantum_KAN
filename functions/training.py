import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import copy
import pandas as pd
import pickle


def accuracy(pred, true):
    # 예측값이 로짓 혹은 확률값인 경우, 최대 값을 가진 인덱스를 구함 (가장 확률이 높은 클래스)
    pred = pred.detach().cpu()
    true = true.cpu()
    pred_labels = torch.round(pred)
    # 예측 레이블과 실제 레이블이 일치하는 경우를 계산
    correct = (pred_labels == true).sum()
    # 정확도를 계산
    acc = correct / true.size(0)
    return acc.item()

class Early_stop_train():
    def __init__(self,model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion



        self.loss_list = [1e100]
        self.stop_count = 0

    def train_model(self,train_loader,test_loader=None ,epochs=200,res = 10,device = 'cpu'):
        self.model.train()
        for epoch in range(epochs):
            if self.stop_count>=res:
                break
            loss_val,_ = self.test(test_loader,device)
            self.loss_list.append(loss_val)

            if self.loss_list[-1]>=np.min(self.loss_list[:-1]):
                self.stop_count+=1
            else:
                self.stop_count = 0
            loss_list = []
            acc_list = []
            for X_train,y_train in train_loader:
                if isinstance(X_train,list):
                    for i,X in enumerate(X_train):
                        X_train[i] = X.to(device)
                else:
                    X_train = X_train.to(device)
                y_train = y_train.to(device)
                self.optimizer.zero_grad()
                output = self.model(X_train)

                loss = self.criterion(output.squeeze(), y_train)

                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
                acc = accuracy(output,y_train)
                acc_list.append(acc)

                sys.stdout.write(f"\rEpoch {epoch+1} Loss {np.mean(loss_list):4f} acc : {np.mean(acc_list):4f} stop count : {self.stop_count}")


    def test(self,test_loader,device='cpu'):
        if test_loader is None:
            return 0,0
        else:
            #self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    if isinstance(data,list):
                        for i,X in enumerate(data):
                            data[i] = X.to(device)
                    else:
                        data = data.to(device)
                    target = target.to(device)
                    data, target = data, target
                    output = self.model(data)
                    test_loss += self.criterion(output.squeeze(), target).item()

                    correct += accuracy(output,target)*len(output)

            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
            return test_loss,correct
        
def reg(acts_scale,KAN_layer, factor=1,lamb_l1=1.,lamb_entropy=2.,lamb_coef=0.,lamb_coefdiff=0.):

    def nonlinear(x, th=1e-16):
        return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

    reg_ = 0.
    for i in range(len(acts_scale)):
        vec = acts_scale[i].reshape(-1, )

        p = vec / torch.sum(vec)
        l1 = torch.sum(nonlinear(vec))
        entropy = - torch.sum(p * torch.log2(p + 1e-4))
        reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

    # regularize coefficient to encourage spline to be zero
    for i in range(len(KAN_layer.act_fun)):
        coeff_l1 = torch.sum(torch.mean(torch.abs(KAN_layer.act_fun[i].coef), dim=1))
        coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(KAN_layer.act_fun[i].coef)), dim=1))
        reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

    return reg_

class Early_stop_train_KAN():
    def __init__(self,model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        

        
        self.loss_list = [1e100]
        self.acc_list = []
        self.stop_count = 0
        
    def train_model(self,train_loader,test_loader=None ,epochs=200,res = 10,lamb=0.,device='cpu'):
        #self.model.train()
        
        for epoch in range(epochs):
            if self.stop_count>=res:
                break
            loss_val,_ = self.test(test_loader,device)
            self.loss_list.append(loss_val)
            
            if self.loss_list[-1]>=np.min(self.loss_list[:-1]):
                self.stop_count+=1
            else:
                self.optimal = self.model.state_dict()
                self.stop_count = 0
            loss_list = []
            acc_list = []
            for X_train,y_train in train_loader:
                if isinstance(X_train,list):
                    for i,X in enumerate(X_train):
                        X_train[i] = X.to(device)
                else:
                    X_train = X_train.to(device)
                self.optimizer.zero_grad()
                output = self.model(X_train)
                reg_ = lamb*reg(self.model.KAN.acts_scale,self.model.KAN)
                try:
                    loss = self.criterion(output.squeeze(), y_train)+reg_
                except:
                    print(output)
                    raise
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())
                acc = accuracy(output,y_train)
                acc_list.append(acc)
                sys.stdout.write(f"\rEpoch {epoch+1} Loss {np.mean(loss_list):4f} acc : {np.mean(acc_list):4f} reg : {reg_:4f} stop count : {self.stop_count}")
        self.model.load_state_dict(self.optimal)
    def test(self,test_loader,device='cpu'):
        if test_loader is None:
            return 0,0
        else:
            #self.model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    for data, target in test_loader:
                        if isinstance(data,list):
                            for i,X in enumerate(data):
                                data[i] = X.to(device)
                        else:
                            data = data.to(device)
                    output = self.model(data)

                    test_loss += self.criterion(output.squeeze(), target).item()
                    
                    correct += accuracy(output,target)*len(output)

            print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)')
            return test_loss,correct