#!/usr/bin/env python

from alpha_net import ChessNet, train
import os
import pickle
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_chessnet(net_to_train="current_net_trained8_iter1.pth.tar",save_as="current_net_trained1_iter1.pth.tar"):
    # gather data
    data_path = "./datasets/iter300/"
    datasets = []
    for idx, file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    
    data_path = "./datasets/iter2/"
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))
    
    datasets = np.array(datasets)
    
    # train net
    net = ChessNet().to(device)
    current_net_filename = os.path.join("./model_data/",net_to_train)
    checkpoint = torch.load(current_net_filename)
    net.load_state_dict(checkpoint['state_dict'])
    train(net,datasets)
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))

if __name__=="__main__":
    train_chessnet()