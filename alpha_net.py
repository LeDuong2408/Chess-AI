#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import datetime
import numpy as np

class board_data(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        """
        Dùng để xử lý dữ liệu.
        Args:
            dataset (np.array): Array of tuples (s, p, v).
                s: state (12 × 8 × 8 tensor)
                p: policy
                v: value
        """
        if isinstance(dataset, np.ndarray):
            self.X = dataset[:, 0]
            self.y_p = dataset[:, 1]
            self.y_v = dataset[:, 2]
        else:
            self.X = [item[0] for item in dataset]
            self.y_p = [item[1] for item in dataset]
            self.y_v = [item[2] for item in dataset]
        # self.X = dataset[:, 0]  # Trạng thái đã ở dạng 12 × 8 × 8
        # self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]
    
    def __len__(self):
        return len(self.X)  # Lấy độ dài của dataset
    
    def __getitem__(self, idx):
        """
        Lấy 1 điểm data.
        Args:
            idx (int): Index của điểm data
        Returns:
            tuple: state, policy, value 
        """
        return self.X[idx], self.y_p[idx], self.y_v[idx]

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(12, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        s = F.relu(self.bn1(self.conv1(s)))
        return s

class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out
    
class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
        
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 4095) 
    
    def forward(self, s):
        # Value head
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.fc1(v))
        v = torch.tanh(self.fc2(v))
        
        # Policy head
        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.fc(p)
        p = F.softmax(p, dim=1)
        return p, v

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
    
    def forward(self, s):
        """
        ChessNet model.
        Args:
            s (torch.Tensor): Trạng thái.
        Returns:
            tuple: Policy, value được dự đoán.
        """
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        p, v = self.outblock(s)
        return p, v
    
    def _initialize_weights(self):
        """
        Khởi tạo trọng số mặc định cho các lớp trong mạng.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

class AlphaLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, eps=1e-8):
        super(AlphaLoss, self).__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, y_value, value, y_policy, policy):
        """
        Hàm loss để tối ưu mô hình
        Args:
            y_value (int): giá trị được dự đoán xác xuất chiến thắng 
            value (int): giá trị lấy từ MCTS (label) xác xuất chiến thắng 
            y_policy (array): chính sách được dự đoán (xác suất các hành động) 
            policy (array): chính sách tù MCTS (label) (xác suất các hành động)
        return (float): giá trị lỗi dựa trên giá trị và chính sách
        """
        value_error = abs(value - y_value)
        policy = policy.squeeze(-1)
        policy_error = torch.sum((-torch.abs(policy - y_policy)) * torch.log(torch.clamp(y_policy.float(), min=self.eps)), dim=1)
        total_error = (self.alpha * value_error.view(-1).float() + (1 - self.alpha) * policy_error).mean()
        return total_error
    

def train(net, dataset, epoch_stop=20, cpu=0):
    torch.manual_seed(cpu)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net.to(device).train()
    
    
    criterion = AlphaLoss() # Hàm loss và tối ưu
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2) # Lịch trình giảm dần hệ số learning rate
    
    train_set = board_data(dataset)
    train_loader = DataLoader(train_set, batch_size= 50, shuffle=True, num_workers=0, pin_memory=False)
    
    losses_per_epoch = [] # Để lưu lại giá trị lỗi từng epoch

    for epoch in range(epoch_stop):
        print(f"Start training... Epoch {epoch+1}")
        
        total_loss = 0.0
        losses_per_batch = []
        
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            if cuda:
                state, policy, value = state.cuda().float(), policy.float().cuda(), value.cuda().float()
            
            optimizer.zero_grad()
            
            policy_pred, value_pred = net(state)
            
            loss = criterion(value_pred[:, 0], value, policy_pred, policy) # Tính loss
            loss.backward()  # Lan truyền ngược
            optimizer.step()  # Cập nhật trọng số
            
            total_loss += loss.item()
            if i % 10 == 9:
                print(f'Process ID: {os.getpid()} [Epoch: {epoch + 1}, {i + 1} points] total loss per batch: {total_loss / 10:.3f}')
                print(f"Policy: {policy[0].argmax().item()}, Predicted: {policy_pred[0].argmax().item()}")
                print(f"Value: {value[0].item()}, Predicted: {value_pred[0, 0].item()}")
                
                losses_per_batch.append(total_loss / 10)
                total_loss = 0.0
        
        # TÍnh giá trị lỗi trung bình từng epoch
        losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))
        
        # Tiêu chí dừng sớm dựa trên sự hội tụ mất mát
        if len(losses_per_epoch) > 100:
            recent_loss = sum(losses_per_epoch[-4:-1]) / 3
            previous_loss = sum(losses_per_epoch[-16:-13]) / 3
            if abs(recent_loss - previous_loss) <= 0.01:
                print("Early stopping criteria met, ending training.")
                break
        
        scheduler.step()  # Cập nhất hệ số learning rate theo lịch trình
        
    # Vẽ biểu đồ loss theo epoch
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1, epoch_stop + 1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    
    # Lưu lại biểu đồ
    plt.savefig(os.path.join("./model_data/", f"Loss_vs_Epoch_{datetime.datetime.today().strftime('%Y-%m-%d')}.png"))
    print('Finished Training')
