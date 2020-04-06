# coding=utf-8
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms, utils  
from PIL import Image
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

 
# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 定义超参数
input_size = 784*3
hidden_size = 500
num_classes = 7
num_epochs = 50
batch_size = 16
learning_rate = 0.001
 
def default_loader(path):  
    # 注意要保证每个batch的tensor大小时候一样的。  
    return Image.open(path).convert('RGB')  
  
class MyDataset(Dataset):  
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):  
        fh = open(txt, 'r')  
        imgs = []  
        for line in fh:  
            line = line.strip('\n')  
            # line = line.rstrip()  
            words = line.split(' ')  
            imgs.append((words[0],int(words[1])))  
        self.imgs = imgs  
        self.transform = transform  
        self.target_transform = target_transform  
        self.loader = loader  
      
    def __getitem__(self, index):  
        fn, label = self.imgs[index]  
        img = self.loader(fn)  
        if self.transform is not None:  
            img = self.transform(img)  
        return img,label
      
    def __len__(self):  
        return len(self.imgs)  
  
def get_loader(dataset='train.txt', crop_size=128, image_size=32, batch_size=2, mode='train', num_workers=1):  
    """Build and return a data loader."""  
    transform = []  
    if mode == 'train':  
        transform.append(transforms.RandomHorizontalFlip())  
    transform.append(transforms.CenterCrop(crop_size))  
    transform.append(transforms.Resize(image_size))  
    transform.append(transforms.ToTensor())  
    transform.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))  
    transform = transforms.Compose(transform)  
    train_data=MyDataset(txt=dataset, transform=transform)  
    data_loader = DataLoader(dataset=train_data,  
                                  batch_size=batch_size,  
                                  shuffle=(mode=='train'),  
                                  num_workers=num_workers)  
    return data_loader  
# 注意要保证每个batch的tensor大小时候一样的。  
# data_loader = DataLoader(train_data, batch_size=2,shuffle=True)  
train_loader = get_loader('train.txt', batch_size=batch_size)  
print(len(train_loader))  
test_loader = get_loader('test.txt', batch_size=batch_size)  
print(len(test_loader))  
 
#Lenet网络代码
class Lenet(nn.Module):
    def __init__(self):
        super(Lenet,self).__init__()
        #定义网络层
        #入通道数，出通道数，卷积尺寸
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,7)
    
    #将二维数据展开成一维数据以输入到全连接层
    def num_flat_features(self,x):
        #size为[batch_size,num_channels,height,width]
        #除去batch_size,num_channels*height*width就是展开后维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features
    
    def forward(self,x):
        #定义前向传播
        #输入 和 窗口尺寸
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
# 定义模型
# model = NeuralNet(input_size, hidden_size, num_classes).to(device)
net = Lenet()
optimizer = optim.Adam(net.parameters(), lr = 0.001, betas=(0.9, 0.99))
#在使用CrossEntropyLoss时target直接使用类别索引，不适用one-hot
loss_fn = nn.CrossEntropyLoss()

loss_list = []
total_step = len(train_loader)
for epoch in range(1,num_epochs+1):
    #训练部分
    for step,(x,y) in enumerate(train_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = net(b_x)
        loss = loss_fn(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #记录loss
        if step%50 == 0:
            loss_list.append(loss)
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, step+1, total_step, loss.item()))
    #每完成一个epoch进行一次测试观察效果
    pre_correct = 0.0
    test_loader = Data.DataLoader(dataset=test_data, batch_size = 100, shuffle=True)
    for (x,y) in (test_loader):
        b_x = Variable(x)
        b_y = Variable(y)
        output = net(b_x)
        pre = torch.max(output,1)[1]
        pre_correct = pre_correct+float(torch.sum(pre==b_y))
    print('EPOCH:{epoch},ACC:{acc}%'.format(epoch=epoch,acc=(pre_correct/float(10000))*100))

# 模型测试部分
# 测试阶段不需要计算梯度，注意
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels, fn in test_loader:
#         images = images.reshape(-1, 28*28*3).to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         print( str(fn)+"\t"+str(predicted)+"\t"+str(labels)+"\t\n")
 
#     print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
 
# 保存模型参数
torch.save(model.state_dict(), 'model-cnn2.ckpt')



