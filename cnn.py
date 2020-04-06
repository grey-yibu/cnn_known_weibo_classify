# coding=utf-8
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader  
from torchvision import transforms, utils  
# from PIL import Image
from PIL import Image, ImageFont, ImageDraw # 导入模块
import random

 
# 判定GPU是否存在
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 定义超参数
input_size = 784*3
hidden_size = 100
num_classes = 7
num_epochs = 50
batch_size = 16
learning_rate = 0.001
 
pic_type = {0:"meinv",1:"zufang",2:"fengjing"}
ttfont = ImageFont.truetype("msyh.ttf",50)  


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
        return img,label,fn 
      
    def __len__(self):  
        return len(self.imgs)  
  
def get_loader(dataset='train.txt', crop_size=128, image_size=28, batch_size=2, mode='train', num_workers=1):  
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
 
# 定义含有一个隐含层的全连接神经网络。
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
 
# 定义模型
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
 
# 损失函数和优化算法
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  
 
# 训练模型
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels, fn) in enumerate(train_loader):  
        # Move tensors to the configured device
        images = images.reshape(-1, 28*28*3).to(device)
        labels = labels.to(device)
        # print (images, labels)
        
        # 前向传播和计算loss
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 后向传播和调整参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 每100个batch打印一次数据
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
 
# 模型测试部分
# 测试阶段不需要计算梯度，注意
with torch.no_grad():
    correct = 0
    total = 0
    # for images, labels, fn in test_loader:
        # images = images.reshape(-1, 28*28*3).to(device)
        # labels = labels.to(device)
        # outputs = model(images)
        # _, predicted = torch.max(outputs.data, 1)
        # total += labels.size(0)
        # correct += (predicted == labels).sum().item()
        # print( str(fn)+"\t"+str(predicted)+"\t"+str(labels)+"\t\n")
    for i, (images, labels, fn) in enumerate(test_loader):  
        # Move tensors to the configured device
        a = images
        images = images.reshape(-1, 28*28*3).to(device)
        labels = labels.to(device)
        # 前向传播和计算loss
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        fn_list = fn
        images_list = a.numpy().tolist()
        predicted_list = predicted.numpy().tolist()
        labels_list = labels.numpy().tolist()

        for  i  in  range(0,len(fn_list)):
            if( predicted_list[i]!= labels_list[i]):
                tmp_filename_zu = fn_list[i].split('/')
                tmp_real_filename = tmp_filename_zu[-1]
                im = Image.open(fn_list[i])
                draw = ImageDraw.Draw(im)
                draw.text((10,10), pic_type[predicted_list[i]], fill = (255, 0 ,0),font=ttfont) #利用ImageDraw的内置函数，在图片上写入文字
                draw.text((40,40), pic_type[labels_list[i]], fill = (0, 255 ,0),font=ttfont) #利用ImageDraw的内置函数，在图片上写入文字
                im.save("./ans/"+pic_type[predicted_list[i]]+"_"+pic_type[labels_list[i]]+"_"+str(random.randint(0,999))+tmp_real_filename)

            # tmp_img = transforms.ToPILImage()( torch.Tensor(images_list[i])  )
            # tmp_img.save("./ans/"+pic_type[predicted_list[i]]+"_"+pic_type[labels_list[i]]+"_"+tmp_real_filename )

 
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
 
# 保存模型参数
torch.save(model.state_dict(), 'model-cnn.ckpt')



def predict(img_path):
    net = torch.load('model.pkl') 
    net = net.to(device)
    torch.no_grad()
    img = PIL.Image.open(img_path)
    img_ = transform(img).unsqueeze(0)
    img_ = img_.to(device)
    outputs = net(img_)
    _, predicted = torch.max(outputs,1)
    print(predicted)

