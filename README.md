微博图片CNN分类测试
====

### 简述
通过微博文本对微博进行分类，将分类后的图片收集整理，使用pytorch cnn进行分类  
后利用训练好的模型对不能通过文本进行分类的微博进行尝试分类  
  
  
### 运行
将已分类的图片按分类名保存至./source/分类名  目录  
运行 make_train_list.php 生成  test.txt  train.txt 用于python3获取目录  
后运行   nohup  python3  -u  cnn.py  > rizhi.log &  进行训练

### 目录
./source   用于存放分类文件夹，每个分类文件夹下存放分类图片
./ans      暂用于存放测试集拼接结果
  
  
### ps
1.未解决分类目前只能传入数字问题  
2.cnn  为摘自git的一个全连接网络  
3.cnn2 为摘自git的一个cnn网络  minist训练集  
4.可以把测试集以及测试输出分离出一个单独的文件进行  
5.可以写一个run.sh进行启动控制  
