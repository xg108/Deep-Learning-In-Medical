'''
用预训练模型实现alexnet训练cifar这个数据集
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
from focalloss import *

#  首先定义了一个变换transform，利用的是上面提到的transforms模块中的Compose( )
#  把多个变换组合在一起，可以看到这里面组合了ToTensor和Normalize这两个变换
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
            transforms.Scale(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
transform_test = transforms.Compose([transforms.Scale(256),#缩放图片短边为256，长边为图像原始比例乘以256
                                transforms.CenterCrop(224),#在中心裁剪大小为224*224的图片
                                transforms.ToTensor(),
                                #transforms.Normalize()的作用对象需要是一个Tensor，因此在此之前有一个
                                #transforms.ToTensor()就是用来生成Tensor的，并归一化到[0,1]
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


train_dataset = torchvision.datasets.ImageFolder(root='data/VOCdevkit/VOC2007/classifypic/train', transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
print(len(train_loader))


test_dataset = torchvision.datasets.ImageFolder(root='data/VOCdevkit/VOC2007/classifypic/test', transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)
print(len(test_loader))

# 类别信息也是需要我们给定的
classes = ('0', '1')

'''
以下几行代码是将已经标准化了的图像数据恢复成原始数据，以便于显示输出
'''
X_train,y_train = next(iter(train_loader))
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
img = torchvision.utils.make_grid(X_train)
img = img.numpy().transpose((1,2,0))
img = img*std + mean


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, 2)




for name,parma in model.named_parameters():
    parma.requires_grad = True
# checkpoint = torch.load('Alex5.16-70-2_params.pkl')
# model.load_state_dict(checkpoint)

print(model)
model=model.cuda()

#criterion是规范、评判标准的意思。而optimizer是优化器的意思
# criterion = nn.CrossEntropyLoss().cuda()
criterion = FocalLoss(gamma=5, alpha=0.75,size_average=False).cuda()


'''
开始训练
'''
lr1 = 0.0001
lr2 = 0.00001
lr3 = 0.000001
f = open("ACC/resnet50/5.18_acc.txt", "w")
for epoch in range(50):  # 遍历数据集k次

    running_loss = 0.0
    running_correct = 0.0
    batch = 0.0
    # scheduler.step()  #更新学习率
    # if epoch > 25 & epoch <40:
    #     optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr2)
    # elif epoch>=40:
    #     optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr3)
    # else:
    # optimizer = torch.optim.SGD(model.classifier.parameters(), lr=lr1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr1)
    # enumerate(sequence, [start=0])，i是序号，data是数据.
    # 这里我们遇到了第一步中出现的trainloader，代码传入数据
    # enumerate是python的内置函数，既获得索引也获得数据
    #enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在for 循环当中。
    for i, data in enumerate(train_loader,0):# i获取索引，从0开始计，方便每2000次进行打印。而data是获取数据和对应标签
        batch+=1
        # get the inputs
        inputs, labels = data  # data是从enumerate返回的data，包含数据和标签信息，分别赋值给inputs和labels
                            # data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())  # 把input数据从tensor转为Variable

        # zero the parameter gradients
        optimizer.zero_grad() # 要把梯度重新归零，因为反向传播过程中梯度会累加上一次循环的梯度


        # forward + backward + optimize
        outputs = model(inputs) #把数据输进网络net，这个net()在第二步的代码最后一行我们已经定义了
        _, pred = torch.max(outputs.data, 1)
        # loss = criterion(outputs, labels.data)# 将output和labels使用交叉熵计算损失，使用交叉熵损失函数的时候会
                                                # 自动把labels转化成onehot，不用手动转化。
        loss = criterion(outputs, labels)
        loss.backward()  # 反向传播
        optimizer.step()  # 当执行反向传播之后，把优化器的参数进行更新，以便进行下一轮
        # 这几行代码不是必须的，为了打印出loss方便我们看而已，不影响训练过程
        # running_loss += loss.item() # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
                                    # 从下面一行代码可以看出它是每循环0-1999共两千次才打印一次
        running_loss += loss.data[0]
       # print(pred,'========',labels)
        running_correct += torch.sum(pred == labels.data)#这里得到的running_correct的type是tensor，比如在训练集的
    # 50000张图片中某一轮最后总共有40000张预测正确，则running_correct为tensor(40000),则在计算准确率时，
        # 必须将其转为float型再去除以总的图像数，才能得到80%，否则结果为0%

        if batch % 50 == 0:
            # print(running_loss,running_correct,'----',4*batch)
            print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.3f}".format(
                batch, running_loss / (4 * 50), float( running_correct) / float(4 * 50)))
            f.write("Batch {}, Train Loss:{:.4f}, Train ACC:{:.3f}\n".format(
                batch, running_loss / (4 * 50), float(running_correct) / float(4 * 50)))
            running_loss=0.0
            running_correct = 0.0
        # if i % 2000 == 1999: # print every 2000 mini-batches 所以每个2000次之内先用running_loss进行累加
        #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 8000)) # 然后再除以2000，就得到
        #                                                                             # 这两千次的平均损失值
        #     running_loss = 0.0 # 这一个2000次结束后，就把running_loss归零，下一个2000次继续使用
print('Finished Training')
# 保存神经网络
torch.save(model.state_dict(), 'resnet50_pkl/Resnet5.18-50_params.pkl')  # 保存整个神经网络的结构和模型参数
# torch.save(model.state_dict(), 'Alexmodel_params2.pkl')  # 只保存神经网络的模型参数
f.close()
'''
开始测试
'''
model.eval()
correct = 0   # 定义预测正确的图片数，初始化为0
total = 0     # 总共参与测试的图片数，也初始化为0
for data in test_loader: # 循环每一个batch
    images, labels = data
    outputs = model(Variable(images.cuda())) # 输入网络进行测试
    #print(outputs.data)
    _, predicted = torch.max(outputs.data, 1)  # outputs.data是一个4x10张量，函数将每一行的最大的那一列的值和序号各自
                                                #组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
                                                # 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实
                                                #会返回两个值,但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，
                                                # 我们只关心第二个值predicted
                                                #比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，
                                                # 你只关心后面的等式中的第二个位置的值是多少
    # print(predicted,'-----',labels)
    total += labels.size(0)# 更新测试图片的数量,每次加4
    correct += (predicted == labels.cuda()).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
    # correct += predicted.eq(labels.view_as(predicted)).sum().item()
    print(correct,'=====',total)
#print(correct,total)
print('Accuracy of the network on the test images: %.3f %%' % (100 * float(correct) / total))


