import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models


transform = transforms.Compose([transforms.Scale(256),  # 缩放图片短边为256，长边为图像原始比例乘以256
                                transforms.CenterCrop(224),  # 在中心裁剪大小为224*224的图片
                                transforms.ToTensor(),
                                # transforms.Normalize()的作用对象需要是一个Tensor，因此在此之前有一个
                                # transforms.ToTensor()就是用来生成Tensor的，并归一化到[0,1]
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])


def test(n):
    test_dataset = torchvision.datasets.ImageFolder(root='/Disk1/Guest/TorchBrain/data/duibi/cam_1/' + str(n) + "/", transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    classes = ('0', '1')

    model = models.alexnet(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(9216, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(p=0.5),
                                    nn.Linear(4096, 2))
    checkpoint = torch.load('Alex5.17-100_params.pkl')
    model.load_state_dict(checkpoint)
    model = model.cuda()
    model.eval()

    RCcorrect = 0  # 定义预测正确的图片数，初始化为0
    RCtotal = 0  # 总共参与测试的图片数，也初始化为0

    ACcorrect = 0
    ACtotal = 0

    PCcorrect = 0
    PCtotal = 0

    TP = 0
    FN = 0
    FP = 0
    TN = 0
    f = open("single_test/cam_1.txt", "w")
    for data in test_loader:  # 循环每一个batch
        images, labels = data

        outputs = model(Variable(images.cuda()))  # 输入网络进行测试
        # print(outputs.data)
        out = str(F.softmax(outputs))
        wz1 = int(out.find(','))
        wz2 = int(out.find(']])'))
        wz3 = wz1 + 2
        num1 = out[9: wz1]
        num2 = out[wz3: wz2]
        _, predicted = torch.max(outputs.data, 1)
        # outputs.data是一个4x10张量，函数将每一行的最大的那一列的值和序号各自
        # 组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
        # 这个 _ , predicted是python的一种常用的写法，表示后面的函数其实
        # 会返回两个值,但是我们对第一个值不感兴趣，就写个_在那里，把它赋值给_就好，
        # 我们只关心第二个值predicted
        # 比如 _ ,a = 1,2 这中赋值语句在python中是可以通过的，
        # 你只关心后面的等式中的第二个位置的值是多少

        if classes[labels[0]] == '1':
            RCtotal = RCtotal + 1
        if classes[labels[0]] == '1' and classes[predicted[0]] == '1':
            RCcorrect = RCcorrect + 1

        if classes[predicted[0]] == '1':
            PCtotal = PCtotal + 1
        if classes[labels[0]] == '1' and classes[predicted[0]] == '1':
            PCcorrect = PCcorrect + 1

        if classes[labels[0]] == '0' and classes[predicted[0]] == '0':
            TN = TN + 1
        if classes[labels[0]] == '0' and classes[predicted[0]] == '1':
            FP = FP + 1
        if classes[labels[0]] == '1' and classes[predicted[0]] == '0':
            FN = FN + 1
        if classes[labels[0]] == '1' and classes[predicted[0]] == '1':
            TP = TP + 1

        f.write('{:s} {:s}\n'.format(classes[labels[0]], classes[predicted[0]]))
        #f.write('{:s} {:s}\n'.format(classes[labels[0]],classes[predicted[0]]))

        ACtotal += labels.size(0)  # 更新测试图片的数量,每次加4
        ACcorrect += (predicted == labels.cuda()).sum()  # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
        # print(correct,'=====',total)
    # print(correct,total)
    f.close()
    print(TP)
    print(FN)
    print(FP)
    print(TN)
    # print('accuracy of the network on the test images: %.3f %%' % (100 * float(ACcorrect) / ACtotal))
    # print('specificity of the network on the test images: %.3f %%' % (100 * float(PCcorrect) / PCtotal))
    # print('negative precision of the network on the test images: %.3f %%' % (100 * float(RCcorrect) / RCtotal))

def testresult():
    for i in range(1, 51):
        print("进行输出+", i)
        test(i)


testresult()
