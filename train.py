import os
import torch.utils.data
import torchvision.transforms

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from math import ceil
from utils import DirectDataset, Classify_Task, Classifier
# # SimpleClassifier

# ## data
# * dataset
#     * transform
# * dataloader

# ## model
# * backbone

# ## forward(similar with model)

# ## loss
# * ce loss

# ## backward
# * backward

# ## weight/bias update
# * optimizer
# * sgd


# ## metric

# ## export

## 悟1：框架负责抽象类的定义

root_path = r"../../../autodl-tmp/direction_dataset/"
# transform
## 减均值，除以方差，rgb2bgr, to_tensor, 这些应该来讲torch中都有，悟2：有了就使用，无了就继承
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.Normalize(mean = [0, 0, 0], std= [1, 1, 1]),
                                            ])  # 这里，应该再增加一个pad_resize到256，先直接resize吧
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Resize((256, 256)),
                                            torchvision.transforms.Normalize(mean = [0, 0, 0], std= [1, 1, 1]),
                                            ])  # 这里，应该再增加一个pad_resize到256，先直接resize吧
direct_dataset = DirectDataset(root_path, transform)  # 问题，如何将区分trainval呢？
train_dataloader = DataLoader(direct_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(direct_dataset, batch_size=8, shuffle=False) # 这里偷了一个懒
        

classifier = Classifier(num_classes=4)
classifier.to(device="cuda")  # 两移动：model和data(注：包含image和label)
classifier_task = Classify_Task(classifier)

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)  # optimizer四板斧1：构造
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)  # optimizer其实就是solver, solvers/optimizers
# optimizer四板斧: 1构造；2清零；3执行; 4调整学习率

epoches = 100
for epoch in range(epoches):
    total_iter_num = ceil(len(direct_dataset) / train_dataloader.batch_size)
    # print(len(next(iter(train_dataloader)))) # , len(next(iter(train_dataloader))))
    # print(len(next(iter(train_dataloader))))
    for idx in range(total_iter_num):
        image_label = next(iter(train_dataloader))
        # for image_label in next(iter(train_dataloader)):  # 如果一个iter, 它到底了，如何拨回0点呢？,网上的说法是dataloader内置了自动重置状态的操作
        image, label = image_label[0].to(device="cuda"), image_label[1].to(device="cuda")
        # print("image.shape:", image.shape, "label.shape:", label.shape)
        optimizer.zero_grad()  # optimizer四板斧2：清零
        loss = classifier_task(image, label)  # 思考: classifier到底是一个Module还是一个什么东东？
        loss.backward()  # 如果这样的话，loss只能为tensor
        optimizer.step()  # optimizer四板斧3：执行参数调整
        # if idx % 5 == 0:
        #     print(f"Item: {idx}/{total_iter_num}, loss: {loss}")
    scheduler.step()  # optimizer四板斧4： 执行学习率调整
    print(f"Epoch: {epoch}/{epoches}, loss: {loss}")

# # evaluation
# predictions = []
# labels = []
# for image, label in next(iter(val_dataloader)):
#     prediction = classifier(image)
    
#     predictions.extend(prediction)
#     labels.extend(label)

# acc = torch.sum(torch.where(predictions == labels)) / labels.numel()
# print(f"Acc: {acc}")    


# # export
# ## save pth
# torch.save(classifier.state_dict(), "save.pth")
# ## export pth
# classifier.export("save.pt")





