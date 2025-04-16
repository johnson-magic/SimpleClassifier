import os
import torch.utils.data
import torchvision.transforms

from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import onnxruntime as ort
import numpy as np
from math import ceil
import onnxruntime as ort
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
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8], gamma=0.1)  # optimizer其实就是solver, solvers/optimizers
# optimizer四板斧: 1构造；2清零；3执行; 4调整学习率

epoches = 10
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

# evaluation
classifier.eval()
predictions = []
labels = []
for i, (image, label) in enumerate(iter(val_dataloader)):
    image, label = image.to(device="cuda"), label.to(device="cuda")
    prediction = classifier_task(image)
    
    predictions.extend(list(prediction.cpu().numpy()))
    labels.extend(list(label.cpu().numpy()))

acc = np.sum(np.array(predictions) == np.array(labels)) / len(labels)
print(f"Acc: {acc}")    


# # export
# ## save pth
torch.save(classifier.state_dict(), "save.pth")

classifier_loaded = Classifier(num_classes=4)
classifier_loaded.load_state_dict(torch.load("save.pth", weights_only=True))  #  map_location=torch.device("cuda")，并非是指model在cuda上的意思
classifier_loaded.to("cuda")
classifier_loaded.eval()
classifier_task_loaded = Classify_Task(classifier_loaded)
predictions = []
labels = []
for i, (image, label) in enumerate(iter(val_dataloader)):
    image, label = image.to(device="cuda"), label.to(device="cuda")
    prediction = classifier_task_loaded(image)
    
    predictions.extend(list(prediction.cpu().numpy()))
    labels.extend(list(label.cpu().numpy()))

acc = np.sum(np.array(predictions) == np.array(labels)) / len(labels)

print(f"Acc by loaded model: {acc}")

# export pth
classifier_task.export("save.onnx")

ort_sess = ort.InferenceSession('save.onnx')
for i, (image, label) in enumerate(iter(val_dataloader)):
    outputs = ort_sess.run(None, {'input': image.numpy()})
    predictions.extend(list(outputs[0].argmax(1)))
    labels.extend(list(label.cpu().numpy()))
acc = np.sum(np.array(predictions) == np.array(labels)) / len(labels)
print(f"Acc by loaded model: {acc}")







