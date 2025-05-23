import os
import cv2
import math
import random
from PIL import Image
import torch
from torch.utils.data import Dataset  # from xxx import yyy, 是导入具体的类
import torch.nn as nn  # import xxx 或 import xxx as yyy, 是导入模块（或者说目录），具体的类或者函数，要进一步在使用的时候yyy.zzz来使用
import torch.nn.functional as F

class DirectDataset(Dataset):
    """继承自基类
    加载---》处理---》供使用
    """
    def __init__(self, root_path, img_transform=None, joint_transform=None):
        super().__init__()
        self.dataset = []
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        
        # images_path = os.path.join(root_path, "images")
        # labels_path = os.path.join(root_path, "labels")
        degrees_label_map = {"0": 0, "90": 1,  "180":2, "270":3}
        # files = [os.path.splitext(x)[0] for x in os.listdir()]
        
        # loader即可，transform在__getitem__中做
        ## 问题2, 用pil还是用opencv呢？应该是opencv，torch.from_numpy()
        ## 问题3，在这个层面的class的操作，就比较随意了
        for degree, label in degrees_label_map.items():
            for file in os.listdir(os.path.join(root_path, degree)):
                # img = cv2.imread(os.path.join(root_path, degree, file))
                img = Image.open(os.path.join(root_path, degree, file))
                self.dataset.append((img, label))
    
    def __len__(self):  # 只有__init__和__getitem__是必须overwrite的
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # return self.transform(self.dataset[idx][0]), self.target_transform(self.dataset[idx][1])  # 很多情形下，target的transform是随着img的变换而变换，如何做呢？
        # return (self.transform(self.dataset[idx][0]), torch.tensor(self.dataset[idx][1], dtype=torch.long))  # 什么时候，数据应该放到cuda上呢？还有模型？
        image, label = self.dataset[idx][0], self.dataset[idx][1]
        
        # 依次进行joint_transform、img_transform
        if self.joint_transform:
            image, label = self.joint_transform(image, label)
        if self.img_transform:
            image = self.img_transform(image)
        return image, label

class Classify_Task(object):
    def __init__(self, classifier, loss=nn.CrossEntropyLoss()):
        self.classifier = classifier
        self.loss = loss
    
    def __call__(self, imgs, targets=None):
        if targets is None:  # inference
            outs = self.classifier(imgs)
            probs = F.softmax(outs, dim=1)  # (B, C)  # 体会到了F.softmax是表示函数，可以直接用；nn.Softmax是类，需要先构建，再使用
            predictions = torch.argmax(probs, dim=1)  # (B)
            return predictions
            
        else:  # train
            outs = self.classifier(imgs)
            loss = self.loss(outs, targets)
            return loss
    
    def export(self, model_path):
        torch.onnx.export(self.classifier, (torch.randn(8, 3, 256, 256, device=next(self.classifier.parameters()).device),), model_path, input_names=["input"], dynamic_axes={'input' : {0 : 'batch_size'}}, opset_version=15)  # (动态batch) https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
        # torch.onnx.export(self.classifier, (torch.randn(8, 3, 256, 256, device=next(self.classifier.parameters()).device),), model_path, input_names=["input"], dynamo=True) # , dynamo=True)

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # in_channel, out_channel, kernel_size, kernel_size设置为3，padding设置为1，可以保证特征的尺度不发生变化
        self.pool1 = nn.MaxPool2d(2, 2)  # kernel_size
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*64*128, 128)  # ??要根据输入图片尺寸，进行推算
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # xxx
    def forward(self, x):
        B, C, H, W = x.shape
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # print(x.shape)
        x = self.fc1(x.view(B, -1))
        x = self.fc2(x)
        out = self.fc3(x)
        
        return out  # logits, (-inf, inf), 表示待送入sigmoid或者softmax的raw网络输出https://stackoverflow.com/questions/41455101/what-is-the-meaning-of-the-word-logits-in-tensorflow

    


class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant', new_w=256, new_h=256):
        # assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        self.new_w = new_w
        self.new_h = new_h
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        temp =  F.pad(img, self.get_padding(img, self.new_w, self.new_h), mode=self.padding_mode, value=self.fill)
        # print(temp.shape)
        
        return temp
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)
    
    def get_padding(self, image, new_w, new_h):    
        # print(type(image), image.shape)
        _, h, w = image.shape
        # max_wh = np.max([w, h])
        # h_padding = (max_wh - w) / 2
        # v_padding = (max_wh - h) / 2
        h_padding = math.ceil((new_w - w) / 2)  # 1/2 --->1
        v_padding = math.ceil((new_h - h) / 2)
        
        l_pad = h_padding
        t_pad = v_padding
        r_pad = new_w - w - l_pad
        b_pad = new_h - h - t_pad
        padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
        # print(l_pad, t_pad, r_pad, b_pad, new_h, new_w, w, h)
        return padding

class RandomRotate(object):
    def __init__(self, random_angles=[0, 90, 180, 270]):
        self.random_angles = random_angles
    
    def __call__(self, image, label):
        angle = random.choices(self.random_angles)
        if angle == 0:
            return image, label  # 原图不旋转
        elif angle == 90:
            image = image.transpose(Image.ROTATE_270)  # 逆时针旋转270度相当于顺时针旋转90度
        elif angle == 180:
            image = img.transpose(Image.ROTATE_180)
        elif angle == 270:
            image = img.transpose(Image.ROTATE_90)
        else:
            raise ValueError("only support 0/90/180/270 degree")
        
        label = int(((label * 90 + angle) % 360) / 90)
        
        return image, label