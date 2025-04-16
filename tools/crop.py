import os
import cv2
import numpy as np

def transform_and_crop(image, src, dst, output_width, output_height):
    """
    :param image: 输入图像
    :param src: 四边形四个顶点的坐标，格式为np.ndarray([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    :param dst: 四边形四个顶点的坐标，格式为np.ndarray([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
    :return: 裁剪后的矩形图像
    """
    # 计算透视变换矩阵（注意这里使用getPerspectiveTransform而不是仿射变换）
    M = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    
    # 执行透视变换
    warped = cv2.warpPerspective(image, M, (int(output_width), int(output_height)))
    
    return warped


# 检测数据集

# 数据集中每一个样本 （图片， txt）

# 如果是文本的话，则将其仿射变换crop出来

# 分类数据集

detection_dataset_root = r"../../../autodl-tmp/project-ocr-202504/"  # images和labels两个子文件夹
# detection_dataset_root = r"./debug"  # images和labels两个子文件夹
detection_dataset_images_path = os.path.join(detection_dataset_root, "images")
detection_dataset_labels_path = os.path.join(detection_dataset_root, "labels")
widths = []
heights = []
samples = [os.path.splitext(x)[0] for x in os.listdir(detection_dataset_images_path)]  # without ext
for sample in samples:
    img_path = os.path.join(detection_dataset_images_path, sample + ".bmp")
    label_path = os.path.join(detection_dataset_labels_path, sample + ".txt")
    
    img = cv2.imread(img_path)
    h_img, w_img, _ = img.shape
    with open(label_path, "r") as fr:
        lines = fr.readlines()
    
    for line in lines:
        line = line.rstrip('\n')
        s_res = line.split(" ")
        label = s_res[0]
        if label == "2":  # 只有text类才crop下来
            coors_normalized = np.array([float(x) for x in s_res[1:]])  # 将普通List转变为np.ndarray，方便使用一些slice操作
            coors_normalized[0::2] = coors_normalized[0::2] * w_img
            coors_normalized[1::2] = coors_normalized[1::2] * h_img
                        
            xmin = np.min(coors_normalized[::2])
            xmax = np.max(coors_normalized[::2])
            ymin = np.min(coors_normalized[1::2])  # 注意，这里是::
            ymax = np.max(coors_normalized[1::2])
            x_margin = xmax - xmin
            y_margin = ymax - ymin
            
            direction = "horizonal"
            # if y_margin > 1.5 * x_margin:
            #     direction = "vertical"
            
            
            coors_numpy = coors_normalized.reshape(-1, 2)
            
            # 确保按照[left-top, right-top, right-bottom, left-bottom]
            center_x, center_y = np.mean(coors_numpy, 0)
            left_id = coors_numpy[:, 0] < center_x
            right_id = coors_numpy[:, 0] > center_x
            coors_numpy_list = []
            coors_numpy_left = coors_numpy[left_id]  # (2, 2)
            if coors_numpy_left[0][1] > coors_numpy_left[1][1]: # compare y
                coors_numpy_left = coors_numpy_left[[1, 0]]                          
            coors_numpy_right = coors_numpy[right_id]  # (2, 2)
            if coors_numpy_right[0][1] > coors_numpy_right[1][1]: # compare y
                coors_numpy_right = coors_numpy_right[[1, 0]]          
            
            coors_numpy_list = [coors_numpy_left[0], coors_numpy_right[0], coors_numpy_right[1], coors_numpy_left[1]]
            coors_numpy = np.stack(coors_numpy_list, 0)
            
            src = coors_numpy
            edge1 = np.linalg.norm(coors_numpy[0] - coors_numpy[1])
            edge2 = np.linalg.norm(coors_numpy[1] - coors_numpy[2])      
            if direction == "horizonal":
                dst = np.array([[0, 0],
                               [edge1, 0],
                               [edge1, edge2],
                               [0, edge2]])
                output_width = edge1
                output_height = edge2
            else:
                dst = np.array([[edge2, 0],
                                [edge2, edge1],
                                [0, edge1],
                                [0, 0]])
                output_width = edge2
                output_height = edge1
            widths.append(output_width)
            heights.append(output_height)
            img_crop = transform_and_crop(img, src, dst, output_width, output_height)
            symbol = "_".join([str(int(x)) for x in list(coors_numpy.reshape(-1))])
            cv2.imwrite(f"./classifier_dataset/{sample}_{symbol}.bmp", img_crop)
            # cv2.imwrite(f"./debug/{sample}_{symbol}.bmp", img_crop)
            
        
print(f"width: {np.max(np.array(widths))}, height: {np.min(np.array(heights))}")
