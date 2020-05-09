# 采用sklearn的聚类算法KMeans Algorithm，识别一幅图片的主颜色
# fork自Github：Siyuan Li，感谢

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import imageio
from skimage import io
from sklearn.cluster import KMeans
import warnings
import os


def color_picker(pic_name):
    """
    The program use the KMeans algorithm, the idea from  Github: Siyuan Li.
    :param pic_name:
    :return: a group of list showing the main RGB color of the picture
    """
    warnings.filterwarnings('ignore')
    k = 4   # 决定有几个中心点
    pic = cv2.imread(pic_name)   # 读取图片
    image = cv2.resize(pic, (100, 100), Image.ANTIALIAS)  # 将图片尺寸改成需要的大小
    new_name = pic_name+'_resized.jpg'
    cv2.imwrite(new_name, image)
    img = io.imread(new_name)   # 读取图片
    img_ori_shape = img.shape   # 图片的维度形状
    assert img_ori_shape[2] == 3
    img1 = img.reshape((img_ori_shape[0] * img_ori_shape[1], img_ori_shape[2]))  # 转换数组的维度为二维数组
    img_shape = img1.shape  # 更改后图片的维度形状
    assert img_shape[1] == 3
    n_channels = img_shape[1]   # 获取图片的维度
    clf = KMeans(n_clusters=k)  # 构造聚类器
    clf.fit(img1)   # 聚类
    centroids = clf.cluster_centers_    # 获取的聚类中心
    labels = list(clf.labels_)      # 标签

    color_info = {}
    for center_index in range(k):
        colorRatio = labels.count(center_index) / len(labels)  # 获取这个颜色中心点的个数占总中心点的ratio
        key = colorRatio
        value = list(centroids[center_index])  # 将对应中心点对应ratio存入字典中
        color_info.__setitem__(key, value)
    color_info_sorted = sorted(color_info.keys(), reverse=True)
    colorInfo = [(k, color_info[k]) for k in color_info_sorted]
    assert color_info.__len__() == k
    for color in colorInfo:
        print('ratio:', color[0], '         color:', color[1])



    # 使用算法跑出的中心点，生成一个矩阵，为数据可视化做准备
    result_width = 200  # 可视化结果矩阵的宽度
    result_height = 300  # 可视化结果矩阵的高度
    result = []
    height = []
    for center_indexx in range(k):
        height.append(int(result_height * colorInfo[center_indexx][0]))
    real_height = sum(height)
    for center_index in range(k):
        # 为每一个颜色中心，创建一个shape为（长*宽*该颜色ratio）x3的矩阵，填充数字为每种颜色的RGB数值
        result.append(
            np.full((result_width * height[center_index], n_channels), colorInfo[center_index][1], dtype=int))
    i = 0
    d = np.ones(shape=(1, n_channels))
    while i < k:
        d = np.concatenate((d, result[i]), axis=0)
        i += 1
    result = np.delete(d, 0, axis=0)
    result = result.reshape(real_height, result_width, n_channels)
    io.imsave(os.path.splitext(pic_name)[0] + '_result.bmp', result)


color_picker("A.jpg")

