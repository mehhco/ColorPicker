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
    k = 3   # 决定有几个中心点
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
    for color in colorInfo:
        print('ratio:', color[0], '         color:', color[1])

    # 使用算法跑出的中心点，生成一个矩阵，为数据可视化做准备
    result = []
    result_width = 200
    result_height_per_center = 80
    for center_index in range(k):
        result.append(
            np.full((result_width * result_height_per_center, n_channels), colorInfo[center_index][1], dtype=int))
    result = np.array(result)
    result = result.reshape((result_height_per_center * k, result_width, n_channels))

    # 保存图片
    io.imsave(os.path.splitext(pic_name)[0] + '_result.bmp', result)


color_picker("Allan.jpg")

