import cv2
from PIL import Image   # image 是 PIL库中代表一个图像的类
import numpy.matlib
import numpy as np
from scipy import sparse as sp


def blur(img, count = 10, var1 = 1/4, var2 = 1/2, var3 = 1/4):
    v = [var1, var2, var3]
    diags = np.array([-1, 0, 1])
    B = sp.diags(v, diags, shape=(img.shape[0], img.shape[0]))
    C = sp.diags(v, diags, shape=(img.shape[1], img.shape[1]))
    img_B = numpy.power(B, count) * img
    img_C= img* numpy.power(C, count)
    img_blur = img_B * numpy.power(C, count)
    img_B = img_B.astype(np.uint8)
    img_C = img_C.astype(np.uint8)
    img_blur = img_blur.astype(np.uint8)
    return img_blur, img_B, img_C


def cus_filter2D(image, a11 = 0, a12 = -1, a13 = 0, a21 = -1, a22 = 5, a23 = -1,
                     a31 = 0, a32 = -1, a33 = 0):
    kernel = np.array([[a11, a12, a13],
                       [a21, a22, a23],
                       [a31, a32, a33]],
                      np.float32)  # 默认锐化
    return cv2.filter2D(image, -1, kernel=kernel)


def sketch(img_name):

    # 打开一张图片 “F:\PycharmProjects\cui.jpg” 是图片位置
    a = np.asarray(Image.open("media/" + img_name + "/source.jpg")
                   .convert('L')).astype('float')

    depth = 10.  # 浮点数，预设深度值为10
    grad = np.gradient(a)  # 取图像灰度的梯度值
    grad_x, grad_y = grad  # 分别取横纵图像的梯度值
    grad_x = grad_x * depth / 100.  # 根据深度调整 x 和 y 方向的梯度值
    grad_y = grad_y * depth / 100.
    A = np.sqrt(grad_x ** 2 + grad_y ** 2 + 1.)  # 构造x和y轴梯度的三维归一化单位坐标系
    uni_x = grad_x / A
    uni_y = grad_y / A
    uni_z = 1. / A

    vec_el = np.pi / 2.2  # 光源的俯视角度，弧度值
    vec_az = np.pi / 4.  # 光源的方位角度，弧度值
    dx = np.cos(vec_el) * np.cos(vec_az)  # 光源对 x 轴的影响，np.cos(vec_el)为单位光线在地平面上的投影长度
    dy = np.cos(vec_el) * np.sin(vec_az)  # 光源对 y 轴的影响
    dz = np.sin(vec_el)  # 光源对 z 轴的影响

    b = 255 * (dx * uni_x + dy * uni_y + dz * uni_z)  # 梯度与光源相互作用，将梯度转化为灰度
    b = b.clip(0, 255)  # 为避免数据越界，将生成的灰度值裁剪至0‐255区间

    sketchimg = Image.fromarray(b.astype('uint8'))  # 重构图像
    sketchimg.save("media/"+img_name+"/img_sketch.jpg")  # 保存图片的地址
    return


def relief(image):
    imgInfo = image.shape
    height = imgInfo[0]
    width = imgInfo[1]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # newP = gray0-gray1+150

    # 定义灰度图片
    dst = np.zeros((height, width, 1), np.uint8)

    # 遍历原图片的每一个点
    for i in range(0, height):
        # 因为边界值时相邻像素会溢出
        for j in range(0, width - 1):
            # 当前像素
            grayP0 = int(gray[i, j])
            # 下一个像素。我们宽度-1就是为了防止j+1越界
            grayP1 = int(gray[i, j + 1])
            # 当前像素-下一个像素+150
            newP = grayP0 - grayP1 + 150

            # 当前像素大于255.
            if newP > 255:
                newP = 255
            if newP < 0:
                newP = 0
            dst[i, j] = newP
    return dst
