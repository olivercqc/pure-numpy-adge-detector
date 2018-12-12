import numpy as np
import cv2

def preprocessing(rawimg):
    """
    :param img: 传入某一副图像
    :return: 返回读入图像的灰度图像
    """
    src = cv2.imread(rawimg)
    processedSrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    return processedSrc

def pad_with(array, pad_width, iaxis, kwargs):
    """
    此函数作为padding函数的参数函数
    :param array: 传入想要进行padding操作的矩阵
    :param pad_width: 每一个坐标系边缘进行padding操作的层数
    :param kwargs: 可变参数，可用来指定进行padding操作的数值，默认为0
    """
    pad_value = kwargs.get("padder", 0)
    array[:pad_width[0]] = pad_value
    array[-pad_width[1]:] = pad_value
    return array

def padding(array, pad_width):
    """

    :param array: 传入想要进行padding操作的矩阵
    :param pad_width: 每一个坐标系边缘进行padding操作的层数
    :return: padding操作后的矩阵
    """
    paddingarray = np.pad(array, pad_width, pad_with)
    return paddingarray

def convolution(rowlength, columnlength, kernel, array):
    """

    :param rowlength: 矩阵的行数
    :param columnlength: 矩阵的列数
    :param kernel: 进行卷积操作的卷积核
    :param array: 与卷积核进行卷积操作的矩阵
    :return: 卷积操作完成后的矩阵，即卷积操作完成后的图像
    """
    temp = np.zeros((rowlength, columnlength), dtype=np.float_)
    for i in range(rowlength):
        for j in range(columnlength):
            temp[i][j] = np.inner(kernel, array[i:i+3, j:j+3].flatten())
    return temp



if __name__ == "__main__":
    # 将原始的图像经预处理变为灰度图像
    # 将已经变换的灰度图像增加一层padding
    gray_lena = preprocessing("chessboard.png")
    padding_lena = padding(gray_lena, 1)
    # 获取lena图像的长与宽，方便进行后续的卷积操作
    rowlength = gray_lena.shape[0]
    columnlength = gray_lena.shape[1]
    # 创建prewitt算子在x方向和y方向上的卷积核
    prewitt_kernel_x = np.array([-1, -1, -1,
                                 0, 0, 0,
                                 1, 1, 1])
    prewitt_kernel_y = np.array([-1, 0, 1,
                                 -1, 0, 1,
                                 -1, 0, 1])
    # 创建sobel算子在x方向和y方向上的卷积核
    sobel_kernel_x = np.array([-1, -2, -1,
                               0, 0, 0,
                               1, 2, 1])
    sobel_kernel_y = np.array([-1, 0, 1,
                               -2, 0, 2,
                               -1, 0, 1])
    # 生成prewitt算子处理后的图像并保存
    prewitt_gx = convolution(rowlength, columnlength, prewitt_kernel_x, padding_lena)
    prewitt_gy = convolution(rowlength, columnlength, prewitt_kernel_y, padding_lena)
    prewitt_g = np.abs(prewitt_gx) + np.abs(prewitt_gy)
    cv2.imwrite("prewitt_gx.jpg", prewitt_gx)
    cv2.imwrite("prewitt_gy.jpg", prewitt_gy)
    cv2.imwrite("prewitt_g.jpg", prewitt_g)
    # 生成sobel算子处理后的图像并保存
    sobel_gx = convolution(rowlength, columnlength, sobel_kernel_x, padding_lena)
    sobel_gy = convolution(rowlength, columnlength, sobel_kernel_y, padding_lena)
    sobel_g = np.abs(sobel_gx) + np.abs(sobel_gy)
    cv2.imwrite("sobel_gx.jpg", sobel_gx)
    cv2.imwrite("sobel_gy.jpg", sobel_gy)
    cv2.imwrite("sobel_g.jpg", sobel_g)