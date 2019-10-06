import cv2
import numpy as np
import math
from alyn import SkewDetect
from alyn import Deskew


def skew_detect_canny(file_path):
    sd = SkewDetect(
        input_file=file_path,
        display_output='Yes')
    sd.run()

def skew_pic(file_path):
    d = Deskew(
        input_file=file_path,
        display_image=True,
        # output_file='demo_o.jpg',
        output_file="/".join(file_path.split("/")[:-1]) + "/" + file_path.split("/")[-1:][0].split(".")[:-1][0] + "_o.jpg",
        r_angle=0)
    d.run()

def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, tuple(pt1), tuple(pt2), color, lineWidth)
    cv2.line(img, tuple(pt2), tuple(pt3), color, lineWidth)
    cv2.line(img, tuple(pt3), tuple(pt4), color, lineWidth)
    cv2.line(img, tuple(pt1), tuple(pt4), color, lineWidth)


def rotate(file_path):
    img = cv2.imread(file_path)

    # 图片去噪
    img_c = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 灰度图
    gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", gray)

    # 如果是纯文本，可省略此步。本步通过边缘检测提取文本(可用sobel或canny算子)
    # Sobel算子，x方向求梯度,主要用于获得数字图像的一阶梯度，常见的应用和物理意义是边缘检测。
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    cv2.imshow("sobel", sobel)

    # 二值化(采用了简单阈值函数，通过otsu自动寻找阈值，将灰度图sobel二值化)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)

    # 霍夫直线
    hufu = binary.astype(np.uint8)
    # 参数:灰度图或canny图，rou精度，theta精度，直线最少曲线交点，最小直线像素点，一条直线亮点最大距离
    lines = cv2.HoughLinesP(hufu, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=20)

    for line in lines:
        cv2.line(img, (line[0][0], line[0][1]), (line[0][2], line[0][3]), (0, 0, 255), 2)

    k_dict = {}
    k = 0

    # 求出所有直线斜率，求出众数（考虑误差）
    for line in lines:
        if line[0][2] - line[0][0] == 0:  # 跳过一个点
            continue
        print(line[0][3], line[0][1], line[0][2], line[0][0])
        k = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        # α = atan(k) * 180 / PI
        k = math.atan(k) * 180 / np.pi
        if len(k_dict.keys()) == 0:
            k_dict[k] = 1
        else:
            flag = False
            for item in k_dict.keys():
                if abs(item - k) < 2:  # 考虑误差，斜率差最小为2
                    flag = True
                    k_dict[item] += 1
                    break
            if not flag:
                k_dict[k] = 1

    must_k_num = 0
    must_key = 0
    for item in k_dict.keys():
        if k_dict[item] > must_k_num:
            must_k_num = k_dict[item]
            must_key = item
    print("图像朝向的角度：", must_key)

    # 旋转图像
    h, w = img.shape[:2]
    add_w = int((((w * w + h * h) ** 0.5) - w) / 2)
    add_h = int((((w * w + h * h) ** 0.5) - h) / 2)
    print(add_w, add_h)

    img = cv2.copyMakeBorder(img, add_h, add_h, add_w, add_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, must_key, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    destination_path = "o/" + file_path.split("/")[1]
    print(destination_path)
    cv2.imwrite(destination_path, rotated)

    cv2.imshow("houghP", img)
    cv2.imshow("rotated", rotated)
    cv2.waitKey(0)


def rotated_img_with_fft(file_path):
    img = cv2.imread(file_path)

    cv2.imshow("img", img)
    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图像延扩
    '''OpenCV中的DFT采用的是快速算法，这种算法要求图像的尺寸是2的、3和5的倍数是处理速度最快。
    所以需要用getOptimalDFTSize()找到最合适的尺寸，
    然后用copyMakeBorder()填充多余的部分。这里是让原图像和扩大的图像左上角对齐。
    填充的颜色如果是纯色，对变换结果的影响不会很大，后面寻找倾斜线的过程又会完全忽略这一点影响。
'''
    h, w = gray.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(gray, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)
    cv2.imshow("enlarge_img", nimg)

    # 执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)  # 图像从空间域->频域
    fshift = np.fft.fftshift(f)  # 将低频分量移动到中心

    fft_img = np.log(np.abs(fshift))
    fft_img = (fft_img - np.amin(fft_img)) / (np.amax(fft_img) - np.amin(fft_img))

    fft_img *= 255  # 实现归一化
    ret, thresh = cv2.threshold(fft_img, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)

    # 霍夫直线变换
    '''因为HougnLinesP()函数要求输入图像必须为8位单通道图像，
    所以我们用astype()函数把图像数据转成uint8类型，
    接下来执行二值化操作。在操作过程中参数要自己根据情况设定。'''
    thresh = thresh.astype(np.uint8)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, 30, minLineLength=40, maxLineGap=100)
    try:
        lines1 = lines[:, 0, :]
    except Exception as e:
        lines1 = []

    # 创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape, dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi / 180
    pi2 = np.pi / 2
    angle = 0
    for line in lines1:
        # x1, y1, x2, y2 = line[0]
        x1, y1, x2, y2 = line
        cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            angle = abs(theta)
            break

    # 计算倾斜角，将弧度转为角度，并考虑误差
    angle = math.atan(angle)
    angle = angle * (180 / np.pi)
    print(angle)
    # cv2.imshow("line image", lineimg)
    center = (w // 2, h // 2)
    height_1 = int(w * math.fabs(math.sin(math.radians(angle))) + h * math.fabs(math.cos(math.radians(angle))))
    width_1 = int(h * math.fabs(math.sin(math.radians(angle))) + w * math.fabs(math.cos(math.radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (width_1 - w) / 2
    M[1, 2] += (height_1 - h) / 2
    rotated = cv2.warpAffine(gray, M, (width_1, height_1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cv2.imshow('rotated_img_with_fft', rotated)
    cv2.waitKey(0)
    return rotated


def rotated_img_with_radiation(file_path, is_show=False):
    img = cv2.imread(file_path)

    # 图片去噪
    # img_c = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    # 灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    if is_show:
        cv2.imshow('thresh', thresh)
    # 计算包含了旋转文本的最小边框
    coords = np.column_stack(np.where(thresh > 0))
    # 画框
    min_rect = cv2.minAreaRect(coords)  # 由点集获取最小矩形（包含中心坐标点、宽和高、偏转角度）
    box = cv2.boxPoints(min_rect)  # 获取最小矩形的4个顶点坐标。
    drawRect(img, box[0], box[1], box[2], box[3], (255, 0, 255), 5)
    cv2.imshow("img", img)
    # 该函数给出包含着整个文字区域矩形边框，这个边框的旋转角度和图中文本的旋转角度一致
    angle = cv2.minAreaRect(coords)[-1]
    print(angle)
    # 调整角度
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    # 仿射变换
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    print(angle)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if is_show:
        cv2.putText(rotated, 'Angle: {:.2f} degrees'.format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        print('[INFO] angel :{:.3f}'.format(angle))
        cv2.imshow('Rotated_image_with_radiation', rotated)
        cv2.waitKey()
    return rotated


def expand_pic(img_file):
    img = cv2.imread(img_file)
    kernel = (5, 5)
    dilation = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    blur = cv2.bilateralFilter(dilation, 9, 75, 75)
    cv2.imshow("blur", blur)
    cv2.waitKey(0)


expand_pic(r"C:\Users\63423\Desktop\pic1.png")

# rotate("pic/318.jpg")
# rotated_img_with_fft("pic/1529897101_0_9218_aug_1552142585.83_1_rotate-0-0-264.jpg")
# rotated_img_with_radiation("pic/1529897101_0_9218_aug_1552142585.83_1_rotate-0-0-264.jpg", True)
