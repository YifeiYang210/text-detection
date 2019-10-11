## Libraries
import argparse
import decimal
import os
import sys

import math
import numpy as np
from PIL import Image
from scipy.stats import mode, norm

import cv2
import matplotlib.pyplot as plt
import progressbar
import pytesseract
from aip import AipOcr


""" 你的 APPID AK SK """
APP_ID = '14975855'
API_KEY = 'jMHIB69opRk58UjZWGXdEZFb'
SECRET_KEY = 'Eh8KpbQD0SuN73G7FKruy2s4j6Xq12Vh'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)

## Arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, help="Path to the input image")
parser.add_argument("-s", "--dir", type=str, help="Path to the dir of image")
parser.add_argument("-o", "--output", type=str, help="Path to the output image")
parser.add_argument("-d", "--direction", default='both+', type=str, choices={"light", "dark", "both", "both+"},
                    help="Text searching")
parser.add_argument("-t", "--tesseract", action='store_true', help="Tesseract assistance")
parser.add_argument("-f", "--fulltesseract", action='store_true', help="Full Tesseract")
args = vars(parser.parse_args())
IMAGE_PATH = args["image"]
DIR_PATH = args["dir"]
OUTPUT_PATH = args["output"]
DIRECTION = args["direction"]
TESS = args["tesseract"]
FULL_OCR = args["fulltesseract"]

## Parameters
AREA_LIM = 2.0e-4
PERIMETER_LIM = 1e-4
ASPECT_RATIO_LIM = 5.0
OCCUPATION_LIM = (0.23, 0.90)
COMPACTNESS_LIM = (3e-3, 1e-1)
SWT_TOTAL_COUNT = 10
SWT_STD_LIM = 20.0
STROKE_WIDTH_SIZE_RATIO_LIM = 0.02  ## Min value
STROKE_WIDTH_VARIANCE_RATIO_LIM = 0.15  ## Min value
STEP_LIMIT = 10
KSIZE = 3
ITERATION = 7
MARGIN = 10


# Displaying function
def pltShow(*images):
    count = len(images)
    nRow = np.ceil(count / 3.)
    for i in range(count):
        plt.subplot(nRow, 3, i + 1)
        if len(images[i][0].shape) == 2:
            plt.imshow(images[i][0], cmap='gray')
        else:
            plt.imshow(images[i][0])
        plt.xticks([])
        plt.yticks([])
        plt.title(images[i][1])
    plt.show()


class TextDetection(object):

    def __init__(self, image_path):
        self.angle_list = []
        self.imagaPath = image_path
        img = cv2.imread(image_path)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img = rgbImg
        self.final = rgbImg.copy()
        self.height, self.width = self.img.shape[:2]
        self.grayImg = cv2.cvtColor(self.img.copy(), cv2.COLOR_RGB2GRAY)
        self.cannyImg = self.applyCanny(self.img)
        self.sobelX = cv2.Sobel(self.grayImg, cv2.CV_64F, 1, 0, ksize=-1)
        self.sobelY = cv2.Sobel(self.grayImg, cv2.CV_64F, 0, 1, ksize=-1)
        self.stepsX = self.sobelY.astype(int)  ## Steps are inversed!! (x-step -> sobelY)
        self.stepsY = self.sobelX.astype(int)
        self.magnitudes = np.sqrt(self.stepsX * self.stepsX + self.stepsY * self.stepsY)
        self.gradsX = self.stepsX / (self.magnitudes + 1e-10)
        self.gradsY = self.stepsY / (self.magnitudes + 1e-10)
        self.dir_name = ""
        self.pic_name = ""
        self.new_pic_path = ""
        self.dilated = None
        self.res = np.zeros_like(self.grayImg)
        self.dir_name = "/".join(self.imagaPath.split("/")[:-1])  # 'D:\\abpycharm\\text-detection/pic'
        self.pic_name = self.imagaPath.split("/")[-1:][0]  # '249.jpg'
        self.new_pic_path = self.dir_name.replace('pic', 'd') + "/00_" + self.pic_name
        # 'D:\\abpycharm\\text-detection/d/00_249.jpg'

    def getMSERegions(self, img):
        mser = cv2.MSER_create()
        # img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2GRAY)
        regions, bboxes = mser.detectRegions(img)
        return regions, bboxes

    def colorRegion(self, img, region):
        img[region[:, 1], region[:, 0], 0] = np.random.randint(low=100, high=256)
        img[region[:, 1], region[:, 0], 1] = np.random.randint(low=100, high=256)
        img[region[:, 1], region[:, 0], 2] = np.random.randint(low=100, high=256)
        return img

    def applyCanny(self, img, sigma=0.33):
        v = np.median(img)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        return cv2.Canny(img, lower, upper)

    def getRegionShape(self, region):
        return (max(region[:, 1]) - min(region[:, 1]), max(region[:, 0]) - min(region[:, 0]))

    def getRegionArea(self, region):
        return len(list(region))  ## Number of pixels

    def getRegionPerimeter(self, region):
        x, y, w, h = cv2.boundingRect(region)
        return len(np.where(self.cannyImg[y:y + h, x:x + w] != 0)[0])

    def getOccupyRate(self, region):
        return (1.0 * self.getRegionArea(region)) / (
                self.getRegionShape(region)[0] * self.getRegionShape(region)[1] + 1.0e-10)

    def getAspectRatio(self, region):
        return (1.0 * max(self.getRegionShape(region))) / (min(self.getRegionShape(region)) + 1e-4)

    def getCompactness(self, region):
        return (1.0 * self.getRegionArea(region)) / (1.0 * self.getRegionPerimeter(region) ** 2)

    def getSolidity(self, region):
        x, y, w, h = cv2.boundingRect(region)
        return (1.0 * self.getRegionArea(region)) / ((1.0 * w * h) + 1e-10)

    def getStrokeProperties(self, strokeWidths):
        if len(strokeWidths) == 0:
            return (0, 0, 0, 0, 0, 0)
        try:
            mostStrokeWidth = mode(strokeWidths, axis=None)[0][0]  ## Most probable stroke width is the most one
            mostStrokeWidthCount = mode(strokeWidths, axis=None)[1][0]  ## Most probable stroke width is the most one
        except IndexError:
            mostStrokeWidth = 0
            mostStrokeWidthCount = 0
        try:
            mean, std = norm.fit(strokeWidths)
            xMin, xMax = int(min(strokeWidths)), int(max(strokeWidths))
        except ValueError:
            mean, std, xMin, xMax = 0, 0, 0, 0
        return (mostStrokeWidth, mostStrokeWidthCount, mean, std, xMin, xMax)

    def getStrokes(self, xywh):
        # strokes = np.zeros(self.grayImg.shape)
        x, y, w, h = xywh
        strokeWidths = np.array([[np.Infinity, np.Infinity]])
        for i in range(y, y + h):
            for j in range(x, x + w):
                if self.cannyImg[i, j] != 0:
                    stepX = self.stepsX[i, j]
                    stepY = self.stepsY[i, j]
                    gradX = self.gradsX[i, j]
                    gradY = self.gradsY[i, j]

                    prevX, prevY, prevX_opp, prevY_opp, stepSize = i, j, i, j, 0

                    if DIRECTION == "light":
                        go, go_opp = True, False
                    elif DIRECTION == "dark":
                        go, go_opp = False, True
                    else:
                        go, go_opp = True, True

                    strokeWidth = np.Infinity
                    strokeWidth_opp = np.Infinity
                    while (go or go_opp) and (stepSize < STEP_LIMIT):
                        stepSize += 1

                        if go:
                            curX = np.int(np.floor(i + gradX * stepSize))
                            curY = np.int(np.floor(j + gradY * stepSize))
                            if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
                                go = False
                            if go and ((curX != prevX) or (curY != prevY)):
                                try:
                                    if self.cannyImg[curX, curY] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX, curY] + gradY * -self.gradsY[
                                            curX, curY]) < np.pi / 2.0:
                                            strokeWidth = int(np.sqrt((curX - i) ** 2 + (curY - j) ** 2))

                                            go = False
                                except IndexError:
                                    go = False

                                prevX = curX
                                prevY = curY

                        if go_opp:
                            curX_opp = np.int(np.floor(i - gradX * stepSize))
                            curY_opp = np.int(np.floor(j - gradY * stepSize))
                            if (curX_opp <= y or curY_opp <= x or curX_opp >= y + h or curY_opp >= x + w):
                                go_opp = False
                            if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                                try:
                                    if self.cannyImg[curX_opp, curY_opp] != 0:
                                        if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + gradY * -self.gradsY[
                                            curX_opp, curY_opp]) < np.pi / 2.0:
                                            strokeWidth_opp = int(np.sqrt((curX_opp - i) ** 2 + (curY_opp - j) ** 2))

                                            go_opp = False

                                except IndexError:
                                    go_opp = False

                                prevX_opp = curX_opp
                                prevY_opp = curY_opp

                    strokeWidths = np.append(strokeWidths, [(strokeWidth, strokeWidth_opp)], axis=0)

        strokeWidths_opp = np.delete(strokeWidths[:, 1], np.where(strokeWidths[:, 1] == np.Infinity))
        strokeWidths = np.delete(strokeWidths[:, 0], np.where(strokeWidths[:, 0] == np.Infinity))
        return strokeWidths, strokeWidths_opp

    def detect(self):
        res10 = np.zeros_like(self.img)
        boxRes = self.img.copy()

        regions, bboxes = self.getMSERegions(self.grayImg)

        n1 = len(regions)
        n2, n3, n4, n5, n6, n7, n8, n9, n10 = [0] * 9
        bar = progressbar.ProgressBar(maxval=n1, widgets=[progressbar.Bar(marker='=', left='[', right=']'), ' ',
                                                          progressbar.SimpleProgress()])

        bar.start()
        ## Coloring the regions
        for i, region in enumerate(regions):
            bar.update(i + 1)

            if self.getRegionArea(region) > self.grayImg.shape[0] * self.grayImg.shape[1] * AREA_LIM:
                n2 += 1

                if self.getRegionPerimeter(region) > 2 * (
                        self.grayImg.shape[0] + self.grayImg.shape[1]) * PERIMETER_LIM:
                    n3 += 1

                    if self.getAspectRatio(region) < ASPECT_RATIO_LIM:
                        n4 += 1

                        if (self.getOccupyRate(region) > OCCUPATION_LIM[0]) and (
                                self.getOccupyRate(region) < OCCUPATION_LIM[1]):
                            n5 += 1

                            if (self.getCompactness(region) > COMPACTNESS_LIM[0]) and (
                                    self.getCompactness(region) < COMPACTNESS_LIM[1]):
                                n6 += 1

                                # x, y, w, h = cv2.boundingRect(region)
                                x, y, w, h = bboxes[i]

                                # strokeWidths, strokeWidths_opp, strokes = self.getStrokes((x, y, w, h))
                                strokeWidths, strokeWidths_opp = self.getStrokes((x, y, w, h))
                                if DIRECTION != "both+":
                                    strokeWidths = np.append(strokeWidths, strokeWidths_opp, axis=0)
                                    strokeWidth, strokeWidthCount, mean, std, xMin, xMax = self.getStrokeProperties(
                                        strokeWidths)
                                else:
                                    strokeWidth, strokeWidthCount, mean, std, xMin, xMax = self.getStrokeProperties(
                                        strokeWidths)
                                    strokeWidth_opp, strokeWidthCount_opp, mean_opp, std_opp, xMin_opp, xMax_opp = self.getStrokeProperties(
                                        strokeWidths_opp)
                                    if strokeWidthCount_opp > strokeWidthCount:  ## Take the strokeWidths with max of counts strokeWidth (most probable one)
                                        strokeWidths = strokeWidths_opp
                                        strokeWidth = strokeWidth_opp
                                        strokeWidthCount = strokeWidthCount_opp
                                        mean = mean_opp
                                        std = std_opp
                                        xMin = xMin_opp
                                        xMax = xMax_opp

                                if len(strokeWidths) > SWT_TOTAL_COUNT:
                                    n7 += 1

                                    if std < SWT_STD_LIM:
                                        n8 += 1

                                        strokeWidthSizeRatio = strokeWidth / (1.0 * max(self.getRegionShape(region)))
                                        if strokeWidthSizeRatio > STROKE_WIDTH_SIZE_RATIO_LIM:
                                            n9 += 1

                                            strokeWidthVarianceRatio = (1.0 * strokeWidth) / (std ** std)
                                            if strokeWidthVarianceRatio > STROKE_WIDTH_VARIANCE_RATIO_LIM:
                                                n10 += 1
                                                res10 = self.colorRegion(res10, region)

        bar.finish()
        print("{} regions left.".format(n10))

        # 2.Binarize regions
        binarized = np.zeros_like(self.grayImg)
        rows, cols, color = np.where(res10 != [0, 0, 0])
        binarized[rows, cols] = 255

        # 3.Dilate regions and find contours
        kernel = np.zeros((KSIZE, KSIZE), dtype=np.uint8)
        kernel[(KSIZE // 2)] = 1
        self.dilated = cv2.dilate(binarized.copy(), kernel, iterations=ITERATION)

        # 4.get minArea rectangle
        self.get_minArea_angle(self.dilated, 200, 2000)
        print("minArea法旋转角度：", self.angle_list)

        # io1: display chosen minArea
        # cv2.imshow("res", res)

        # io2: write chosen minArea to disk
        print("新写入二值化文件的路径：", self.new_pic_path)
        cv2.imwrite(self.new_pic_path, self.res)

        # 5.cal skew angle needs to be rectified
        data = self.cal_skew_angle()
        ans_res = data["Estimated Angle"]
        print("minAreaMethod:", data)

        # 6.get rectified angle
        direction = self.baidu_ocr(self.imagaPath)
        angle_final = self.get_skew_angle(ans_res, direction)
        print("最终输出的角度是；", angle_final)

        # rotate if needed
        # self.skew_pic_minArea(self.img, ans_res)
        return self.res, angle_final

    def get_minArea_angle(self, img, lop, los):
        image, contours, hierarchies = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Error[2] image bad quality: not found swt contour")
            exit(0)
        for i, (contour, hierarchy) in enumerate(zip(contours, hierarchies[0])):
            perimeter = cv2.arcLength(contour, True)
            area = cv2.contourArea(contour)
            if perimeter <= lop and area <= los:
                continue
            if perimeter >= 500 and area >= 5000:
                continue
            if hierarchy[-1] == -1:

                if TESS:
                    # draw rect with margin
                    x, y, w, h = cv2.boundingRect(contour)
                    if (y - MARGIN > 0) and (y + h + MARGIN < self.height) and (x - MARGIN > 0) and (
                            x + w + MARGIN < self.width):
                        cv2.imwrite("text.jpg", self.final[y - MARGIN:y + h + MARGIN, x - MARGIN:x + w + MARGIN])
                    else:
                        cv2.imwrite("text.jpg", self.final[y:y + h, x:x + w])

                    ###################
                    ## Run tesseract ##
                    ###################
                    string = pytesseract.image_to_string(Image.open("text.jpg"))
                    if string is not u'':
                        rect = cv2.minAreaRect(contour)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)
                        cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
                        cv2.drawContours(self.res, [box], 0, 255, -1)
                    os.remove("text.jpg")

                else:
                    rect = cv2.minAreaRect(contour)
                    self.angle_list.append(rect[2])  # angle
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(self.final, [box], 0, (0, 255, 0), 2)
                    cv2.drawContours(self.res, [box], 0, 255, -1)

    def cal_skew_angle(self):
        # first eliminates errors that intentionally cause offset rotation
        if self.angle_list.count(0) / len(self.angle_list) >= 0.35 and self.angle_list.count(0) >= 15:
            print("0占比过高", self.angle_list.count(0) / len(self.angle_list), self.angle_list.count(0))
            print("最终输出的角度是；0")
            exit(0)
        angle_str_list = [self.get_cut_num(str(i), 1) for i in self.angle_list]
        angle_float_list = [float(i) for i in angle_str_list]
        angle_maxq = self.get_max_cnt_list(angle_float_list)
        if angle_maxq != 100.0:
            angle_quality_list = self.get_quality_list(self.angle_list, 2.5, angle_maxq)
            ans_res = self.cal_avg_nonzero(angle_quality_list)
            data = {"Estimated Angle": ans_res}
        else:
            self.get_minArea_angle(self.dilated, 50, 500)
            print("update_angle_list:", self.angle_list)
            data = self.cal_skew_angle()
        return data

    def fullOCR(self):
        bounded = self.img.copy()
        H, W = self.height, self.width
        res = np.zeros_like(self.grayImg)

        string = pytesseract.image_to_string(Image.open(self.imagaPath))
        if string == u'':
            return bounded, res

        boxes = pytesseract.image_to_boxes(Image.open(self.imagaPath))
        boxes = [list(map(int, i)) for i in [b.split(" ")[1:-1] for b in boxes.split("\n")]]

        for box in boxes:
            b = (int(box[0]), int(H - box[1]), int(box[2]), int(H - box[3]))
            cv2.rectangle(bounded, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.rectangle(res, (b[0], b[1]), (b[2], b[3]), 255, -1)

        # pltShow((img, "Original"), (bounded, "Boxes"), (res, "Mask"))
        return bounded, res

    def get_skew_angle(self, angle, direction):
        angle_final = 0
        angle_towards = math.fabs(angle)
        if 0 <= angle_towards <= 45:
            if direction == 0:
                angle_final = angle_towards
            elif direction == 1:
                angle_final = angle_towards + 90
            elif direction == 2:
                angle_final = angle_towards + 180
            elif direction == 3:
                angle_final = angle_towards + 270
        else:
            if direction == 1:
                angle_final = angle_towards
            elif direction == 2:
                angle_final = angle_towards + 90
            elif direction == 3:
                angle_final = angle_towards + 180
            elif direction == 0:
                angle_final = angle_towards + 270
        return angle_final

    def skew_pic_minArea(self, img, angle):
        rot_angle = (math.fabs(angle))
        rotated_0 = self.rotate_bound(img, rot_angle)
        plt.imshow(rotated_0)
        plt.show()

    def rotate_bound(self, image, angle):
        # 获取图像的尺寸
        # 旋转中心
        (h, w) = image.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        rectify_angle = 0
        # 设置旋转矩阵
        if 0 <= angle < 90:
            rectify_angle = angle
        elif 90 <= angle < 180:
            rectify_angle = angle - 180
        elif 180 <= angle < 270:
            rectify_angle = angle - 180
        elif 270 <= angle < 360:
            rectify_angle = angle - 90

        M = cv2.getRotationMatrix2D((cx, cy), -rectify_angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # 计算图像旋转后的新边界
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy

        return cv2.warpAffine(image, M, (nW, nH))

    def get_file_content(self, filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    def baidu_ocr(self, file_path):
        """
        文字角度识别
        """
        img = self.get_file_content(file_path)
        """ 如果有可选参数 """
        options = {}
        options["language_type"] = "CHN_ENG"
        options["detect_direction"] = "true"
        result = client.basicGeneral(img, options)
        direct = result['direction']
        print("文字方向:" + str(direct))
        return direct

    def get_cut_num(self, f_str, n):
        f_str = str(f_str)  # f_str = '{}'.format(f_str) 也可以转换为字符串
        a, b, c = f_str.partition('.')
        c = (c + "0" * n)[:n]  # 如论传入的函数有几位小数，在字符串后面都添加n为小数0
        return ".".join([a, c])

    def get_max_cnt_list(self, lt):
        temp = 0
        max_str = 100.0
        lt_str = [str(i) for i in lt]
        for i in lt_str:
            if float(i) == 0.0:
                continue
            i0 = decimal.Decimal(i)
            i1 = decimal.Decimal(i) + decimal.Decimal('0.1')
            i2 = decimal.Decimal(i) + decimal.Decimal('0.2')
            i3 = decimal.Decimal(i) - decimal.Decimal('0.1')
            i4 = decimal.Decimal(i) - decimal.Decimal('0.2')
            i5 = decimal.Decimal(i) + decimal.Decimal('0.3')
            i6 = decimal.Decimal(i) - decimal.Decimal('0.3')
            i_group_num = self.cal_cnt_from_strlist(lt_str, i0) + self.cal_cnt_from_strlist(lt_str, i1) \
                          + self.cal_cnt_from_strlist(lt_str, i2) + self.cal_cnt_from_strlist(lt_str, i3) \
                          + self.cal_cnt_from_strlist(lt_str, i4) + self.cal_cnt_from_strlist(lt_str, i5) \
                          + self.cal_cnt_from_strlist(lt_str, i6)
            if i_group_num >= temp:
                max_str = float(i)
                temp = i_group_num
        if temp == 1:
            max_str = 100.0
        return max_str

    def get_quality_list(self, lt, eps, max_angle):
        l = []
        for i in lt:
            if max_angle - eps <= i <= max_angle + eps:
                l.append(i)
        return l

    def cal_avg_nonzero(self, li):
        cnt = 0
        sum = 0
        for i in li:
            if i == 0:
                continue
            cnt += 1
            sum += i
        return sum / cnt

    def cal_cnt_from_strlist(self, li, key):
        cnt = 0
        for i in li:
            if key == decimal.Decimal(i):
                cnt += 1
        return cnt


if IMAGE_PATH:
    td = TextDetection(IMAGE_PATH)
    if FULL_OCR:
        bounded, res = td.fullOCR()
        pltShow((td.img, "Original"), (bounded, "Final"), (res, "Mask"))
    else:
        res, angle = td.detect()
        pltShow((td.img, "Original"), (td.final, "Final"), (res, "Mask"))
        if OUTPUT_PATH:
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_word = cv2.putText(td.final, 'angle = ' + str(angle), (50, 50), font, 1.2, (0, 0, 255), 2)
            plt.imsave(OUTPUT_PATH, img_word)
            print("{} saved".format(OUTPUT_PATH))

if DIR_PATH:
    pic_list = os.listdir(DIR_PATH)
    for i in pic_list:
        td = TextDetection(i)
        res, angle = td.detect()
        pltShow((td.img, "Original"), (td.final, "Final"), (res, "Mask"))
        if OUTPUT_PATH:
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_word = cv2.putText(td.final, 'angle = ' + str(angle), (50, 50), font, 1.2, (0, 0, 255), 2)
            plt.imsave(OUTPUT_PATH, img_word)
            print("{} saved".format(OUTPUT_PATH))
