import cv2
import numpy as np


def non_max_suppression_fast(boxes, overlapThresh):
    # 空数组检测
    if len(boxes) == 0:
        return []

        # 将类型转为float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    # 四个坐标数组
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)  # 计算面积数组
    idxs = np.argsort(y2)  # 返回的是右下角坐标从小到大的索引值

    # 开始遍历删除重复的框
    while len(idxs) > 0:
        # 将最右下方的框放入pick数组
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # 找到剩下的其余框中最大的坐标x1y1，和最小的坐标x2y2,
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # 计算重叠面积占对应框的比例
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]

        # 如果占比大于阈值，则删除
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def mser_text(file_path):
    img = cv2.imread(file_path)
    vis = img.copy()  # 用于绘制矩形框图
    orig = img.copy()  # 用于绘制不重叠的矩形框图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 得到灰度图
    mser = cv2.MSER_create()  # 得到mser算法对象
    regions, _ = mser.detectRegions(gray)  # 获取文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # 绘制文本区域
    cv2.polylines(img, hulls, 1, (255, 0, 0))
    cv2.namedWindow("img", 0)
    cv2.resizeWindow("img", 800, 640)  # 限定显示图像的大小
    cv2.imshow('img', img)

    keep = []
    # 绘制目前的矩形文本框
    for c in hulls:
        x, y, w, h = cv2.minAreaRect(c)
        keep.append([x, y, x + w, y + h])
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 1)
    print("[x] %d initial bounding boxes" % (len(keep)))
    cv2.namedWindow("hulls", 0)
    cv2.resizeWindow("hulls", 800, 640)
    cv2.imshow("hulls", vis)

    # 筛选不重复的矩形框
    keep2 = np.array(keep)
    pick = non_max_suppression_fast(keep2, 0.5)
    print("[x] after applying non-maximum, %d bounding boxes" % (len(pick)))
    for (startX, startY, endX, endY) in pick:
        cv2.rectangle(orig, (startX, startY), (endX, endY), (255, 185, 120), 2)
    cv2.namedWindow("After NMS", 0)
    cv2.resizeWindow("After NMS", 800, 640)
    cv2.imshow("After NMS", orig)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


mser_text("pic/1527143817_4_4072_aug_1552142205.21_3_rotate-0-0-318.jpg")

