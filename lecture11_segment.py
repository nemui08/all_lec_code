# -*- coding: utf-8 -*-
"""บท 11 แบ่งภาพเป็นบริเวณ

Otsu ตัดตามความสว่าง, K-means จัดกลุ่มสี
ของที่ติดกันใช้ distance transform แล้ว watershed
"""
import cv2
import numpy as np
from common import save


def neighbors_demo():
    # มุมชนเฉย ๆ : 4 เพื่อนได้ 2 ก้อน, 8 เพื่อนได้ 1 ก้อน
    img = np.zeros((3, 3), np.uint8)
    img[0, 1] = img[1, 0] = img[1, 1] = img[2, 2] = 255
    n4, lab4 = cv2.connectedComponents(img, connectivity=4)
    n8, lab8 = cv2.connectedComponents(img, connectivity=8)
    print("จำนวนป้ายรวมพื้น 4 เพื่อน =", n4, "8 เพื่อน =", n8)
    print(lab4)
    print(lab8)


def kmeans_colors(img, K=2):
    data = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    out = centers[labels.flatten()].reshape(img.shape).astype(np.uint8)
    return out


def split_objects(binary, color):
    # ระยะถึงขอบยิ่งตรงกลางยิ่งสูง ใช้เป็นเมล็ดแล้วเทน้ำแยกก้อนที่ติดกัน
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    _, sure = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure = sure.astype(np.uint8)
    unknown = cv2.subtract(binary, sure)
    n, markers = cv2.connectedComponents(sure)
    markers = markers + 1
    markers[unknown == 255] = 0
    sheet = color.copy()
    markers = cv2.watershed(sheet, markers.astype(np.int32))
    sheet[markers == -1] = (0, 0, 255)
    return sheet


def main():
    neighbors_demo()
    img = np.zeros((80, 80, 3), np.uint8)
    img[:40] = (200, 80, 40)
    img[40:] = (40, 160, 40)
    save("L11_kmeans.png", kmeans_colors(img, 2))

    binary = np.zeros((90, 90), np.uint8)
    cv2.circle(binary, (30, 45), 18, 255, -1)
    cv2.circle(binary, (58, 45), 18, 255, -1)
    color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    save("L11_watershed.png", split_objects(binary, color))

    gray = np.full((60, 60), 40, np.uint8)
    gray[15:45, 15:45] = 200
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    save("L11_otsu.png", otsu)
    print("บท 11 รันแล้ว")


if __name__ == "__main__":
    main()
