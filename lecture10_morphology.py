# -*- coding: utf-8 -*-
"""บท 10 พอง กัด เปิด ปิด ก้อนขาว

dilate พองสีขาว, erode แฟบสีขาว
opening ลบจุดเศษ, closing อุดรู
นอกภาพในตัวอย่างนี้นับเป็นสีดำ
"""
import cv2
import numpy as np
from common import save


def show(name, img):
    # ตาราง 0/1 ขยายเป็นภาพให้เปิดดูได้
    view = (img.astype(np.uint8) * 255)
    save(name, cv2.resize(view, (180, 180), interpolation=cv2.INTER_NEAREST))
    print(name)
    print(img)


def main():
    k = np.ones((3, 3), np.uint8)
    border = dict(borderType=cv2.BORDER_CONSTANT, borderValue=0)

    block = np.zeros((6, 6), np.uint8)
    block[1:5, 1:5] = 1
    show("L10_block.png", block)
    show("L10_dilate.png", cv2.dilate(block, k, **border))
    show("L10_erode.png", cv2.erode(block, k, **border))

    speck = np.zeros((7, 7), np.uint8)
    speck[1:6, 1:6] = 1
    speck[0, 6] = 1
    opened = cv2.morphologyEx(speck, cv2.MORPH_OPEN, k, **border)
    show("L10_open.png", opened)

    hole = speck.copy()
    hole[0, 6] = 0
    hole[3, 3] = 0
    closed = cv2.morphologyEx(hole, cv2.MORPH_CLOSE, k, **border)
    show("L10_close.png", closed)

    edge = cv2.morphologyEx((block * 255), cv2.MORPH_GRADIENT, k, **border)
    show("L10_edge.png", (edge > 0).astype(np.uint8))

    # hit-or-miss หาจุดขาวที่เพื่อนรอบตัวเป็นดำ
    dots = np.zeros((5, 5), np.uint8)
    dots[2, 2] = 255
    kernel = np.array([[-1, -1, -1], [-1, 1, -1], [-1, -1, -1]], np.int8)
    hit = cv2.morphologyEx(dots, cv2.MORPH_HITMISS, kernel)
    print("hit-or-miss จุดกลาง =", int(hit[2, 2]))


if __name__ == "__main__":
    main()
