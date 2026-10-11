# -*- coding: utf-8 -*-
"""บท 5 ฟิลเตอร์เพื่อนบ้าน และเลือกวัตถุด้วยสี

จุดด่างใช้ median, นอยส์ฟุ้งใช้ Gaussian, แยกสีใช้ HSV
"""
import cv2
import numpy as np
from common import save


def slide_convolution():
    # ตัวอย่างในสไลด์ ภาพทุกแถวเป็น 1..7 หน้ากาก 3x3 ไม่ได้หารผลรวม
    image = np.tile(np.arange(1, 8), (7, 1)).astype(np.float32)
    kernel = np.array([[1, 2, 1], [2, 3, 2], [1, 2, 1]], np.float32)
    out = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    # กลางแถวที่ได้คือ 30 45 60 75 90
    got = out[1, 1:6].astype(int).tolist()
    print("คอนโวลูชันสไลด์ =", got)
    return got


def denoise_demo():
    rng = np.random.default_rng(0)
    clean = np.full((80, 80), 120, np.uint8)
    cv2.circle(clean, (40, 40), 20, 200, -1)
    sp = clean.copy()
    y = rng.integers(0, 80, 300)
    x = rng.integers(0, 80, 300)
    sp[y, x] = rng.integers(0, 2, 300) * 255
    save("L05_salt.png", sp)
    save("L05_median.png", cv2.medianBlur(sp, 3))
    save("L05_gaussian.png", cv2.GaussianBlur(sp, (5, 5), 0))
    save("L05_blur.png", cv2.blur(sp, (5, 5)))
    save("L05_bilateral.png", cv2.bilateralFilter(sp, 9, 75, 75))


def green_mask(img):
    # H ของ OpenCV อยู่ที่ 0-180 สีเขียวราว 40-85
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (40, 60, 40), (85, 255, 255))


def main():
    assert slide_convolution() == [30, 45, 60, 75, 90]
    denoise_demo()
    sign = np.full((60, 80, 3), 180, np.uint8)
    cv2.rectangle(sign, (15, 20), (65, 45), (40, 180, 40), -1)
    save("L05_green_mask.png", green_mask(sign))
    print("บท 5 รันแล้ว")


if __name__ == "__main__":
    main()
