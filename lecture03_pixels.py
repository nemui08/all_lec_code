# -*- coding: utf-8 -*-
"""บท 3 อ่านภาพ แก้ทีละจุด และสูตรมุมมองกล้อง"""
import cv2
import numpy as np
from common import save


def field_of_view(sensor_mm, focal_mm):
    # FOV = 2 * arctan(ขนาดเซนเซอร์ / (2 * ทางยาวโฟกัส)) ได้หน่วยเรเดียน แล้วแปลงเป็นองศา
    theta = 2 * np.arctan(sensor_mm / (2 * focal_mm))
    return np.degrees(theta)


def main():
    img = np.zeros((40, 60, 3), np.uint8)
    img[:, :] = (80, 40, 20)
    print("สูง กว้าง ช่องสี =", img.shape)
    print("ชนิดข้อมูล =", img.dtype)
    print("จุดแถว 10 คอลัมน์ 20 =", img[10, 20])

    # แก้จุดเดียวให้ขาว แถวมาก่อนคอลัมน์
    img[10, 20] = (255, 255, 255)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("ภาพเทาจุดเดิม =", int(gray[10, 20]))
    save("L03_gray.png", gray)

    # ตัวอย่างเลนส์โฟกัส 50 mm เซนเซอร์กว้าง 36 mm
    print("FOV องศา =", round(float(field_of_view(36, 50)), 1))


if __name__ == "__main__":
    main()
