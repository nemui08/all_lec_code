# -*- coding: utf-8 -*-
"""บท 1 ภาพในคอมพิวเตอร์คือตารางตัวเลข

สไลด์ช่วง Stable Diffusion เป็นโปรเจกต์ ข้อสอบห้ามใช้โมเดล
ไฟล์นี้เลยมีแค่การดูภาพเป็นตาราง ซึ่งเป็นจุดเริ่มของทุกบทหลัง
"""
import cv2
import numpy as np
from common import save


def main():
    # สร้างภาพเทา 3x4 ด้วยมือ จะได้เห็นว่าแต่ละช่องคือเลข 0 ถึง 255
    gray = np.array([
        [0, 80, 160, 255],
        [10, 20, 30, 40],
        [255, 255, 0, 0],
    ], dtype=np.uint8)
    print("ภาพเทา shape =", gray.shape, "ชนิด =", gray.dtype)
    print(gray)

    # ภาพสีหนึ่งจุดมี 3 เลข เรียงน้ำเงิน เขียว แดง (BGR)
    color = np.zeros((2, 2, 3), np.uint8)
    color[0, 0] = (255, 0, 0)    # น้ำเงิน
    color[0, 1] = (0, 255, 0)    # เขียว
    color[1, 0] = (0, 0, 255)    # แดง
    color[1, 1] = (255, 255, 255)
    print("ภาพสี shape =", color.shape)
    save("L01_gray.png", gray)
    save("L01_color.png", color)


if __name__ == "__main__":
    main()
