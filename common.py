# -*- coding: utf-8 -*-
"""ที่เก็บไฟล์ผลลัพธ์ของโค้ดฝึก ทุกบทใช้ไฟล์นี้ร่วมกัน"""
import os
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)


def save(name, img):
    # เซฟรูปไว้ในโฟลเดอร์ output แล้วคืนพาธ
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    return path
