# -*- coding: utf-8 -*-
"""บท 4 ปรับความสว่างและคอนทราสต์ทีละจุด พร้อมฮิสโตแกรม

ปัญหาที่แก้: ภาพมืด ของในเงามองไม่เห็น หรือคอนทราสต์ต่ำ
สูตรหลัก: ค่าใหม่ = alpha * ค่าเก่า + beta
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, save


def apply_linear(img, alpha, beta):
    # คูณบนทศนิยม แล้วตัดให้อยู่ 0-255 ก่อนกลับเป็น uint8
    # alpha < 1 ลดคอนทราสต์, beta > 0 เพิ่มความสว่าง
    f = img.astype(np.float32)
    return np.clip(alpha * f + beta, 0, 255).astype(np.uint8)


def negative(gray):
    # กลับดำเป็นขาว สูตร 255 - f
    return cv2.bitwise_not(gray)


def gamma_correct(gray, gamma):
    # ยกกำลัง ต้องสเกลเป็น 0-1 ก่อน แล้วคูณกลับ 255
    # gamma < 1 ทำให้ส่วนมืดสว่างขึ้น
    f = gray.astype(np.float32) / 255.0
    return np.clip((f ** gamma) * 255, 0, 255).astype(np.uint8)


def log_transform(gray):
    # บีบค่าที่สูงมากให้เห็นรายละเอียดส่วนมืด ใช้ดูสเปกตรัมฟูเรียร์
    f = gray.astype(np.float32)
    c = 255 / np.log(1 + f.max())
    return np.clip(c * np.log(1 + f), 0, 255).astype(np.uint8)


def match_histogram(source, template):
    # ทำให้ฮิสโตแกรมของ source ใกล้ template
    src = source.ravel()
    tpl = template.ravel()
    src_cdf = np.cumsum(np.bincount(src, minlength=256)).astype(np.float64)
    tpl_cdf = np.cumsum(np.bincount(tpl, minlength=256)).astype(np.float64)
    src_cdf /= src_cdf[-1]
    tpl_cdf /= tpl_cdf[-1]
    lut = np.interp(src_cdf, tpl_cdf, np.arange(256))
    return lut[source].astype(np.uint8)


def save_hist(img, filename, title):
    # นับความสว่างแล้วเซฟกราฟ ภาพสีจะได้ 3 เส้น น้ำเงิน เขียว แดง
    plt.figure()
    if img.ndim == 2:
        h = cv2.calcHist([img], [0], None, [256], [0, 256])
        plt.plot(h, color="k")
    else:
        for i, c in enumerate("bgr"):
            h = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(h, color=c)
    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}")
    plt.close()


def main():
    dark = np.full((80, 120), 30, np.uint8)
    dark[20:50, 70:100] = 50
    bright = apply_linear(dark, alpha=0.6, beta=80)
    save("L04_bright.png", bright)
    save_hist(dark, "L04_hist_before.png", "before")
    save_hist(bright, "L04_hist_after.png", "after")

    save("L04_negative.png", negative(dark))
    save("L04_gamma.png", gamma_correct(dark, 0.5))
    save("L04_equalize.png", cv2.equalizeHist(dark))

    # equalizeHist รับได้แค่ภาพเทา 8 บิต
    ref = np.full_like(dark, 180)
    save("L04_match.png", match_histogram(dark, ref))
    print("บท 4 รันแล้ว ค่าจุดมืด 30 ->", int(bright[0, 0]))


if __name__ == "__main__":
    main()
