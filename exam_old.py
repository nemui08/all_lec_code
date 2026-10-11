# -*- coding: utf-8 -*-
"""เฉลยข้อสอบเก่า 4 ข้อ

ข้อสอบจริงให้ใช้ไฟล์ภาพที่อาจารย์แนบ ห้ามครอปจาก PDF
ไฟล์นี้สร้างภาพตัวอย่างขึ้นมาเองเพื่อให้รันดูผลได้
เวลาสอบให้เปลี่ยนชื่อไฟล์ในตัวแปร INPUT ของแต่ละข้อ
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import OUT, save


def save_hist(img, filename):
    # นับสามช่องสีแล้วเซฟกราฟ ตามชื่อที่ข้อ 1 สั่ง
    plt.figure()
    for i, c in enumerate("bgr"):
        h = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(h, color=c)
    plt.xlabel("Intensity")
    plt.ylabel("Pixels")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUT}/{filename}")
    plt.close()


def q1(img=None):
    # ข้อ 1 ลดคอนทราสต์และเพิ่มความสว่าง ให้ของในท้องฟ้ามืดโผล่
    if img is None:
        img = np.full((120, 180, 3), 12, np.uint8)
        img[30:55, 120:155] = 28
        img[70:110, 20:80] = (230, 210, 190)
    alpha, beta = 0.5, 100
    out = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    save("Result1.png", out)
    save_hist(img, "Result1-Hist-Before.png")
    save_hist(out, "Result1-Hist-After.png")
    return img, out


def barcode_mask(img):
    # ข้อ 2 หาบาร์โค้ดจากขอบที่หนาแน่นในทิศตั้ง
    # ไม่จำพิกัดตายตัว เลยทนทั้งการเลื่อนและการหมุน 180 องศา
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.convertScaleAbs(cv2.absdiff(np.abs(gx), np.abs(gy)))
    grad = cv2.GaussianBlur(grad, (9, 9), 0)
    _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 9))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    bw = cv2.erode(bw, None, iterations=2)
    bw = cv2.dilate(bw, None, iterations=4)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, np.uint8)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:2]:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 150:
            continue
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask


def make_ticket(shift=0, turn180=False):
    img = np.full((180, 360, 3), 235, np.uint8)
    x = 40 + shift
    rng = np.random.default_rng(2)
    for i, w in enumerate(rng.integers(1, 4, size=36)):
        if i % 2 == 0:
            img[40:120, x:x + int(w)] = 0
        x += int(w)
    if turn180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def q2():
    a = make_ticket(0, False)
    b = make_ticket(25, True)
    m1 = barcode_mask(a)
    m2 = barcode_mask(b)
    save("Result2-1.png", m1)
    save("Result2-2.png", m2)
    return m1, m2


def green_mask(img):
    # ข้อ 3 ป้ายเขียว ทนการเลื่อนเพราะดูจากสี ไม่ได้จำตำแหน่ง
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 50, 40), (90, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros(mask.shape, np.uint8)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def make_sign(shift=0):
    img = np.full((160, 240, 3), (180, 160, 120), np.uint8)
    x = 30 + shift
    cv2.rectangle(img, (x, 50), (x + 140, 110), (40, 170, 50), -1)
    cv2.putText(img, "SIGN", (x + 20, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def q3():
    mask = green_mask(make_sign(15))
    save("Result3.png", mask)
    return mask


def q4(img=None):
    # ข้อ 4 จุดด่างสีสุ่ม ใช้ median จุดที่กระโดดจะหลุดตอนเลือกค่ากลาง
    if img is None:
        rng = np.random.default_rng(3)
        img = np.full((140, 140, 3), (180, 200, 210), np.uint8)
        cv2.circle(img, (70, 70), 40, (40, 40, 200), -1)
        n = 500
        y = rng.integers(0, 140, n)
        x = rng.integers(0, 140, n)
        img[y, x] = rng.integers(0, 2, (n, 1)) * 255
    out = cv2.medianBlur(img, 3)
    save("Result4.png", out)
    return out


def main():
    before, after = q1()
    sky_before = int(before[5, 5, 0])
    sky_after = int(after[5, 5, 0])
    print("ข้อ 1 ท้องฟ้า", sky_before, "->", sky_after)
    m1, m2 = q2()
    print("ข้อ 2 จุดขาว", int(np.count_nonzero(m1)), int(np.count_nonzero(m2)))
    if m1.max() == 0 or m2.max() == 0:
        raise SystemExit("ข้อ 2 หาบาร์โค้ดไม่เจอ")
    mask = q3()
    print("ข้อ 3 จุดขาว", int(np.count_nonzero(mask)))
    if mask.max() == 0:
        raise SystemExit("ข้อ 3 หาป้ายไม่เจอ")
    q4()
    print("ข้อ 4 เซฟ Result4.png แล้ว")
    print("ไฟล์อยู่ที่", OUT)


if __name__ == "__main__":
    main()
