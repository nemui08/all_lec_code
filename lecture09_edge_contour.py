# -*- coding: utf-8 -*-
"""บท 9 ขอบ มุม และบรรยายรูปร่าง

บาร์โค้ดคือบริเวณที่ขอบหนาแน่นในทิศเดียว
contour ใช้เลือกก้อนแล้วระบายเป็นหน้ากาก
"""
import cv2
import numpy as np
from common import save


def edges(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.convertScaleAbs(cv2.magnitude(sx, sy))
    canny = cv2.Canny(blur, 50, 150)
    return mag, canny


def harris_corners(gray):
    # คะแนนมุมสูงตรงจุดที่เลื่อนกรอบไปทางไหนภาพก็เปลี่ยน
    response = cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out[response > 0.05 * response.max()] = (0, 0, 255)
    return out


def chain_code(contour):
    # เดินตามขอบ จำทิศ 0=ขวา แล้วเพิ่มตามเข็ม ทีละ 45 องศา จนถึง 7
    pts = contour.reshape(-1, 2)
    directions = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
                  (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7}
    code = []
    for (x0, y0), (x1, y1) in zip(pts, np.roll(pts, -1, axis=0)):
        step = (int(np.sign(x1 - x0)), int(np.sign(y1 - y0)))
        if step == (0, 0):
            continue
        code.append(directions[step])
    return code


def largest_mask(binary):
    # เลือกก้อนขาวที่พื้นที่มากสุด แล้วระบายทึบ ความหนา -1 คือถมข้างใน
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if not cnts:
        return mask, []
    best = max(cnts, key=cv2.contourArea)
    cv2.drawContours(mask, [best], -1, 255, -1)
    return mask, chain_code(best)


def main():
    gray = np.full((80, 80), 20, np.uint8)
    cv2.rectangle(gray, (15, 15), (60, 55), 220, -1)
    mag, canny = edges(gray)
    save("L09_sobel.png", mag)
    save("L09_canny.png", canny)
    save("L09_harris.png", harris_corners(gray))
    mask, code = largest_mask(cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1])
    save("L09_contour.png", mask)
    print("chain code ต้น ๆ =", code[:12])


if __name__ == "__main__":
    main()
