# -*- coding: utf-8 -*-
"""บท 7 ย้อนภาพเบลอเมื่อรู้ PSF

เบลอ = คอนโวลูชันกับ PSF
inverse หารตรง ๆ แล้วพังตรงที่ H ใกล้ 0
Wiener ใส่ค่า K กันนอยส์ถูกขยาย
ช่วง Flask / SQLite ในสไลด์เป็นโปรเจกต์ ไม่ได้ใส่ในไฟล์นี้
"""
import cv2
import numpy as np
from common import save


def blur_with_psf(gray, psf):
    return cv2.filter2D(gray, cv2.CV_32F, psf, borderType=cv2.BORDER_REFLECT)


def wiener(blurred, psf, K=0.01):
    # H คือฟูเรียร์ของ PSF ต้องให้จุดกลางของ PSF ไปอยู่ที่มุม (0,0) ก่อนแปลง
    H = np.fft.fft2(np.fft.ifftshift(psf), s=blurred.shape)
    B = np.fft.fft2(blurred.astype(np.float32))
    G = np.conj(H) / (np.abs(H) ** 2 + K) * B
    out = np.fft.ifft2(G).real
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    sharp = np.full((100, 100), 30, np.uint8)
    cv2.rectangle(sharp, (20, 20), (70, 80), 220, -1)
    psf = np.zeros((100, 100), np.float32)
    psf[50, 47:54] = 1 / 7
    blurred = blur_with_psf(sharp, psf)
    restored = wiener(blurred, psf, K=0.002)
    save("L07_blur.png", np.clip(blurred, 0, 255).astype(np.uint8))
    save("L07_wiener.png", restored)
    # ภาพที่ย้อนแล้วควรใกล้ภาพคมกว่าภาพเบลอ
    err_blur = np.mean(np.abs(blurred - sharp.astype(np.float32)))
    err_fix = np.mean(np.abs(restored.astype(np.float32) - sharp))
    print("ความต่างจากภาพคม เบลอ =", round(float(err_blur), 1), "Wiener =", round(float(err_fix), 1))
    if err_fix >= err_blur:
        raise SystemExit("Wiener ไม่ดีขึ้น")


if __name__ == "__main__":
    main()
