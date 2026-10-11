# -*- coding: utf-8 -*-
"""บท 6 โดเมนความถี่

กลางสเปกตรัมคือภาพรวมหยาบ ขอบสเปกตรัมคือรายละเอียด
low-pass เบลอ, high-pass เก็บขอบ, notch ลบลายซ้ำ
"""
import cv2
import numpy as np
from common import save


def spectrum(gray):
    # ย้ายความถี่ศูนย์มากลาง แล้วย่อด้วย log เพื่อให้ตามองเห็น
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.log(1 + np.abs(F))
    mag = mag / mag.max() * 255
    return F, mag.astype(np.uint8)


def apply_mask(gray, mask):
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    out = np.fft.ifft2(np.fft.ifftshift(F * mask)).real
    return np.clip(out, 0, 255).astype(np.uint8)


def circle_mask(shape, radius, highpass=False):
    h, w = shape
    y, x = np.ogrid[:h, :w]
    disk = (y - h // 2) ** 2 + (x - w // 2) ** 2 <= radius ** 2
    mask = disk.astype(np.float32)
    if highpass:
        mask = 1 - mask
    return mask


def notch_stripes(gray, offset=18, radius=4):
    # ปิดจุดสว่างคู่สมมาตรที่เกิดจากลายเส้นซ้ำ
    h, w = gray.shape
    mask = np.ones((h, w), np.float32)
    cv2.circle(mask, (w // 2 + offset, h // 2), radius, 0, -1)
    cv2.circle(mask, (w // 2 - offset, h // 2), radius, 0, -1)
    return apply_mask(gray, mask)


def main():
    img = np.full((120, 120), 140, np.uint8)
    cv2.rectangle(img, (30, 30), (90, 90), 40, -1)
    x = np.arange(120)
    stripes = np.clip(img.astype(np.float32) + 40 * np.sin(2 * np.pi * x / 18), 0, 255).astype(np.uint8)

    _, mag = spectrum(stripes)
    save("L06_stripes.png", stripes)
    save("L06_spectrum.png", mag)
    save("L06_lowpass.png", apply_mask(img, circle_mask(img.shape, 15, highpass=False)))
    save("L06_highpass.png", apply_mask(img, circle_mask(img.shape, 10, highpass=True)))
    save("L06_notch.png", notch_stripes(stripes, offset=7, radius=3))
    print("บท 6 รันแล้ว")


if __name__ == "__main__":
    main()
