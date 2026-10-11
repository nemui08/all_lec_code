# -*- coding: utf-8 -*-
"""บท 12 ลายผิวด้วย GLCM

นับว่าค่าความสว่างคู่เพื่อนบ้านเกิดกี่ครั้ง
contrast สูงแปลว่าเพื่อนบ้านค่าห่างกันบ่อย ผิวหยาบ
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def glcm_right(img, levels):
    # นับเพื่อนทางขวา 1 ช่อง ช่อง (i, j) คือเจอค่า i แล้วขวาเป็น j
    g = np.zeros((levels, levels), np.int32)
    for r in range(img.shape[0]):
        for c in range(img.shape[1] - 1):
            g[img[r, c], img[r, c + 1]] += 1
    return g


def main():
    img = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [0, 2, 2, 2],
        [2, 2, 3, 3],
    ], np.uint8)
    manual = glcm_right(img, 4)
    lib = graycomatrix(img, distances=[1], angles=[0], levels=4, symmetric=False, normed=False)
    lib = lib[:, :, 0, 0]
    print("GLCM นับเอง")
    print(manual)
    print("ผลรวมคู่ =", int(manual.sum()))
    if not np.array_equal(manual, lib):
        raise SystemExit("GLCM ไม่ตรงกับ skimage")
    contrast = float(graycoprops(lib[:, :, None, None], "contrast")[0, 0])
    print("contrast =", round(contrast, 2))


if __name__ == "__main__":
    main()
