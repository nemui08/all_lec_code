# -*- coding: utf-8 -*-
"""บท 8 เวิร์กช็อป MediaPipe

ข้อสอบห้ามใช้โมเดล ไฟล์นี้ไว้ทำโปรเจกต์เท่านั้น
เครื่องนี้ยังไม่ได้ลง mediapipe โค้ดด้านล่างคือโครงที่ใช้ตอนลงแล้ว
ติดตั้งด้วย: pip install mediapipe
"""


def main():
    try:
        import mediapipe as mp
    except ImportError:
        print("ยังไม่มี mediapipe ในเครื่อง ข้ามการรัน โค้ดตัวอย่างอยู่ในฟังก์ชัน run_hands")
        return
    print("พบ mediapipe", mp.__version__)


def run_hands(image_bgr):
    # หาจุดบนมือ แล้ววาดลงภาพ ส่งภาพ BGR เข้ามา
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    rgb = image_bgr[:, :, ::-1]
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands:
        result = hands.process(rgb)
    out = image_bgr.copy()
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(out, hand, mp_hands.HAND_CONNECTIONS)
    return out


if __name__ == "__main__":
    main()
