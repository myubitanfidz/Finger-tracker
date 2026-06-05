import cv2
import mediapipe as mp
import pyautogui
import math
import time

# 1. Inisialisasi MediaPipe Tasks
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0 

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# Variabel Status untuk Gestur Pinch
is_pinching = False
pinch_start_time = 0
last_pinch_release_time = 0
click_ready = False

# Parameter Kalibrasi
PINCH_THRESHOLD = 0.04       # Jarak minimal jempol-telunjuk dianggap menempel
HOLD_DELAY = 0.4             # Durasi menempel untuk dianggap "Klik Tahan" (detik)
DOUBLE_CLICK_DELAY = 0.35    # Jeda maksimal antar klik untuk "Double Click" (detik)

cap = cv2.VideoCapture(0)
print("Project Finger Tracker - Thumb Anchor Mode Aktif...")
print("Kursor sekarang dikendalikan oleh IBU JARI agar lebih stabil saat klik.")

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        detection_result = landmarker.detect_for_video(mp_image, timestamp)

        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Titik 4: Ujung Jempol, Titik 8: Ujung Telunjuk
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]

                # 1. GERAKKAN KURSOR BERDASARKAN IBU JARI (THUMB_TIP)
                cursor_x = int(thumb_tip.x * screen_width)
                cursor_y = int(thumb_tip.y * screen_height)
                pyautogui.moveTo(cursor_x, cursor_y)

                # 2. Hitung Jarak Jari untuk Deteksi Klik
                distance = math.sqrt(
                    (index_tip.x - thumb_tip.x)**2 + 
                    (index_tip.y - thumb_tip.y)**2 + 
                    (index_tip.z - thumb_tip.z)**2
                )

                current_time = time.time()

                # 3. Logika Pinch (Klik & Drag)
                if distance < PINCH_THRESHOLD:
                    if not is_pinching:
                        is_pinching = True
                        pinch_start_time = current_time
                        click_ready = True
                    else:
                        if click_ready and (current_time - pinch_start_time > HOLD_DELAY):
                            pyautogui.mouseDown()
                            print("🔒 MOUSE DRAG/HOLD (Klik Kiri Ditahan)")
                            click_ready = False
                else:
                    if is_pinching:
                        is_pinching = False
                        pyautogui.mouseUp()
                        
                        if click_ready:
                            if current_time - last_pinch_release_time < DOUBLE_CLICK_DELAY:
                                pyautogui.doubleClick()
                                print("💥 DOUBLE CLICK!")
                            else:
                                pyautogui.click()
                                print("👆 SINGLE CLICK!")
                                
                            last_pinch_release_time = current_time

                # Visualisasi Status di Monitor
                h, w, _ = frame.shape
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                
                # Gambar lingkaran pada ibu jari (sebagai pusat kursor)
                cv2.circle(frame, (tx, ty), 10, (255, 0, 255), cv2.FILLED)
                
                # Gambar garis indikator antar jari
                line_color = (0, 0, 255) if is_pinching else (255, 0, 0)
                cv2.line(frame, (ix, iy), (tx, ty), line_color, 2)
                
                status_text = "HOLD/DRAG" if (is_pinching and not click_ready) else ("PINCH" if is_pinching else "FREE")
                cv2.putText(frame, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Project Finger Tracker - Monitor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()