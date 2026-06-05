import cv2
import mediapipe as mp
import pyautogui
import math
import time

# 1. Inisialisasi MediaPipe Tasks Baru
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

screen_width, screen_height = pyautogui.size()

# Pengaturan PyAutoGUI agar respons gerakan instan
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.8, # Ketat saat pertama kali mendeteksi tangan
    min_hand_presence_confidence=0.7   # Sedikit lebih longgar agar saat menjauh tidak mudah lepas
)

# Variabel Status untuk Gestur Pinch
is_pinching = False
pinch_start_time = 0
last_pinch_release_time = 0
click_ready = False

# Variabel Filter Smoothing EMA
prev_x, prev_y = 0, 0
SMOOTHING_FACTOR = 0.20  

# PARAMETER KALIBRASI
BASE_PINCH_THRESHOLD = 0.04    
RELEASE_PINCH_THRESHOLD = 0.06 
HOLD_DELAY = 0.4               
DOUBLE_CLICK_DELAY = 0.35    
MAX_JUMP_DISTANCE = 300        # Jarak piksel maksimum yang diizinkan dalam 1 frame (Anti-Lompat)

# Atur resolusi kamera ke 640x480
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Project Finger Tracker - Anti-Jump & Distance Protection Mode Aktif...")

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
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]

                # Target koordinat mentah dari kamera berdasarkan posisi Ibu Jari
                target_x = int(thumb_tip.x * screen_width)
                target_y = int(thumb_tip.y * screen_height)

                # Cek apakah gerakan terlalu ekstrem / melompat acak (Anti-Spike)
                if prev_x != 0 and prev_y != 0:
                    movement_distance = math.sqrt((target_x - prev_x)**2 + (target_y - prev_y)**2)
                    if movement_distance > MAX_JUMP_DISTANCE:
                        # Mengabaikan frame ini jika kursor melompat terlalu jauh secara instan
                        target_x, target_y = prev_x, prev_y

                # Penerapan Rumus Smoothing EMA
                if prev_x == 0 and prev_y == 0:
                    curr_x, curr_y = target_x, target_y
                else:
                    curr_x = int((target_x * SMOOTHING_FACTOR) + (prev_x * (1 - SMOOTHING_FACTOR)))
                    curr_y = int((target_y * SMOOTHING_FACTOR) + (prev_y * (1 - SMOOTHING_FACTOR)))

                # Pindahkan kursor ke koordinat yang sudah difilter
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # Hitung Jarak Jari secara 2D murni
                distance = math.sqrt(
                    (index_tip.x - thumb_tip.x)**2 + 
                    (index_tip.y - thumb_tip.y)**2
                )

                current_time = time.time()
                current_threshold = RELEASE_PINCH_THRESHOLD if is_pinching else BASE_PINCH_THRESHOLD

                # Logika Pinch (Klik & Drag)
                if distance < current_threshold:
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
                        print("🔓 HOLD RELEASED (Klik Dilepas)")
                        
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
                
                cv2.circle(frame, (tx, ty), 15, (255, 0, 255), cv2.FILLED) 
                cv2.circle(frame, (ix, iy), 15, (0, 255, 255), cv2.FILLED) 
                
                line_color = (0, 0, 255) if is_pinching else (255, 0, 0)
                cv2.line(frame, (ix, iy), (tx, ty), line_color, 3)
                
                status_text = "HOLD/DRAG" if (is_pinching and not click_ready) else ("PINCH" if is_pinching else "FREE")
                cv2.putText(frame, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Tangan benar-benar hilang dari kamera, reset koordinat dengan aman tanpa memaksa ada
            prev_x, prev_y = 0, 0

        cv2.imshow('Project Finger Tracker - Monitor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()