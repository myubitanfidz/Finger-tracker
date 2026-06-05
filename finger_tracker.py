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

# OPTIMASI 1: Pengaturan PyAutoGUI agar respons gerakan instan
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.8, # Ketat agar tidak salah mendeteksi objek latar belakang
    min_hand_presence_confidence=0.8   # Tetap mengunci tangan dengan kuat
)

# Variabel Status untuk Gestur Pinch
is_pinching = False
pinch_start_time = 0
last_pinch_release_time = 0
click_ready = False

# OPTIMASI 2: Variabel Filter Smoothing EMA (Peredam Getaran Kursor)
prev_x, prev_y = 0, 0
SMOOTHING_FACTOR = 0.20  # Nilai diatur ke 0.20 agar pergerakan kursor sangat halus di layar

# Parameter Kalibrasi Jarak & Waktu
PINCH_THRESHOLD = 0.04       # Ambang batas jarak 2D jempol-telunjuk untuk klik
HOLD_DELAY = 0.4             # Waktu tahan untuk memicu Drag & Drop (detik)
DOUBLE_CLICK_DELAY = 0.35    # Jeda maksimal antar klik untuk Double Click (detik)

# OPTIMASI 3: Atur resolusi kamera ke 640x480 agar beban komputasi AI ringan dan FPS naik
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Project Finger Tracker - Optimized Version Aktif...")
print("Kursor stabil (Thumb Anchor) + Mulus (EMA Filter) + Akurat (Jarak 2D).")

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

                # Target koordinat mentah dari kamera berdasarkan posisi Ibu Jari
                target_x = int(thumb_tip.x * screen_width)
                target_y = int(thumb_tip.y * screen_height)

                # OPTIMASI 4: Penerapan Rumus Smoothing EMA
                if prev_x == 0 and prev_y == 0:
                    curr_x, curr_y = target_x, target_y
                else:
                    curr_x = int((target_x * SMOOTHING_FACTOR) + (prev_x * (1 - SMOOTHING_FACTOR)))
                    curr_y = int((target_y * SMOOTHING_FACTOR) + (prev_y * (1 - SMOOTHING_FACTOR)))

                # Pindahkan kursor ke koordinat yang sudah difilter
                pyautogui.moveTo(curr_x, curr_y)
                prev_x, prev_y = curr_x, curr_y

                # OPTIMASI 5: Hitung Jarak Jari secara 2D murni (Menghapus variabel .z)
                # Langkah ini meningkatkan akurasi deteksi cubitan secara signifikan
                distance = math.sqrt(
                    (index_tip.x - thumb_tip.x)**2 + 
                    (index_tip.y - thumb_tip.y)**2
                )

                current_time = time.time()

                # Logika Pinch (Klik & Drag)
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
                
                # Gambar lingkaran pada ibu jari (Pusat Kendali Kursor)
                cv2.circle(frame, (tx, ty), 10, (255, 0, 255), cv2.FILLED)
                
                # Gambar garis indikator transisi antar jari
                line_color = (0, 0, 255) if is_pinching else (255, 0, 0)
                cv2.line(frame, (ix, iy), (tx, ty), line_color, 2)
                
                status_text = "HOLD/DRAG" if (is_pinching and not click_ready) else ("PINCH" if is_pinching else "FREE")
                cv2.putText(frame, f"Status: {status_text}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Reset koordinat lama jika tangan sempat hilang dari frame kamera
            prev_x, prev_y = 0, 0

        cv2.imshow('Project Finger Tracker - Monitor', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()