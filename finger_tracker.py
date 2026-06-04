import cv2
import mediapipe as mp
import pyautogui

# 1. Inisialisasi MediaPipe Tasks Baru (Koreksi Struktur)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Mengambil resolusi layar laptop untuk memetakan kursor
screen_width, screen_height = pyautogui.size()

# Konfigurasi detektor tangan
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

# 2. Aktifkan Kamera
cap = cv2.VideoCapture(0)

print("Project Finger Tracker (Versi Baru) Aktif...")
print("Tekan tombol 'ESC' pada jendela kamera untuk keluar.")

# Membuka detektor MediaPipe
with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Gagal mengakses webcam.")
            break

        # Balikkan kamera (efek cermin) agar gerakan tangan sinkron
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Konversi gambar ke Format MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Dapatkan timestamp dalam milidetik (Wajib untuk mode VIDEO)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        
        # Proses deteksi koordinat tangan
        detection_result = landmarker.detect_for_video(mp_image, timestamp)

        # Jika ada tangan yang terdeteksi
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                # Titik nomor 8 adalah Ujung Jari Telunjuk (INDEX_FINGER_TIP)
                index_tip = hand_landmarks[8]

                # Konversi koordinat matriks (0.0 - 1.0) ke piksel layar asli kamu
                cursor_x = int(index_tip.x * screen_width)
                cursor_y = int(index_tip.y * screen_height)

                # Eksekusi: Gerakkan kursor mouse asli laptop
                pyautogui.moveTo(cursor_x, cursor_y)

                # Opsional: Gambar lingkaran hijau kecil di ujung jari pada jendela preview
                h, w, _ = frame.shape
                cx, cy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        # Tampilkan jendela visualisasi kamera di layar
        cv2.imshow('Project Finger Tracker - Monitor', frame)

        # Berhenti jika menekan tombol ESC
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()