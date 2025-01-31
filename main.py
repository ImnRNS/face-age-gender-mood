import cv2
import numpy as np
from deepface import DeepFace
import os

# Cek apakah file model ada sebelum diload
def check_model_files():
    model_files = [
        ('model/gender_deploy.prototxt', 'model/gender_net.caffemodel'),
        ('model/age_deploy.prototxt', 'model/age_net.caffemodel')
    ]
    for prototxt, model in model_files:
        if not os.path.exists(prototxt) or not os.path.exists(model):
            raise FileNotFoundError(f"File model {prototxt} atau {model} tidak ditemukan.")

check_model_files()

# Inisialisasi model gender dan usia
gender_net = cv2.dnn.readNetFromCaffe('model/gender_deploy.prototxt', 'model/gender_net.caffemodel')
age_net = cv2.dnn.readNetFromCaffe('model/age_deploy.prototxt', 'model/age_net.caffemodel')

# Daftar label usia dan gender
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Pria', 'Wanita']

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Gagal membuka webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari webcam.")
        break

    # Konversi ke RGB untuk DeepFace
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        results = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
    except Exception as e:
        print(f"Error dalam analisis wajah: {e}")
        continue

    for result in results:
        try:
            region = result.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            
            # Validasi koordinat wajah
            if w <= 0 or h <= 0:
                continue
            
            # Pastikan koordinat tidak keluar dari frame
            x, y = max(0, x), max(0, y)
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue

            # Pra-pemrosesan gambar
            blob = cv2.dnn.blobFromImage(face_roi, 1.0, (227, 227),
                                         (78.426, 87.769, 114.896), swapRB=False)

            # Prediksi gender
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[np.argmax(gender_preds[0])]

            # Prediksi usia
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[np.argmax(age_preds[0])]

            # Ambil emosi dominan
            emotion = result.get('dominant_emotion', 'Tidak diketahui')

            # Gambar kotak dan label hasil deteksi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            label = f"{gender}, {age}, {emotion}"
            cv2.putText(frame, label, (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        except Exception as e:
            print(f"Kesalahan dalam pemrosesan wajah: {e}")
            continue

    # Tampilkan hasil
    cv2.imshow('Deteksi Wajah', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
