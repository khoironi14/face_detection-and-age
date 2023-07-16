import cv2
import dlib
#Impor library OpenCV dan dlib yang diperlukan untuk face detection dan estimasi usia

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#Inisialisasi detektor wajah dan prediktor landmark menggunakan model 

age_model = cv2.dnn.readNetFromCaffe(
    "age_gender_models/deploy_age.prototxt",
    "age_gender_models/age_net.caffemodel"
)
#Baca model estimasi usia yang dilatih sebelumnya menggunakan OpenCV
age_classes = ['0-2', '4-6', '8-12', '25-32', '38-43', '48-53', '60+']
#Definisikan kelas usia yang akan digunakan untuk memberi label pada hasil estimasi usia.
def detect_and_estimate_age(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
#Fungsi detect_and_estimate_age mengambil satu frame sebagai argumen dan mengubahnya menjadi grayscale.
#Kemudian, fungsi ini menggunakan detektor wajah untuk mendeteksi wajah dalam gambar grayscale.
    for face in faces:
        landmarks = predictor(gray, face)
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
#Loop ini akan berjalan melalui setiap wajah yang terdeteksi dalam gambar.
#Untuk setiap wajah, prediktor landmark akan digunakan untuk mendapatkan koordinat landmark wajah yang diperlukan untuk mengambil ROI wajah.
        face_roi = frame[y1:y2, x1:x2].copy()
        blob = cv2.dnn.blobFromImage(face_roi, scalefactor=1.0, size=(227, 227),
                                     mean=(78.4263377603, 87.7689143744, 114.895847746),
                                     swapRB=False, crop=False)
        #Ambil ROI wajah dari frame asli menggunakan koordinat yang didapatkan sebelumnya.
        #ROI ini akan digunakan sebagai input untuk model estimasi usia setelah pra-pemrosesan dengan membuat blob.

        age_model.setInput(blob)
        age_preds = age_model.forward()
        age_index = age_preds[0].argmax()
        age = age_classes[age_index]
#Atur input untuk model estimasi usia dan dapatkan prediksi usia. Prediksi ini merupakan distribusi probabilitas untuk setiap kelas usia.
#Kemudian, kelas usia dengan probabilitas tertinggi dipilih dan disimpan dalam variabel age.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, age, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#Gambar kotak di sekitar wajah dan tampilkan label usia pada frame asli.
    return frame

# Mengambil daftar kamera yang tersedia
def get_available_cameras():
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras

# Memilih kamera
def select_camera():
    cameras = get_available_cameras()
    num_cameras = len(cameras)
    if num_cameras == 0:
        print("Tidak ada kamera yang terdeteksi.")
        return None
    else:
        print("Kamera yang tersedia:")
        for i, camera in enumerate(cameras):
            print(f"{i+1}. Camera {camera}")
        while True:
            choice = input(f"Masukkan nomor kamera (1-{num_cameras}): ")
            if choice.isdigit() and 1 <= int(choice) <= num_cameras:
                camera_index = cameras[int(choice) - 1]
                return camera_index
            else:
                print("Pilihan tidak valid.")

# Memilih kamera
camera_index = select_camera()
if camera_index is None:
    exit()

# Menginisialisasi kamera
cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()

    frame = detect_and_estimate_age(frame)

    cv2.imshow('Age Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
