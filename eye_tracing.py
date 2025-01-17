import cv2
import face_recognition
import os
import numpy as np

# Yüz veri tabanını yükleme
def load_known_faces():
    known_encodings = []
    known_names = []

    for user_name in os.listdir("users"):
        user_folder = os.path.join("users", user_name)
        for image_file in os.listdir(user_folder):
            image_path = os.path.join(user_folder, image_file)
            image = face_recognition.load_image_file(image_path)
            try:
                encoding = face_recognition.face_encodings(image)[0]
                known_encodings.append(encoding)
                known_names.append(user_name)
            except IndexError:
                print(f"Yüz tespit edilemedi: {image_file}")

    return known_encodings, known_names

def detect_and_track_eyes():
    known_encodings, known_names = load_known_faces()

    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)
    print("Kamera başlatıldı. Yüz ve göz tanıma işlemi başlıyor...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kameradan görüntü alınamıyor.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Bilinmiyor"

            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            top, right, bottom, left = face_location

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            roi_gray = gray_frame[top:bottom, left:right]
            roi_color = frame[top:bottom, left:right]

            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes:
                eye_left = left + ex
                eye_top = top + ey
                eye_right = eye_left + ew
                eye_bottom = eye_top + eh

                if (eye_left < 0 or eye_top < 0 or eye_right > frame.shape[1] or eye_bottom > frame.shape[0]):
                    print(f"UYARI: {name} kullanıcısının gözleri ekran dışına çıkıyor!")

        
                cv2.rectangle(frame, (eye_left, eye_top), (eye_right, eye_bottom), (255, 0, 0), 2)
                cv2.putText(frame, "Goz", (eye_left, eye_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        cv2.imshow("Yüz ve Göz Tanıma", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Program sonlandırıldı.")


if __name__ == "__main__":
    detect_and_track_eyes()
