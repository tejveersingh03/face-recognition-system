import cv2
import os
import sys
import numpy as np

# ---------------- USER ID → NAME MAPPING ----------------
names = {
    1: "Tejveer",
    2: "sagar",
    3: "tushar"
}

# Create directories
if not os.path.exists("dataset"):
    os.makedirs("dataset")
if not os.path.exists("trainer"):
    os.makedirs("trainer")

# ---------------- 1) COLLECT FACE DATA ----------------
def collect_faces(user_id):
    cam = cv2.VideoCapture(0)
    face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    print("[INFO] Look at the camera...")
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(
                f"dataset/User.{user_id}.{count}.jpg",
                gray[y:y+h, x:x+w]
            )
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Collecting Faces", img)

        if cv2.waitKey(1) & 0xFF == 27 or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] Dataset collection done")

# ---------------- 2) TRAIN MODEL ----------------
def train_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []

    for file in os.listdir("dataset"):
        if file.endswith(".jpg"):
            path = os.path.join("dataset", file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            id = int(file.split(".")[1])
            faces = detector.detectMultiScale(img)

            for (x, y, w, h) in faces:
                face_samples.append(img[y:y+h, x:x+w])
                ids.append(id)

    if len(face_samples) == 0:
        print("❌ No face data found. Collect dataset first.")
        return

    ids = np.array(ids, dtype=np.int32)   # 🔥 MAIN FIX

    recognizer.train(face_samples, ids)
    recognizer.write("trainer/trainer.yml")
    print("[INFO] Training completed")

# ---------------- 3) RECOGNIZE FACE ----------------
def recognize_face():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer/trainer.yml")

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 60:
                name = names.get(id, "Unknown")
            else:
                name = "Unknown"

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", img)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("python face.py collect <id>")
        print("python face.py train")
        print("python face.py recognize")
        sys.exit()

    if sys.argv[1] == "collect":
        collect_faces(int(sys.argv[2]))
    elif sys.argv[1] == "train":
        train_model()
    elif sys.argv[1] == "recognize":
        recognize_face()