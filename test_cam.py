import cv2

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Camera is NOT opening!")
else:
    print("Camera is opening...")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Frame not received!")
        break

    cv2.imshow("Test Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()