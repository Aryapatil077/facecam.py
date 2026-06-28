import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

color_maps = {
    ord('0'): None,
    ord('1'): cv2.COLORMAP_AUTUMN,
    ord('2'): cv2.COLORMAP_BONE,
    ord('3'): cv2.COLORMAP_JET,
    ord('4'): cv2.COLORMAP_WINTER,
    ord('5'): cv2.COLORMAP_RAINBOW,
    ord('6'): cv2.COLORMAP_OCEAN,
    ord('7'): cv2.COLORMAP_SUMMER,
    ord('8'): cv2.COLORMAP_PINK,
    ord('9'): cv2.COLORMAP_HOT
}

current_filter = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.1,
        5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            "Face Detected",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )

    display = frame.copy()

    if current_filter is not None:
        display = cv2.applyColorMap(display, current_filter)

    cv2.putText(
        display,
        "Press 0-9 | Q-Quit",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Face Detection + Color Map Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break

    if key in color_maps:
        current_filter = color_maps[key]

cap.release()
cv2.destroyAllWindows()