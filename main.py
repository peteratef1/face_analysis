import cv2
import dlib
import numpy as np
from imutils import face_utils
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import threading

# Load Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Load Dlib's face landmark predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

# Variables for PDF report
report_data = {}

# Initialize blink variables
blink_counter = 0
blink_threshold = 2  # Number of blinks to confirm a real face

# Global flag to indicate if the PDF has been created
pdf_created = False
pdf_generation_started = False

def detect_dark_spots(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dark_spots = [c for c in contours if cv2.contourArea(c) > 100]
    dark_spots_area = sum(cv2.contourArea(c) for c in dark_spots)

    for contour in dark_spots:
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return roi, dark_spots_area

def detect_wrinkles(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    wrinkle_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 50)

    for contour in contours:
        if cv2.contourArea(contour) > 50:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return roi, wrinkle_area

def detect_pores(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    _, thresh = cv2.threshold(np.abs(laplacian), 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pore_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) < 100 and cv2.contourArea(c) > 10)

    for contour in contours:
        if cv2.contourArea(contour) < 100 and cv2.contourArea(contour) > 10:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 255, 0), 2)

    return roi, pore_area

def detect_blink(landmarks):
    # Extract eye landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]

    # Calculate eye aspect ratio (EAR) for blink detection
    def eye_aspect_ratio(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    EAR_LEFT = eye_aspect_ratio(left_eye)
    EAR_RIGHT = eye_aspect_ratio(right_eye)

    return EAR_LEFT < 0.2 and EAR_RIGHT < 0.2

def generate_pdf_report(data):
    global pdf_created
    file_name = "Skin_Feature_Report.pdf"
    c = canvas.Canvas(file_name, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 50, "Face Skin Analysis Report")
    y_position = height - 100
    image_path = 'images.png'  # Update with the correct path to your image file
    c.drawImage(image_path, 200, height - 400, width=200, height=200)
    for feature, value in data.items():
        c.drawString(100, y_position, f"{feature}: {value:.2f}%")
        y_position -= 20

    c.save()
    print(f"PDF report generated: {file_name}")
    pdf_created = True  # Set the flag to indicate PDF creation

def analyze_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)
        (x, y, w, h) = face_utils.rect_to_bb(face)
        roi = frame[y:y + h, x:x + w]

        roi, dark_spots_area = detect_dark_spots(roi)
        roi, wrinkle_area = detect_wrinkles(roi)
        roi, pore_area = detect_pores(roi)

        total_face_area = w * h
        dark_spots_percentage = (dark_spots_area / total_face_area) * 100
        wrinkle_percentage = (wrinkle_area / total_face_area) * 100
        pore_percentage = (pore_area / total_face_area) * 100

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(frame, f"Dark Spots: {dark_spots_percentage:.2f}%", (x, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Wrinkles: {wrinkle_percentage:.2f}%", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"Pores: {pore_percentage:.2f}%", (x, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Store the data for the report
        global report_data
        report_data = {
            "Dark Spots": dark_spots_percentage,
            "Wrinkles": wrinkle_percentage,
            "Pores": pore_percentage
        }

        # Blink detection
        if detect_blink(landmarks):
            global blink_counter
            blink_counter += 1

        # Check if PDF generation has started or not
        global pdf_generation_started
        if not pdf_generation_started:
            # Wait for 15 seconds and generate the PDF report
            threading.Timer(7.0, generate_pdf_report, [report_data]).start()
            pdf_generation_started = True  # Set the flag to indicate PDF generation has started

while True:
    ret, frame = cap.read()

    if not ret:
        break

    analyze_frame(frame)

    # Display the frame
    cv2.imshow("Skin Feature Detection", frame)

    # Check if PDF has been created
    if pdf_created:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()