import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import tkinter.simpledialog as sd
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import os
import numpy as np
import cv2

# =========================
# CONFIG
# =========================
MODEL_PATH = 'D:\\Projects\\panel-defect\\pv_defect_model_v2.pth'

CLASS_NAMES = [
    'Bird-drop', 
    'Clean', 
    'Dusty', 
    'Electrical-damage', 
    'Physical-Damage', 
    'Snow-Covered',
    'Panel-Hail',
    'poop',
    'Good',
    'Shattering',
]

# =========================
# MAIN CLASS
# =========================
class PVDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PV Panel Defect Analyzer")
        self.root.geometry("500x550")

        self.file_path = None
        self.model = self.load_model()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # UI
        Label(root, text="PV Panel Defect Analyzer",
              font=("Helvetica", 18, "bold")).pack(pady=10)

        self.image_frame = Frame(root, width=300, height=300, bd=2, relief="sunken")
        self.image_frame.pack(pady=10)

        self.image_label = Label(self.image_frame, text="Upload Image")
        self.image_label.pack()

        button_frame = Frame(root)
        button_frame.pack(pady=10)

        Button(button_frame, text="Upload Image",
               command=self.open_image).grid(row=0, column=0, padx=10)

        self.analyze_btn = Button(button_frame, text="Analyze",
                                 state="disabled",
                                 command=self.analyze_image)
        self.analyze_btn.grid(row=0, column=1, padx=10)

        Button(button_frame, text="Start Camera",
               command=self.start_camera).grid(row=0, column=2, padx=10)

        self.result_label = Label(root, text="Prediction: ---")
        self.result_label.pack()

        self.confidence_label = Label(root, text="Confidence: ---")
        self.confidence_label.pack()

    # =========================
    # LOAD MODEL
    # =========================
    def load_model(self):
        print("Loading model...")

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            return None

        try:
            model = models.resnet18(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

            model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
            model.eval()

            print("✅ Model loaded successfully")
            return model

        except Exception as e:
            print("❌ Error loading model:", e)
            return None

    # =========================
    # IMAGE UPLOAD
    # =========================
    def open_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
        )

        if not self.file_path:
            return

        img = Image.open(self.file_path)
        img.thumbnail((300, 300))

        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image, text="")

        self.analyze_btn.config(state="normal")

    # =========================
    # ANALYZE IMAGE
    # =========================
    def analyze_image(self):
        if not self.model:
            print("Model not loaded")
            return

        image = Image.open(self.file_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        label = CLASS_NAMES[pred.item()]
        confidence = conf.item() * 100

        self.result_label.config(text=f"Prediction: {label}")
        self.confidence_label.config(text=f"Confidence: {confidence:.2f}%")

    # =========================
    # CAMERA TYPE SELECTION
    # =========================
    def choose_camera_type(self):
        choice = sd.askstring(
            "Camera Type",
            "Enter camera type:\n1 = USB Webcam\n2 = Raspberry Pi Camera\n3 = RTSP Stream"
        )
        return choice

    # =========================
    # CAMERA
    # =========================
    def start_camera(self):
        if not self.model:
            print("Model not loaded")
            return

        print("Starting camera...")

        choice = self.choose_camera_type()
        cap = None

        if choice == "1":
            print("Using USB Camera")
            cap = cv2.VideoCapture(0)

        elif choice == "2":
            print("Using Pi Camera")
            cap = cv2.VideoCapture(
                "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink",
                cv2.CAP_GSTREAMER
            )

        elif choice == "3":
            url = input("Enter RTSP URL: ")
            cap = cv2.VideoCapture(url)

        else:
            print("Invalid choice, trying default camera...")
            cap = cv2.VideoCapture(0)

        if not cap or not cap.isOpened():
            print("❌ Camera not working")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.detect_on_frame(frame)

            cv2.imshow("Live Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break

        cap.release()
        cv2.destroyAllWindows()

    # =========================
    # DETECTION WITH NMS, STEP, PANEL CROP
    # =========================
    def detect_on_frame(self, frame):
        h, w, _ = frame.shape

        # =========================
        # Step 4: Panel crop for testing (optional)
        # =========================
        crop_top = int(h * 0.1)
        crop_bottom = int(h * 0.9)
        crop_left = int(w * 0.1)
        crop_right = int(w * 0.9)
        frame_crop = frame[crop_top:crop_bottom, crop_left:crop_right]

        # =========================
        # Step 3: Sliding window
        # =========================
        step = 180
        size = 224

        boxes = []
        scores = []

        fh, fw, _ = frame_crop.shape

        for y in range(0, fh - size, step):
            for x in range(0, fw - size, step):
                patch = frame_crop[y:y+size, x:x+size]

                image = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                tensor = self.transform(image).unsqueeze(0)

                with torch.no_grad():
                    outputs = self.model(tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)

                label = CLASS_NAMES[pred.item()]
                confidence = conf.item()

                # =========================
                # Step 2: Confidence + filtering
                # =========================
                if confidence < 0.95:
                    continue
                if label in ["Clean", "Good"]:
                    continue

                # Adjust box coordinates for cropped frame
                x1 = x + crop_left
                y1 = y + crop_top
                x2 = x1 + size
                y2 = y1 + size

                boxes.append([x1, y1, x2, y2])
                scores.append(float(confidence))

        # =========================
        # Non-Max Suppression
        # =========================
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.95, nms_threshold=0.3)
            for i in indices:
                i = i[0] if isinstance(i, (tuple, list, np.ndarray)) else i
                x1, y1, x2, y2 = boxes[i]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{label} {scores[i]:.2f}",
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2)

        return frame


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("Starting application...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found at: {MODEL_PATH}")
        input("Press Enter to exit...")
    else:
        root = tk.Tk()
        app = PVDefectApp(root)
        root.mainloop()