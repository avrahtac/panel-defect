import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import tkinter.simpledialog as sd
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import cv2
import numpy as np

# =========================
# CONFIG
# =========================
# Update this path to where your YOLO classifier weights (.pt) live on the machine
MODEL_PATH = MODEL_PATH = r'D:\Projects\panel-defect\runs\classify\runs\detect\train_fast\weights\best.pt'

# =========================
# MAIN CLASS
# =========================
class PVDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PV Panel Defect Analyzer (YOLO Engine)")
        self.root.geometry("500x550")

        self.file_path = None
        self.model = self.load_model()

        # UI Layout
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

        self.result_label = Label(root, text="Prediction: ---", font=("Helvetica", 12, "bold"))
        self.result_label.pack(pady=2)

        self.confidence_label = Label(root, text="Confidence: ---", font=("Helvetica", 12))
        self.confidence_label.pack(pady=2)

    # =========================
    # LOAD YOLO MODEL
    # =========================
    def load_model(self):
        print("Loading YOLO Classifier model...")
        if not os.path.exists(MODEL_PATH):
            print(f"❌ YOLO Model weights not found at: {MODEL_PATH}")
            return None

        try:
            # Native, ultra-fast YOLO loading
            model = YOLO(MODEL_PATH)
            print("✅ YOLO Classifier loaded successfully")
            return model
        except Exception as e:
            print("❌ Error loading YOLO model:", e)
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
    # ANALYZE STATIC IMAGE
    # =========================
    def analyze_image(self):
        if not self.model or not self.file_path: return
        
        print(f"\n--- Analyzing Static Image: {self.file_path} ---")
        try:
            results = self.model.predict(self.file_path, verbose=False)
            
            if hasattr(results[0], 'probs') and results[0].probs is not None:
                # Wrapped in int() to prevent Tensor Parse Errors
                top_class_idx = int(results[0].probs.top1) 
                top_conf = float(results[0].probs.top1conf.item()) * 100
                label = results[0].names[top_class_idx]
                
                self.result_label.config(text=f"Prediction: {label}")
                self.confidence_label.config(text=f"Confidence: {top_conf:.2f}%")
                print(f"Success: {label} at {top_conf:.2f}%")
            else:
                print("Error: Model did not return probability metrics.")
                
        except Exception as e:
            print(f"CRITICAL PARSE ERROR in analyze_image: {e}")


    def detect_on_frame(self, frame):
        h, w, _ = frame.shape
        
        crop_top, crop_bottom = int(h * 0.15), int(h * 0.85)
        crop_left, crop_right = int(w * 0.15), int(w * 0.85)
        
        # Safety Check: Prevent parsing empty frames
        if crop_bottom <= crop_top or crop_right <= crop_left:
            return frame 
            
        ai_view = frame[crop_top:crop_bottom, crop_left:crop_right]

        try:
            results = self.model.predict(ai_view, verbose=False)

            # Masking the UI
            cv2.rectangle(frame, (0, 0), (w, crop_top), (30, 30, 30), -1) 
            cv2.rectangle(frame, (0, crop_bottom), (w, h), (30, 30, 30), -1) 
            cv2.rectangle(frame, (0, crop_top), (crop_left, crop_bottom), (30, 30, 30), -1) 
            cv2.rectangle(frame, (crop_right, crop_top), (w, crop_bottom), (30, 30, 30), -1) 
            
            cv2.rectangle(frame, (crop_left, crop_top), (crop_right, crop_bottom), (0, 255, 0), 2)

            if hasattr(results[0], 'probs') and results[0].probs is not None:
                # Wrapped in int() and float() to prevent Tensor Parse Errors
                top1_idx = int(results[0].probs.top1)
                top1_conf = float(results[0].probs.top1conf.item())
                top1_label = results[0].names[top1_idx]
                
                top5_confs = results[0].probs.top5conf.tolist()
                top2_conf = float(top5_confs[1]) if len(top5_confs) > 1 else 0.0

                confidence_delta = top1_conf - top2_conf

                if confidence_delta < 0.40 or top1_conf < 0.80:
                    current_status = "INTERFERENCE"
                    self.history_buffer.append(current_status)
                    cv2.rectangle(frame, (0, 0), (w, 40), (128, 128, 128), -1)
                    cv2.putText(frame, "IGNORING BACKGROUND", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                elif top1_label in ["Clean", "Good"]:
                    current_status = "CLEAN"
                    self.history_buffer.append(current_status)
                    cv2.rectangle(frame, (0, 0), (w, 40), (0, 153, 76), -1)
                    cv2.putText(frame, f"STATUS: {top1_label.upper()}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                else:
                    current_status = top1_label
                    self.history_buffer.append(current_status)

                    if self.history_buffer.count(current_status) == 5:
                        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 204), -1)
                        cv2.putText(frame, f"ALERT: {top1_label} ({top1_conf*100:.1f}%)", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        cv2.rectangle(frame, (0, 0), (w, 40), (0, 140, 255), -1)
                        cv2.putText(frame, "VERIFYING ANOMALY...", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        except Exception as e:
            # This will catch the exact parse error and print it to the terminal without crashing the camera stream
            print(f"CRITICAL PARSE ERROR in detect_on_frame: {e}")

        return frame

    def detect_on_frame(self, frame):
        h, w, _ = frame.shape
        
        # --- 1. DYNAMIC PANEL ISOLATION (OpenCV) ---
        # Convert to grayscale and detect edges to find the physical panel
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Find all geometric shapes on the camera
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Assume the largest shape is the solar panel
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, panel_w, panel_h = cv2.boundingRect(largest_contour)

            # Safety Check: Only trigger if the object is reasonably large
            if panel_w > 100 and panel_h > 100:
                
                # --- 2. AI VISION CROP ---
                # Extract ONLY the clean panel pixels for the AI to read
                ai_view = frame[y:y+panel_h, x:x+panel_w]

                try:
                    # Run YOLO purely on the cropped panel
                    results = self.model.predict(ai_view, verbose=False)

                    # --- 3. APPLY DUMMY COLOR BACKGROUND ---
                    # Create a solid Navy Blue background (OpenCV uses BGR format)
                    dummy_bg = np.full((h, w, 3), (50, 20, 20), dtype=np.uint8)

                    # Paste the real panel exactly back into the center of the dummy background
                    dummy_bg[y:y+panel_h, x:x+panel_w] = ai_view
                    frame = dummy_bg

                    # Draw a bright Cyan boundary exactly around the panel
                    cv2.rectangle(frame, (x, y), (x+panel_w, y+panel_h), (255, 255, 0), 3)

                    # --- 4. EXTRACT CONFIDENCE & LOGIC ---
                    if hasattr(results[0], 'probs') and results[0].probs is not None:
                        top1_idx = int(results[0].probs.top1)
                        top1_conf = float(results[0].probs.top1conf.item())
                        top1_label = results[0].names[top1_idx]
                        
                        top5_confs = results[0].probs.top5conf.tolist()
                        top2_conf = float(top5_confs[1]) if len(top5_confs) > 1 else 0.0

                        confidence_delta = top1_conf - top2_conf

                        # The HUD will display right above the panel bounding box
                        hud_y = max(y - 10, 30)

                        if confidence_delta < 0.40 or top1_conf < 0.80:
                            current_status = "INTERFERENCE"
                            self.history_buffer.append(current_status)
                            cv2.rectangle(frame, (x, hud_y-30), (x+panel_w, hud_y+5), (128, 128, 128), -1)
                            cv2.putText(frame, "IGNORING BACKGROUND", (x+5, hud_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        elif top1_label in ["Clean", "Good"]:
                            current_status = "CLEAN"
                            self.history_buffer.append(current_status)
                            cv2.rectangle(frame, (x, hud_y-30), (x+panel_w, hud_y+5), (0, 153, 76), -1)
                            cv2.putText(frame, f"STATUS: {top1_label.upper()}", (x+5, hud_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        else:
                            current_status = top1_label
                            self.history_buffer.append(current_status)

                            if self.history_buffer.count(current_status) == 5:
                                cv2.rectangle(frame, (x, hud_y-30), (x+panel_w, hud_y+5), (0, 0, 204), -1)
                                cv2.putText(frame, f"ALERT: {top1_label} ({top1_conf*100:.1f}%)", (x+5, hud_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            else:
                                cv2.rectangle(frame, (x, hud_y-30), (x+panel_w, hud_y+5), (0, 140, 255), -1)
                                cv2.putText(frame, "VERIFYING ANOMALY...", (x+5, hud_y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                except Exception as e:
                    print(f"CRITICAL PARSE ERROR in detect_on_frame: {e}")

        # If no panel is found on screen, it just returns the normal camera view
        return frame
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
    # LIVE FEED CONTROL
    # =========================
    def start_camera(self):
        if not self.model:
            print("Model not loaded")
            return

        print("Starting camera capture sequence...")
        choice = self.choose_camera_type()
        cap = None

        if choice == "1":
            print("Using USB Camera")
            cap = cv2.VideoCapture(0)
        elif choice == "2":
            print("Using Pi Camera (GStreamer Pipeline)")
            cap = cv2.VideoCapture(
                "libcamerasrc ! video/x-raw,width=640,height=480,framerate=30/1 ! videoconvert ! appsink",
                cv2.CAP_GSTREAMER
            )
        elif choice == "3":
            url = sd.askstring("RTSP URL", "Enter RTSP Stream URL:")
            if url:
                cap = cv2.VideoCapture(url)
        else:
            print("Invalid or cancelled choice, fallback to default local camera...")
            cap = cv2.VideoCapture(0)

        if not cap or not cap.isOpened():
            print("❌ Target video device could not be opened")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Execute fast single-pass prediction
            frame = self.detect_on_frame(frame)

            cv2.imshow("Live YOLO Classification", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to safely exit live preview
                break

        cap.release()
        cv2.destroyAllWindows()

    # =========================
    # HIGH-EFFICIENCY HUD PIPELINE
    # =========================
    def detect_on_frame(self, frame):
        h, w, _ = frame.shape
        
        # Isolate the core panel region (crops out surrounding sky/ground clutter)
        crop_top, crop_bottom = int(h * 0.1), int(h * 0.9)
        crop_left, crop_right = int(w * 0.1), int(w * 0.9)
        frame_crop = frame[crop_top:crop_bottom, crop_left:crop_right]

        # Single-pass YOLO inference on the region (No more sliding window bottleneck)
        results = self.model.predict(frame_crop, conf=0.25, verbose=False)

        if hasattr(results[0], 'probs') and results[0].probs is not None:
            top_class_idx = results[0].probs.top1
            top_conf = results[0].probs.top1conf.item()
            label = results[0].names[top_class_idx]

            # Re-draw localized crop boundaries for visual feedback
            cv2.rectangle(frame, (crop_left, crop_top), (crop_right, crop_bottom), (255, 255, 255), 1)

            # Conditional HUD status rendering matching drone.py
            if label not in ["Clean", "Good"] and top_conf > 0.85:
                # Issue prominent header banner for anomalies
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 204), -1)
                cv2.putText(frame, f"ALERT: {label} ({top_conf*100:.1f}%)", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                # Nominal structural status layout
                cv2.rectangle(frame, (0, 0), (w, 40), (0, 153, 76), -1)
                cv2.putText(frame, f"SYSTEM STATUS: {label.upper()}", (15, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            # Secondary fallback rendering
            frame = results[0].plot()

        return frame

# =========================
# MAIN EXECUTION ENTRY
# =========================
if __name__ == "__main__":
    print("Initializing PV Analyzer Desktop Subsystem...")
    root = tk.Tk()
    app = PVDefectApp(root)
    root.mainloop()