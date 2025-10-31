import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import os

# --- 1. Configuration ---

# Path to your saved model
MODEL_PATH = 'pv_defect_model_v2.pth'

# !!! IMPORTANT: This list MUST be identical to the one in your train.py script
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

NUM_CLASSES = len(CLASS_NAMES)
# ---------------------

class PVDefectApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PV Panel Defect Analyzer")
        self.root.geometry("500x550")

        self.file_path = None
        self.model = self.load_model()
        
        # Define the image transforms (must match validation transforms)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # --- Create GUI Widgets ---
        
        # Title
        title_label = Label(root, text="PV Panel Defect Analyzer", font=("Helvetica", 18, "bold"), pady=10)
        title_label.pack()

        # Frame for Image
        self.image_frame = Frame(root, width=300, height=300, relief="sunken", bd=2)
        self.image_frame.pack(padx=10, pady=10)
        self.image_label = Label(self.image_frame, text="Upload an image to analyze", font=("Helvetica", 12))
        self.image_label.pack(expand=True)

        # Frame for Buttons
        button_frame = Frame(root)
        button_frame.pack(pady=10)

        upload_btn = Button(button_frame, text="Upload Image", font=("Helvetica", 12), command=self.open_image)
        upload_btn.grid(row=0, column=0, padx=10)

        self.analyze_btn = Button(button_frame, text="Analyze", font=("Helvetica", 12, "bold"), state="disabled", command=self.analyze_image)
        self.analyze_btn.grid(row=0, column=1, padx=10)

        # Frame for Result
        result_frame = Frame(root, relief="groove", bd=2)
        result_frame.pack(fill="x", padx=10, pady=10)
        
        self.result_label = Label(result_frame, text="Prediction: ---", font=("Helvetica", 14))
        self.result_label.pack(pady=10)
        
        self.confidence_label = Label(result_frame, text="Confidence: ---", font=("Helvetica", 12))
        self.confidence_label.pack(pady=(0, 10))

    def load_model(self):
        """Loads the pre-trained model."""
        print(f"Loading model from {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at {MODEL_PATH}")
            print("Please run train.py first to create the model file.")
            self.root.quit()
            return None
            
        try:
            # Re-create the model architecture
            model = models.resnet18(pretrained=False) # Don't need pretrained weights here
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

            # Load the saved state dict
            # We use map_location='cpu' so it works on any computer
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            
            # Set model to evaluation mode (CRITICAL!)
            model.eval()
            
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            self.root.quit()
            return None

    def open_image(self):
        """Opens a file dialog to select an image."""
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        
        if not self.file_path:
            return
            
        # Clear previous results
        self.result_label.config(text="Prediction: ---")
        self.confidence_label.config(text="Confidence: ---")

        # Open and display the image in the GUI
        img = Image.open(self.file_path)
        img.thumbnail((300, 300))  # Resize for display
        
        # Convert PIL image to Tkinter-compatible image
        self.tk_image = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.tk_image, text="")
        
        # Enable the "Analyze" button
        self.analyze_btn.config(state="normal")

    def analyze_image(self):
        """Analyzes the currently loaded image."""
        if not self.file_path or not self.model:
            return

        try:
            # 1. Load and transform the image
            image = Image.open(self.file_path).convert('RGB')
            image_tensor = self.transform(image)
            # Add a batch dimension (C, H, W) -> (B, C, H, W)
            image_tensor = image_tensor.unsqueeze(0) 

            # 2. Get prediction
            with torch.no_grad(): # Disable gradient calculation
                outputs = self.model(image_tensor)
                
                # Get probabilities
                probabilities = F.softmax(outputs, dim=1)
                
                # Get the top class and its confidence
                confidence, predicted_idx = torch.max(probabilities, 1)
                
                class_name = CLASS_NAMES[predicted_idx.item()]
                confidence_percent = confidence.item() * 100

            # 3. Update the GUI
            self.result_label.config(text=f"Prediction: {class_name}", fg="blue")
            self.confidence_label.config(text=f"Confidence: {confidence_percent:.2f}%", fg="black")

        except Exception as e:
            print(f"Error during analysis: {e}")
            self.result_label.config(text="Error analyzing image", fg="red")
            self.confidence_label.config(text="---")


if __name__ == "__main__":
    # Check if model file exists before launching app
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run the 'train.py' script first to train and save the model.")
    else:
        root = tk.Tk()
        app = PVDefectApp(root)
        root.mainloop()