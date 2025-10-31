import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import os
import time
import copy
import matplotlib.pyplot as plt

# --- 1. Configuration ---

# !!! IMPORTANT: Update this path to your dataset's root directory
DATA_DIR = 'D:\\panel-defect' 

# The folder names inside 'train/' and 'val/' MUST match these exactly.
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
MODEL_SAVE_FILE = 'pv_defect_model_v2.pth'

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 50  # Increased for a larger dataset
LEARNING_RATE = 0.001
# ---------------------

def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=25):
    """
    Main function to train and validate the model.
    Returns the trained model and a history dictionary.
    """
    start_time = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Dictionary to store training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_FILE)
                print(f'New best model saved to {MODEL_SAVE_FILE} with accuracy: {best_acc:.4f}')
        print()

    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def plot_training_history(history):
    """
    Plots training/validation accuracy and loss and saves them as PNG files.
    """
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('accuracy_plot.png')
    print("Accuracy plot saved as 'accuracy_plot.png'")

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')
    print("Loss plot saved as 'loss_plot.png'")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. Data Preprocessing ---
    # Using standard ImageNet transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # --- 3. Load Datasets ---
    print("Loading datasets...")
    try:
        image_datasets = {
            x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
            for x in ['train', 'val']
        }
    except FileNotFoundError:
        print(f"Error: Data directory not found at {DATA_DIR}")
        print("Please update the 'DATA_DIR' variable at the top of the script.")
        return

    # --- 4. Create DataLoaders ---
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    # --- 5. Verify Class Names ---
    print("Classes found in folder:", image_datasets['train'].classes)
    if set(image_datasets['train'].classes) != set(CLASS_NAMES):
        print("\n--- WARNING! ---")
        print("The 'CLASS_NAMES' list in your script does not match the folders found in your dataset.")
        print("Script expected:", sorted(CLASS_NAMES))
        print("Found in folder:", sorted(image_datasets['train'].classes))
        print("Please update the 'CLASS_NAMES' list in the script to match your folders EXACTLY.")
        return

    # --- 6. Define the Model (Transfer Learning) ---
    print(f"Loading pre-trained ResNet-18 model for {NUM_CLASSES} classes...")
    model = models.resnet18(pretrained=True)
    
    # Replace the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model = model.to(device)

    # --- 7. Define Loss and Optimizer ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 8. Start Training ---
    print("Starting training...")
    trained_model, history = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=NUM_EPOCHS)
    
    print("\nTraining finished.")

    # --- 9. Plot History ---
    print("Generating plots...")
    plot_training_history(history)
    print("All tasks complete.")

if __name__ == '__main__':
    main()