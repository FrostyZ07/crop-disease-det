import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# Model architecture (same as training)
class DiseaseDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseDetectionModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.segmentation_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
    
    def forward(self, x):
        classification = self.backbone(x)
        segmentation = self.segmentation_model(x)
        return classification, segmentation

# Load class names
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiseaseDetectionModel(len(class_names)).to(device)

# Load the best model weights
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

# Preprocessing transform
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def predict_disease(frame):
    # Preprocess the frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        classification, segmentation = model(image_tensor)
        probabilities = torch.softmax(classification, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        
        # Get segmentation mask
        mask = torch.sigmoid(segmentation).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        
    predicted_class = class_names[prediction.item()]
    confidence_score = confidence.item() * 100
    
    return predicted_class, confidence_score, mask

def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Make prediction
        predicted_class, confidence, mask = predict_disease(frame)
        
        # Create visualization
        # Original frame with prediction text
        cv2.putText(frame, f"Disease: {predicted_class}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Blend segmentation mask with original frame
        overlay = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Show results
        cv2.imshow('Plant Disease Detection', blended)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 