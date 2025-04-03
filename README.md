# Plant Disease Detection Model Package

This package contains the trained model and necessary files for running plant disease detection on your own system.

## Package Contents

1. `best_model.pth` - The trained PyTorch model
2. `classes.txt` - List of plant diseases the model can detect
3. `inference.py` - Script for making predictions
4. `requirements.txt` - Required Python packages
5. `app.py` - Flask application for web interface (optional)

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Required packages listed in requirements.txt

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Using the Web Interface

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and go to `http://localhost:5000`
3. Upload an image or use your webcam to detect plant diseases

### Option 2: Using Python Script

```python
from inference import predict_disease
import cv2

# Load and process an image
image = cv2.imread('your_image.jpg')
disease, confidence, mask = predict_disease(image)
print(f"Detected Disease: {disease}")
print(f"Confidence: {confidence:.2f}%")
```

## Model Details

- Architecture: ResNet50 backbone with custom classification head
- Input size: 224x224 pixels
- Output: 20 classes (10 plants × 2 states: healthy/diseased)
- Training accuracy: ~95%

## Supported Plants and Diseases

The model can detect diseases in the following plants:
- Alstonia Scholaris (healthy/diseased)
- Arjun (healthy/diseased)
- Chinar (healthy/diseased)
- Gauva (healthy/diseased)
- Jamun (healthy/diseased)
- Jatropha (healthy/diseased)
- Lemon (healthy/diseased)
- Mango (healthy/diseased)
- Pomegranate (healthy/diseased)
- Pongamia Pinnata (healthy/diseased)

## Notes

- The model performs best with well-lit, clear images of plant leaves
- For optimal results, ensure the leaf is centered in the image
- The confidence score indicates the model's certainty in its prediction

## Support

For issues or questions, please contact [Your Contact Information] 