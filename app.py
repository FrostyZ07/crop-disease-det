from flask import Flask, render_template, jsonify, request
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
#from train import DiseaseDetectionModel
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import logging
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import time
import json
import os
from groq import Groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
classes = None
transform = None

# Add Groq API configuration
GROQ_API_KEY = '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'
groq_client = Groq(api_key=GROQ_API_KEY)

def is_valid_class(class_name):
    # Filter out non-disease classes and special entries
    invalid_classes = {'PlantVillage', 'TRAIN', 'VAL', 'VALIDATION'}
    return class_name not in invalid_classes

def load_model_and_classes():
    global model, classes, transform
    try:
        # Load class names
        logger.info("Loading class names...")
        with open('classes.txt', 'r') as f:
            all_classes = [line.strip() for line in f.readlines()]
        
        classes = [c for c in all_classes if c and not c.startswith('__')]
        logger.info(f"Loaded {len(all_classes)} total classes, {len(classes)} valid classes")
        logger.info(f"Valid classes: {classes}")
        
        # Load model and classes
        logger.info("Loading model and classes...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize model with the same architecture as training
        logger.info("Initializing model...")
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_classes = len(classes)
        
        # Use the same architecture as in training
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        logger.info(f"Model architecture initialized with {num_classes} output classes")
        
        # Load the saved state dict
        logger.info("Loading saved model state...")
        state_dict = torch.load('best_model.pth', map_location=device)
        logger.info(f"Loaded state dict type: {type(state_dict)}")
        
        # Handle different state dict formats
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                logger.info("Found 'state_dict' key in loaded model")
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                logger.info("Found 'model_state_dict' key in loaded model")
                state_dict = state_dict['model_state_dict']
        
        # Create a new state dict with the correct keys
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove any module prefix
            if key.startswith('module.'):
                key = key[7:]
            
            if key.startswith('backbone.'):
                # Map backbone keys to base model
                new_key = key.replace('backbone.', '')
                if new_key.startswith('fc.'):
                    # Handle the classifier layers
                    if 'fc.0.' in new_key:  # First linear layer
                        new_key = new_key.replace('fc.0.', 'fc.0.')
                    elif 'fc.3.' in new_key:  # Second linear layer
                        new_key = new_key.replace('fc.3.', 'fc.3.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        logger.info(f"Processed state dict keys: {list(new_state_dict.keys())}")
        
        # Load the modified state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"Unexpected keys: {unexpected_keys}")
        
        model = model.to(device)
        model.eval()
        logger.info(f"Model loaded successfully and moved to {device}")
        
        # Initialize transform to match training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Transform pipeline initialized")
        
        # Test forward pass with sample data
        logger.info("Testing forward pass...")
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            try:
                output = model(dummy_input)
                probabilities = torch.softmax(output, dim=1)
                logger.info(f"Forward pass successful. Output shape: {output.shape}")
                logger.info(f"Sample probabilities: {probabilities[0][:5]}")
                
                # Test with a green image
                green_input = torch.zeros(1, 3, 224, 224).to(device)
                green_input[:, 1, :, :] = 1  # Set green channel to 1
                green_output = model(green_input)
                green_probs = torch.softmax(green_output, dim=1)
                logger.info(f"Green image test probabilities: {green_probs[0][:5]}")
            except Exception as e:
                logger.error(f"Forward pass failed: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model and classes: {str(e)}")
        return False

def is_plant(image):
    """Check if the image contains a plant based on green color percentage"""
    try:
        # Convert image to numpy array if it's not already
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Extract RGB channels
            r, g, b = image_array[:, :, 0], image_array[:, :, 1], image_array[:, :, 2]
            
            # Define green detection criteria
            green_mask = (g > 1.2 * r) & (g > 1.2 * b) & (g > 50)
            
            # Calculate percentage of green pixels
            green_percentage = (np.sum(green_mask) / (image_array.shape[0] * image_array.shape[1])) * 100
            logger.info(f"Green percentage in image: {green_percentage:.2f}%")
            
            return green_percentage > 1.0  # Lower threshold to 1%
        return True  # Default to true for grayscale images

    except Exception as e:
        logger.error(f"Error in is_plant: {str(e)}")
        return False

def get_plant_type(class_name):
    if class_name.startswith('Tomato'):
        return 'Tomato'
    elif class_name.startswith('Potato'):
        return 'Potato'
    elif class_name.startswith('Pepper'):
        return 'Pepper'
    return None

def process_frame(frame):
    if frame is None:
        logger.error("Received None frame")
        return None, None, None

    try:
        # Convert to PIL Image if needed
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(np.uint8(frame))
        
        logger.info(f"Frame size: {frame.size}, mode: {frame.mode}")
        
        # Check if the frame contains a plant
        if not is_plant(frame):
            logger.info("No plant detected in frame")
            return frame, None, None

        # Preprocess the frame
        image_tensor = transform(frame)
        logger.info(f"Tensor shape: {image_tensor.shape}, device: {image_tensor.device}")
        
        image_tensor = image_tensor.unsqueeze(0).to(device)
        logger.info(f"Batched tensor shape: {image_tensor.shape}")

        # Get model prediction
        with torch.no_grad():
            output = model(image_tensor)
            logger.info(f"Model output shape: {output.shape}")
        
        # Get predicted class
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence = confidence.item()
        predicted_class = predicted.item()
        
        # Log all class probabilities for debugging
        class_probs = probabilities[0].cpu().numpy()
        for i, prob in enumerate(class_probs):
            logger.info(f"Class {i} ({classes[i]}): {prob:.2%}")
        
        logger.info(f"Raw prediction: Class {predicted_class} ({classes[predicted_class]}), Confidence: {confidence:.2%}")

        # Lower confidence threshold for debugging
        if confidence < 0.30:  # Changed from 0.90 to 0.30 for testing
            logger.info(f"Confidence {confidence:.2%} below threshold 30%")
            return frame, None, None

        predicted_label = classes[predicted_class]
        logger.info(f"Predicted class: {predicted_label}")
        
        # Draw the prediction on the frame
        draw = ImageDraw.Draw(frame)
        text = f"{predicted_label}"
        conf_text = f"Confidence: {confidence:.2%}"
        
        # Add text with background
        draw.rectangle([(10, 10), (400, 80)], fill=(0, 0, 0))
        draw.text((15, 35), text, fill=(0, 255, 0))
        draw.text((15, 65), conf_text, fill=(0, 255, 0))

        return frame, predicted_label, confidence
        
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        return frame, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    try:
        # Initialize camera
        camera = initialize_camera()
        if camera is None:
            return "Camera initialization failed", 500

        def generate_frames(cap):
            frame_buffer = []
            consecutive_failures = 0
            last_frame_time = time.time()
            
            try:
                while True:
                    try:
                        # Limit frame rate to 30 FPS
                        current_time = time.time()
                        if current_time - last_frame_time < 1/30:
                            time.sleep(0.001)  # Small sleep to prevent CPU hogging
                            continue
                        
                        # Read frame
                        ret, frame = cap.read()
                        if not ret:
                            consecutive_failures += 1
                            logger.error(f"Failed to read frame (failure {consecutive_failures}/5)")
                            
                            if consecutive_failures >= 5:
                                logger.error("Too many consecutive failures, attempting to reconnect...")
                                cap.release()
                                time.sleep(1)  # Wait before reconnecting
                                new_cap = initialize_camera()
                                if new_cap is not None:
                                    cap = new_cap
                                consecutive_failures = 0
                                continue
                            
                            # Use buffered frame if available
                            if frame_buffer:
                                frame = frame_buffer[-1].copy()
                                logger.info("Using buffered frame")
                            else:
                                time.sleep(0.1)
                                continue
                        else:
                            consecutive_failures = 0
                            last_frame_time = current_time
                            
                            # Update frame buffer (maintain fixed size)
                            frame_buffer.append(frame.copy())
                            if len(frame_buffer) > 3:
                                frame_buffer.pop(0)
                        
                        # Process frame
                        processed_frame, predicted_class, confidence = process_frame(frame)
                        
                        # Encode frame
                        ret, buffer = cv2.imencode('.jpg', processed_frame)
                        if not ret:
                            continue
                        
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        
                    except Exception as e:
                        logger.error(f"Error in generate_frames loop: {str(e)}")
                        time.sleep(0.1)  # Prevent tight loop on error
                        continue
                    
            finally:
                # Cleanup
                if cap is not None:
                    cap.release()
                frame_buffer.clear()
        
        return Response(generate_frames(camera),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error in video_feed: {str(e)}")
        return str(e), 500

# Add CORS support
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    # Add security headers for webcam access
    response.headers.add('Permissions-Policy', 'camera=(), microphone=()')
    response.headers.add('Feature-Policy', 'camera *; microphone *')
    return response

def get_disease_recommendations(disease_name):
    """Get treatment recommendations from Groq AI for a specific plant disease"""
    try:
        logger.info(f"Getting recommendations for disease: {disease_name}")
        
        # Clean up disease name for better prompting
        cleaned_disease = disease_name.replace('_', ' ').strip()
        plant_type = cleaned_disease.split()[0]  # Get the plant type (first word)
        
        system_message = 'You are an expert in plant pathology and disease treatment. Provide detailed, well-structured recommendations using markdown formatting. Be specific and practical in your advice.'
        
        user_prompt = f"""As a plant pathology expert, I need detailed recommendations for treating {cleaned_disease}. The plant type is {plant_type}.

Please provide information in the following format:

## Disease Overview
[Brief description of the disease and its impact on {plant_type} plants]

## Common Symptoms
[List the main symptoms and how to identify them]

## Treatment Methods
[Provide specific treatment recommendations]

## Prevention Tips
[List preventive measures and best practices]

Please be specific to {plant_type} plants and this particular disease. Use markdown formatting for the response."""

        try:
            logger.info("Sending request to Groq API...")
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_completion_tokens=2048,
                top_p=0.9,
                stream=False
            )
            
            recommendations = chat_completion.choices[0].message.content
            logger.info(f"Successfully generated recommendations: {recommendations[:100]}...")
            return {
                'success': True,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Groq API error: {str(e)}")
            return {
                'success': False,
                'error': f'API Error: {str(e)}'
            }
            
    except Exception as e:
        logger.error(f"Unexpected error in get_disease_recommendations: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get image data from request
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Process image
        processed_frame, predicted_class, confidence = process_frame(image)
        
        # Get AI recommendations if disease is detected
        recommendations = None
        if predicted_class and '_diseased' in predicted_class.lower():
            logger.info(f"Getting recommendations for predicted class: {predicted_class}")
            recommendations = get_disease_recommendations(predicted_class)
            logger.info(f"Recommendations response: {recommendations}")
        
        # Convert processed frame to base64
        buffered = io.BytesIO()
        processed_frame.save(buffered, format="JPEG")
        processed_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response_data = {
            'success': True,
            'class': predicted_class,
            'confidence': float(confidence) if confidence is not None else None,
            'processed_image': f'data:image/jpeg;base64,{processed_image}',
            'recommendations': recommendations
        }
        
        logger.info(f"Sending prediction response with class {predicted_class} and confidence {confidence}")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    """Health check endpoint to verify model and server status"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'num_classes': len(classes) if classes else 0
    })

# Add a test endpoint to verify camera access
@app.route('/test_camera')
def test_camera():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            return jsonify({
                'success': False,
                'error': 'Failed to open camera'
            })
        
        ret, frame = camera.read()
        camera.release()
        
        if not ret:
            return jsonify({
                'success': False,
                'error': 'Failed to capture frame'
            })
            
        return jsonify({
            'success': True,
            'frame_shape': frame.shape
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    load_model_and_classes()
    logger.info("Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True) 