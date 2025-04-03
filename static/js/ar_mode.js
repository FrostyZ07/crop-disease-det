// Global variables
let lastPredictionTime = 0;
const PREDICTION_INTERVAL = 1000; // 1 second between predictions

// Initialize AR scene
AFRAME.registerComponent('ar-scene-init', {
    init: function() {
        this.setupARScene();
        this.setupPredictionLoop();
    },

    setupARScene: function() {
        const scene = this.el;
        scene.addEventListener('camera-set-active', () => {
            console.log('AR camera is ready');
        });
    },

    setupPredictionLoop: function() {
        setInterval(() => {
            this.captureAndPredict();
        }, PREDICTION_INTERVAL);
    },

    captureAndPredict: async function() {
        try {
            const video = document.querySelector('video');
            if (!video || !video.videoWidth) return;

            // Create canvas and capture frame
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0);

            // Convert to base64
            const imageData = canvas.toDataURL('image/jpeg');

            // Send to server for prediction
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    image: imageData
                })
            });

            const result = await response.json();
            if (result.success) {
                this.updateARDisplay(result);
            }
        } catch (error) {
            console.error('Prediction error:', error);
        }
    },

    updateARDisplay: function(result) {
        // Update UI elements
        const plantName = document.getElementById('plant-name');
        const confidenceLevel = document.getElementById('confidence-level');
        const plantInfo = document.getElementById('plant-info');

        if (result.class) {
            plantName.textContent = result.class;
            confidenceLevel.textContent = `Confidence: ${(result.confidence * 100).toFixed(2)}%`;
            
            // Update AR text
            plantInfo.setAttribute('value', `${result.class}\n${(result.confidence * 100).toFixed(2)}% confident`);
            
            // Change color based on confidence
            const color = result.confidence > 0.7 ? '#00ff00' : '#ffff00';
            plantInfo.setAttribute('color', color);
        } else {
            plantName.textContent = 'No plant detected';
            confidenceLevel.textContent = 'Confidence: 0%';
            plantInfo.setAttribute('value', 'Point camera at a plant');
            plantInfo.setAttribute('color', '#ffffff');
        }
    }
});

// Add the component to the scene
document.addEventListener('DOMContentLoaded', () => {
    const scene = document.querySelector('a-scene');
    scene.setAttribute('ar-scene-init', '');
}); 