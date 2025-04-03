document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadButton = document.getElementById('uploadButton');
    const imageUpload = document.getElementById('imageUpload');
    const uploadSection = document.querySelector('.upload-section');
    const results = document.getElementById('results');
    const uploadedImage = document.getElementById('uploadedImage');
    const prediction = document.getElementById('prediction');
    const recommendations = document.getElementById('recommendations');

    // Handle drag and drop
    uploadSection.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.style.borderColor = '#4CAF50';
    });

    uploadSection.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.style.borderColor = '#ccc';
    });

    uploadSection.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadSection.style.borderColor = '#ccc';
        
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // Handle button upload
    uploadButton.addEventListener('click', () => {
        imageUpload.click();
    });

    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // File handling
    function handleFile(file) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            
            reader.onload = (e) => {
                uploadedImage.src = e.target.result;
                results.style.display = 'block';
                prediction.textContent = 'Processing...';
                uploadImage(e.target.result);
            };
            
            reader.onerror = (error) => {
                console.error('Error:', error);
                showError('Error reading file');
            };
            
            reader.readAsDataURL(file);
        } else {
            showError('Please upload an image file');
        }
    }

    function showError(message) {
        results.style.display = 'block';
        prediction.textContent = message;
        prediction.style.backgroundColor = 'rgba(255, 0, 0, 0.7)';
        recommendations.style.display = 'none';
    }

    async function uploadImage(imageData) {
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData })
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Update the image with the processed version
                uploadedImage.src = data.processed_image;
                
                // Show prediction
                prediction.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                prediction.innerHTML = `
                    ${data.class}<br>
                    Confidence: ${(data.confidence * 100).toFixed(2)}%
                `;
                
                // Show recommendations if available
                if (data.recommendations && data.recommendations.success) {
                    recommendations.style.display = 'block';
                    recommendations.innerHTML = formatRecommendations(data.recommendations.recommendations);
                } else if (data.recommendations && !data.recommendations.success) {
                    recommendations.style.display = 'block';
                    recommendations.innerHTML = `<p class="error">Unable to get recommendations: ${data.recommendations.error}</p>`;
                } else {
                    recommendations.style.display = 'none';
                }
            } else {
                showError(data.error || 'Error processing image');
            }
        } catch (error) {
            console.error('Error:', error);
            showError('Error processing image');
        }
    }

    function formatRecommendations(markdown) {
        let html = markdown
            .replace(/##\s+([^\n]+)/g, '<h3>$1</h3>')
            .replace(/###\s+([^\n]+)/g, '<h4>$1</h4>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
        
        const lines = html.split('\n');
        let inList = false;
        html = lines.map(line => {
            if (line.trim().startsWith('- ')) {
                if (!inList) {
                    inList = true;
                    return '<ul><li>' + line.substring(2) + '</li>';
                }
                return '<li>' + line.substring(2) + '</li>';
            } else if (inList) {
                inList = false;
                return '</ul>' + line;
            }
            return line;
        }).join('\n');
        
        if (inList) {
            html += '</ul>';
        }
        
        return html;
    }
});