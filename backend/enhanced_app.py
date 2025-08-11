"""
Enhanced Flask App with Federated Learning Integration
Includes attention visualization and advanced medical interface
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
from PIL import Image
import os
import numpy as np
import torch
import requests
import logging
from datetime import datetime
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the Hugging Face model with processor for attention visualization
model_name = "lxyuan/vit-xray-pneumonia-classification"
try:
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name, output_attentions=True)
    classifier = pipeline(task="image-classification", model=model_name)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")

class FederatedLearningIntegration:
    def __init__(self, server_url="http://localhost:5001"):
        self.server_url = server_url
        self.client_id = f"medical_client_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.registered = False
        
    def register_with_server(self):
        """Register this client with the federated server"""
        try:
            response = requests.post(f"{self.server_url}/federated/register", 
                                   json={
                                       "client_id": self.client_id,
                                       "client_info": {
                                           "type": "medical_inference_client",
                                           "capabilities": ["classification", "attention_visualization"]
                                       }
                                   })
            if response.status_code == 200:
                self.registered = True
                logger.info(f"Successfully registered with federated server as {self.client_id}")
                return True
        except Exception as e:
            logger.warning(f"Could not register with federated server: {e}")
        return False
    
    def get_server_status(self):
        """Get federated learning server status"""
        try:
            response = requests.get(f"{self.server_url}/federated/status")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Could not get server status: {e}")
        return None

# Initialize federated learning integration
fed_integration = FederatedLearningIntegration()
fed_integration.register_with_server()

def generate_attention_map(image, model, processor):
    """Generate attention visualization for ViT model"""
    try:
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions
        
        # Get the last layer attention (most relevant for final decision)
        last_attention = attentions[-1].squeeze(0)  # Remove batch dimension
        
        # Average across attention heads
        avg_attention = last_attention.mean(dim=0)
        
        # Reshape to 2D grid (excluding CLS token)
        patch_size = int(np.sqrt(avg_attention.shape[0] - 1))  # -1 for CLS token
        attention_grid = avg_attention[1:].reshape(patch_size, patch_size)  # Skip CLS token
        
        # Convert to numpy
        attention_map = attention_grid.cpu().numpy()
        
        return attention_map
    except Exception as e:
        logger.error(f"Error generating attention map: {e}")
        return None

def create_attention_overlay(original_image, attention_map):
    """Create an overlay of attention map on original image"""
    try:
        # Resize attention map to match image size
        from scipy.ndimage import zoom
        
        img_array = np.array(original_image)
        if len(img_array.shape) == 3:
            h, w = img_array.shape[:2]
        else:
            h, w = img_array.shape
            
        # Resize attention map
        attention_resized = zoom(attention_map, (h/attention_map.shape[0], w/attention_map.shape[1]))
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original X-ray')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(original_image, cmap='gray')
        plt.imshow(attention_resized, cmap='jet', alpha=0.5)
        plt.title('Attention Heatmap Overlay')
        plt.axis('off')
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        
        # Convert to base64
        attention_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return attention_image
    except Exception as e:
        logger.error(f"Error creating attention overlay: {e}")
        return None

@app.route('/classify', methods=['POST'])
def classify_image():
    """Enhanced classification with attention visualization"""
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files['image']
    include_attention = request.form.get('include_attention', 'false').lower() == 'true'
    
    if file:
        try:
            # Save file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Open and classify the image
            image = Image.open(filepath).convert("RGB")
            result = classifier(image)
            
            response_data = {
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_name": model_name,
                    "federated_learning_enabled": fed_integration.registered
                }
            }
            
            # Add attention visualization if requested
            if include_attention:
                attention_map = generate_attention_map(image, model, processor)
                if attention_map is not None:
                    attention_overlay = create_attention_overlay(image, attention_map)
                    if attention_overlay:
                        response_data["attention_visualization"] = attention_overlay
            
            # Add federated learning info
            fed_status = fed_integration.get_server_status()
            if fed_status:
                response_data["federated_info"] = {
                    "training_rounds": fed_status.get("training_rounds", 0),
                    "connected_clients": fed_status.get("connected_clients", 0),
                    "latest_accuracy": fed_status.get("latest_accuracy", 0)
                }
            
            # Remove the uploaded image after processing
            os.remove(filepath)
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return jsonify({"error": f"Processing error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Invalid file"}), 400

@app.route('/federated/status', methods=['GET'])
def get_federated_status():
    """Get federated learning status"""
    fed_status = fed_integration.get_server_status()
    if fed_status:
        return jsonify(fed_status)
    else:
        return jsonify({"error": "Federated server not available"}), 503

@app.route('/model/info', methods=['GET'])
def get_model_info():
    """Get detailed model information"""
    return jsonify({
        "model_name": model_name,
        "model_type": "Vision Transformer (ViT)",
        "task": "Chest X-ray Pneumonia Classification",
        "federated_learning": {
            "enabled": fed_integration.registered,
            "client_id": fed_integration.client_id,
            "server_url": fed_integration.server_url
        },
        "capabilities": [
            "Pneumonia Classification",
            "Attention Visualization",
            "Confidence Scoring",
            "Federated Learning Integration"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "model_loaded": model is not None,
            "federated_server": fed_integration.registered
        }
    })

if __name__ == "__main__":
    os.makedirs('./uploads', exist_ok=True)
    logger.info("Starting enhanced medical AI server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
