"""
Federated Learning Server for Pneumonia Classification
Implements FedAvg algorithm with ViT model coordination
"""

import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ViTForImageClassification, ViTConfig
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedServer:
    def __init__(self, model_name="lxyuan/vit-xray-pneumonia-classification"):
        self.model_name = model_name
        self.global_model = None
        self.client_models = {}
        self.training_rounds = 0
        self.performance_history = []
        self.connected_clients = set()
        
        # Initialize global model
        self.initialize_global_model()
        
    def initialize_global_model(self):
        """Initialize the global ViT model"""
        try:
            self.global_model = ViTForImageClassification.from_pretrained(self.model_name)
            logger.info(f"Global model initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            
    def register_client(self, client_id: str, client_info: Dict[str, Any]):
        """Register a new client"""
        self.connected_clients.add(client_id)
        logger.info(f"Client {client_id} registered. Total clients: {len(self.connected_clients)}")
        return {"status": "registered", "client_id": client_id}
    
    def aggregate_models(self, client_updates: List[Dict[str, Any]]):
        """
        Implement FedAvg aggregation algorithm
        """
        if not client_updates:
            return None
            
        # Get the global model state dict
        global_dict = self.global_model.state_dict()
        
        # Initialize aggregated weights
        aggregated_dict = {}
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        for key in global_dict.keys():
            aggregated_dict[key] = torch.zeros_like(global_dict[key])
            
        # Weighted averaging based on number of samples
        for update in client_updates:
            client_weight = update['num_samples'] / total_samples
            client_state_dict = update['model_weights']
            
            for key in aggregated_dict.keys():
                if key in client_state_dict:
                    aggregated_dict[key] += client_weight * torch.tensor(client_state_dict[key])
        
        # Update global model
        self.global_model.load_state_dict(aggregated_dict)
        self.training_rounds += 1
        
        # Log performance
        avg_accuracy = np.mean([update.get('accuracy', 0) for update in client_updates])
        self.performance_history.append({
            'round': self.training_rounds,
            'accuracy': avg_accuracy,
            'num_clients': len(client_updates),
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Model aggregated. Round: {self.training_rounds}, Avg Accuracy: {avg_accuracy:.4f}")
        return {"success": True, "round": self.training_rounds}
    
    def get_global_model_weights(self):
        """Return current global model weights"""
        if self.global_model is None:
            return None
        return {k: v.cpu().numpy().tolist() for k, v in self.global_model.state_dict().items()}
    
    def get_server_status(self):
        """Return server status and statistics"""
        return {
            "status": "active",
            "training_rounds": self.training_rounds,
            "connected_clients": len(self.connected_clients),
            "model_name": self.model_name,
            "performance_history": self.performance_history[-10:],  # Last 10 rounds
            "latest_accuracy": self.performance_history[-1]['accuracy'] if self.performance_history else 0
        }

# Flask app for federated server API
app = Flask(__name__)
CORS(app)

# Initialize federated server
fed_server = FederatedServer()

@app.route('/federated/register', methods=['POST'])
def register_client():
    """Register a new federated learning client"""
    data = request.json
    client_id = data.get('client_id')
    client_info = data.get('client_info', {})
    
    result = fed_server.register_client(client_id, client_info)
    return jsonify(result)

@app.route('/federated/get_model', methods=['GET'])
def get_global_model():
    """Send current global model to client"""
    weights = fed_server.get_global_model_weights()
    if weights is None:
        return jsonify({"error": "Model not available"}), 500
    
    return jsonify({
        "model_weights": weights,
        "round": fed_server.training_rounds
    })

@app.route('/federated/update_model', methods=['POST'])
def update_global_model():
    """Receive model updates from clients and aggregate"""
    try:
        client_updates = request.json.get('client_updates', [])
        result = fed_server.aggregate_models(client_updates)
        
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "No updates to aggregate"}), 400
            
    except Exception as e:
        logger.error(f"Error updating model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/federated/status', methods=['GET'])
def get_server_status():
    """Get federated server status and metrics"""
    status = fed_server.get_server_status()
    return jsonify(status)

@app.route('/federated/performance', methods=['GET'])
def get_performance_history():
    """Get detailed performance history"""
    return jsonify({
        "performance_history": fed_server.performance_history,
        "total_rounds": fed_server.training_rounds
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
