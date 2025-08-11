"""
Federated Learning Client Simulator
Simulates multiple hospitals/clinics participating in federated learning
"""

import torch
import numpy as np
import requests
import json
import logging
import time
import random
from datetime import datetime
from threading import Thread
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FederatedClient:
    def __init__(self, client_id, server_url="http://localhost:5001", client_type="hospital"):
        self.client_id = client_id
        self.server_url = server_url
        self.client_type = client_type
        self.local_model = None
        self.local_data_size = random.randint(100, 1000)  # Simulated local dataset size
        self.current_round = 0
        self.training_history = []
        
        # Simulate different hospital characteristics
        self.hospital_info = self._generate_hospital_info()
        
    def _generate_hospital_info(self):
        """Generate realistic hospital characteristics"""
        hospital_types = [
            {"name": "General Hospital", "accuracy_bias": 0.85, "data_quality": "high"},
            {"name": "Rural Clinic", "accuracy_bias": 0.78, "data_quality": "medium"},
            {"name": "Teaching Hospital", "accuracy_bias": 0.92, "data_quality": "high"},
            {"name": "Community Hospital", "accuracy_bias": 0.82, "data_quality": "medium"},
            {"name": "Specialized Medical Center", "accuracy_bias": 0.89, "data_quality": "high"}
        ]
        
        hospital_type = random.choice(hospital_types)
        return {
            "institution_name": f"{hospital_type['name']} #{self.client_id}",
            "location": f"City_{random.randint(1, 100)}",
            "data_size": self.local_data_size,
            "accuracy_baseline": hospital_type["accuracy_bias"],
            "data_quality": hospital_type["data_quality"],
            "specialization": random.choice(["general", "pulmonology", "radiology", "emergency"])
        }
    
    def register_with_server(self):
        """Register this client with the federated server"""
        try:
            response = requests.post(f"{self.server_url}/federated/register", 
                                   json={
                                       "client_id": self.client_id,
                                       "client_info": {
                                           "type": self.client_type,
                                           "hospital_info": self.hospital_info,
                                           "capabilities": ["training", "inference"]
                                       }
                                   })
            if response.status_code == 200:
                logger.info(f"Client {self.client_id} registered successfully")
                return True
            else:
                logger.error(f"Registration failed for {self.client_id}: {response.text}")
                return False
        except Exception as e:
            logger.error(f"Registration error for {self.client_id}: {e}")
            return False
    
    def simulate_local_training(self):
        """Simulate local model training with realistic variations"""
        # Simulate training time (varies by hospital resources)
        training_time = random.uniform(30, 120)  # 30-120 seconds
        logger.info(f"Client {self.client_id} starting local training (estimated {training_time:.1f}s)")
        
        # Simulate progressive training
        for epoch in range(5):
            time.sleep(training_time / 5)  # Simulate epoch time
            epoch_accuracy = self._simulate_training_accuracy(epoch)
            logger.info(f"Client {self.client_id} - Epoch {epoch+1}/5: {epoch_accuracy:.3f}")
        
        # Generate simulated model weights
        final_accuracy = self._simulate_final_accuracy()
        model_weights = self._generate_mock_weights()
        
        training_result = {
            "client_id": self.client_id,
            "num_samples": self.local_data_size,
            "accuracy": final_accuracy,
            "model_weights": model_weights,
            "training_time": training_time,
            "hospital_info": self.hospital_info
        }
        
        self.training_history.append({
            "round": self.current_round + 1,
            "accuracy": final_accuracy,
            "samples": self.local_data_size,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Client {self.client_id} completed training: accuracy={final_accuracy:.3f}")
        return training_result
    
    def _simulate_training_accuracy(self, epoch):
        """Simulate realistic training progression"""
        base_accuracy = self.hospital_info["accuracy_baseline"]
        
        # Simulate learning curve
        progress = (epoch + 1) / 5
        accuracy = base_accuracy + (0.15 * progress) + random.gauss(0, 0.02)
        
        # Add some noise based on data quality
        if self.hospital_info["data_quality"] == "medium":
            accuracy += random.gauss(0, 0.01)
        elif self.hospital_info["data_quality"] == "low":
            accuracy += random.gauss(0, 0.02)
            
        return max(0.5, min(0.99, accuracy))  # Clamp between 50% and 99%
    
    def _simulate_final_accuracy(self):
        """Simulate final training accuracy with realistic variations"""
        base_accuracy = self.hospital_info["accuracy_baseline"]
        
        # Add variations based on hospital characteristics
        if self.hospital_info["specialization"] == "pulmonology":
            base_accuracy += 0.03  # Pneumonia specialists perform better
        elif self.hospital_info["specialization"] == "emergency":
            base_accuracy -= 0.02  # Emergency departments might have more challenging cases
            
        # Add random variation
        final_accuracy = base_accuracy + random.gauss(0, 0.02)
        return max(0.6, min(0.95, final_accuracy))  # Realistic range
    
    def _generate_mock_weights(self):
        """Generate mock model weights for demonstration"""
        # In a real scenario, these would be actual model parameters
        # For simulation, we generate random weights that could represent model updates
        mock_weights = {}
        layer_names = [
            "vit.embeddings.patch_embeddings.projection.weight",
            "vit.encoder.layer.0.attention.attention.query.weight",
            "vit.encoder.layer.0.attention.attention.key.weight",
            "classifier.weight"
        ]
        
        for name in layer_names:
            # Generate small random weights to simulate model updates
            if "weight" in name:
                if "attention" in name:
                    shape = (768, 768)
                elif "projection" in name: # For vit.embeddings.patch_embeddings.projection.weight
                    shape = (768, 3, 16, 16) # Assuming 3 input channels, 16x16 patch size
                elif "classifier" in name: # For classifier.weight
                    shape = (2, 768) # Assuming 2 classes
                else:
                    # Fallback for other weight layers if any, might need adjustment
                    shape = (768, 768) # Default to a common size
                weights = np.random.normal(0, 0.001, shape).tolist()
            else:
                # This part handles biases, which are typically 1D
                shape = (768,) # Assuming 1D bias for 768 features
                weights = np.random.normal(0, 0.001, shape).tolist()
            mock_weights[name] = weights
            
        return mock_weights
    
    def participate_in_round(self):
        """Participate in one federated learning round"""
        logger.info(f"Client {self.client_id} starting FL round {self.current_round + 1}")
        
        # Simulate local training
        training_result = self.simulate_local_training()
        
        # Send results to server
        try:
            response = requests.post(f"{self.server_url}/federated/update_model",
                                   json={"client_updates": [training_result]})
            
            if response.status_code == 200:
                result = response.json()
                self.current_round = result.get("round", self.current_round + 1)
                logger.info(f"Client {self.client_id} successfully completed round {self.current_round}")
                return True
            else:
                logger.error(f"Failed to send updates: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending updates: {e}")
            return False
    
    def get_status(self):
        """Get client status and statistics"""
        return {
            "client_id": self.client_id,
            "hospital_info": self.hospital_info,
            "current_round": self.current_round,
            "training_history": self.training_history[-5:],  # Last 5 rounds
            "status": "active"
        }

class FederatedSimulation:
    def __init__(self, num_clients=5, server_url="http://localhost:5001"):
        self.server_url = server_url
        self.clients = []
        self.simulation_active = False
        
        # Create diverse set of clients
        for i in range(num_clients):
            client_id = f"hospital_{i+1:02d}"
            client = FederatedClient(client_id, server_url)
            self.clients.append(client)
    
    def register_all_clients(self):
        """Register all clients with the server"""
        logger.info(f"Registering {len(self.clients)} clients...")
        
        success_count = 0
        for client in self.clients:
            if client.register_with_server():
                success_count += 1
            time.sleep(1)  # Avoid overwhelming the server
        
        logger.info(f"Successfully registered {success_count}/{len(self.clients)} clients")
        return success_count == len(self.clients)
    
    def run_federated_round(self, round_number):
        """Run one federated learning round with all clients"""
        logger.info(f"=== Starting Federated Learning Round {round_number} ===")
        
        # Randomly select subset of clients (simulating real-world availability)
        participating_clients = random.sample(self.clients, k=random.randint(3, len(self.clients)))
        
        logger.info(f"Round {round_number}: {len(participating_clients)} clients participating")
        
        # Collect training results from all participating clients
        client_updates = []
        threads = []
        
        def client_training_thread(client):
            training_result = client.simulate_local_training()
            client_updates.append(training_result)
        
        # Start parallel training for all clients
        for client in participating_clients:
            thread = Thread(target=client_training_thread, args=(client,))
            thread.start()
            threads.append(thread)
        
        # Wait for all clients to finish
        for thread in threads:
            thread.join()
        
        # Send aggregated updates to server
        if client_updates:
            try:
                response = requests.post(f"{self.server_url}/federated/update_model",
                                       json={"client_updates": client_updates})
                
                if response.status_code == 200:
                    result = response.json()
                    self.current_round = result.get("round", round_number)
                    logger.info(f"Round {round_number} completed successfully. Server round: {result.get('round')}")
                    
                    # Update client round numbers
                    for client in participating_clients:
                        client.current_round = result.get('round', round_number)
                    
                    return True
                else:
                    logger.error(f"Server rejected updates: {response.text}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error sending aggregated updates: {e}")
                return False
        
        return False
    
    def run_simulation(self, num_rounds=10, delay_between_rounds=30):
        """Run the complete federated learning simulation"""
        logger.info(f"Starting Federated Learning Simulation: {num_rounds} rounds")
        
        if not self.register_all_clients():
            logger.error("Failed to register all clients. Aborting simulation.")
            return
        
        self.simulation_active = True
        
        try:
            for round_num in range(1, num_rounds + 1):
                if not self.simulation_active:
                    break
                
                success = self.run_federated_round(round_num)
                
                if success:
                    logger.info(f"Round {round_num} completed successfully")
                else:
                    logger.error(f"Round {round_num} failed")
                
                # Display server status
                self._display_server_status()
                
                # Wait before next round (except for the last round)
                if round_num < num_rounds:
                    logger.info(f"Waiting {delay_between_rounds}s before next round...")
                    time.sleep(delay_between_rounds)
            
            logger.info("Federated Learning Simulation completed!")
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
            self.simulation_active = False
    
    def _display_server_status(self):
        """Display current server status"""
        try:
            response = requests.get(f"{self.server_url}/federated/status")
            if response.status_code == 200:
                status = response.json()
                logger.info(f"Server Status: Round {status.get('training_rounds')}, "
                          f"Clients: {status.get('connected_clients')}, "
                          f"Accuracy: {status.get('latest_accuracy', 0):.3f}")
        except Exception as e:
            logger.warning(f"Could not fetch server status: {e}")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client Simulator")
    parser.add_argument("--clients", type=int, default=5, help="Number of simulated clients")
    parser.add_argument("--rounds", type=int, default=10, help="Number of FL rounds")
    parser.add_argument("--delay", type=int, default=30, help="Delay between rounds (seconds)")
    parser.add_argument("--server", type=str, default="http://localhost:5001", help="FL server URL")
    
    args = parser.parse_args()
    
    # Create and run simulation
    simulation = FederatedSimulation(num_clients=args.clients, server_url=args.server)
    simulation.run_simulation(num_rounds=args.rounds, delay_between_rounds=args.delay)

if __name__ == "__main__":
    main()
