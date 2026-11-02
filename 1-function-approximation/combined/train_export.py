import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Model for Square function (y = x^2)
class SquareModel(nn.Module):
    def __init__(self):
        super(SquareModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # Linear output layer without activation
        )

    def forward(self, x):
        return self.network(x)

# Model for Cubic function (y = x^3)
class CubicModel(nn.Module):
    def __init__(self):
        super(CubicModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 1)  # Linear output layer without activation
        )

    def forward(self, x):
        return self.network(x)

# Model for Sin function (y = sin(x))
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 16),
            CustomSin(),  # Using sin as nonlinearity
            nn.Dropout(0.2),
            nn.Linear(16, 16),
            CustomSin(),  # Using sin as nonlinearity
            nn.Dropout(0.2),
            nn.Linear(16, 1)  # Linear output layer without activation
        )

    def forward(self, x):
        return self.network(x)

# Custom Sin activation
class CustomSin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

def train_model(model, function_name, true_function, epochs=3000, learning_rate=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Generate training data
    X = torch.linspace(-5, 5, 1000).reshape(-1, 1)
    y = true_function(X)
    
    print(f"\nTraining function: {function_name}")
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Set model to training mode if it has dropout
        if isinstance(model, SinModel):
            model.train()
        
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress every 500 epochs
        if epoch == 1 or epoch % 500 == 0:
            # Set model to evaluation mode for MSE calculation if it has dropout
            if isinstance(model, SinModel):
                model.eval()
                with torch.no_grad():
                    outputs = model(X)
                    loss = criterion(outputs, y)
            print(f"Epoch {epoch}, MSE = {loss.item():.9f}")
    
    training_time = time.time() - start_time
    
    # Calculate model size
    model_path = f"{function_name}_model.onnx"
    
    # Set model to eval mode before export if it has dropout
    if isinstance(model, SinModel):
        model.eval()
    
    # Export to ONNX
    dummy_input = torch.randn(1, 1)
    torch.onnx.export(model, dummy_input, model_path, 
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}})
    
    model_size = os.path.getsize(model_path) / 1024  # Size in KB
    
    print(f"\nTraining time = {training_time:.2f} seconds")
    print(f"Model Size = {model_size:.2f} KB")
    
    return model

# Define the functions to approximate
def square_function(x):
    return x**2

def cube_function(x):
    return x**3

def sin_function(x):
    return torch.sin(x)

# Train models for each function
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Train Square function model
    square_model = train_model(SquareModel(), "Square", square_function)
    
    # Train Cube function model
    cube_model = train_model(CubicModel(), "Cube", cube_function)
    
    # Train Sin function model
    sin_model = train_model(SinModel(), "Sin", sin_function)
