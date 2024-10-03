import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
data = pd.read_csv('your_dataset.csv')

# Split features and target
X = data.drop('sales_volume', axis=1).values  # Features (113 columns)
y = data['sales_volume'].values  # Target (sales volume)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)  # Reshape for output layer
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Define the neural network model
class SalesVolumePredictor(nn.Module):
    def __init__(self):
        super(SalesVolumePredictor, self).__init__()
        self.fc1 = nn.Linear(113, 128)  # Input layer
        self.fc2 = nn.Linear(128, 64)    # Hidden layer 1
        self.fc3 = nn.Linear(64, 32)     # Hidden layer 2
        self.fc4 = nn.Linear(32, 1)      # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Activation for first layer
        x = torch.relu(self.fc2(x))  # Activation for second layer
        x = torch.relu(self.fc3(x))  # Activation for third layer
        x = self.fc4(x)               # Output layer
        return x

# Instantiate the model
model = SalesVolumePredictor()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Backpropagation
    optimizer.step()       # Update weights
    
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)

print(f'Test Loss: {test_loss.item():.4f}')

# Make predictions
predictions = test_outputs.numpy()

# Optionally, plot the results
import matplotlib.pyplot as plt

plt.scatter(y_test_tensor.numpy(), predictions, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Sales Volume')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Identity line
plt.show()
