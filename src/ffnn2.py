import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_percentage_error


# Generate the polynomial data with noise
def generate_data(num_samples, noise_std=1.0):
    torch.manual_seed(42)  # For reproducibility
    X = torch.linspace(-50, 50, num_samples).view(-1, 1)
    y_true = X**4 - 5*X**3 + 3*X**2 - X + 2
    noise = torch.randn_like(y_true) * noise_std
    y_noisy = y_true + noise
    return X, y_noisy

# Define the regression model
class PolynomialRegression(nn.Module):
    def __init__(self):
        super(PolynomialRegression, self).__init__()
        self.layer1 = nn.Linear(1, 100)  # Input feature: x, Output feature: 100
        self.layer2 = nn.Linear(100, 1)  # Input feature: 100, Output feature: y

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Train the model
def train_model(model, X_train, y_train, num_epochs=1000, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        y_pred = model(X_train)
        loss = loss_func(y_pred, y_train)
        r2 = r2_score(y_noisy.detach(), y_pred.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}, r2: {r2:.4f}")

# Generate data
num_samples = 100
X, y_noisy = generate_data(num_samples, noise_std=30)

# Normalize the data
X = (X - X.mean()) / X.std()
y_noisy = (y_noisy - y_noisy.mean()) / y_noisy.std()

# Initialize the model
model = PolynomialRegression()

# Train the model
train_model(model, X, y_noisy)

# Generate predictions
with torch.no_grad():
    model.eval()
    y_pred = model(X)

# Plot the results
plt.scatter(X, y_noisy, label='Noisy Data')
plt.plot(X, y_pred, 'r', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
