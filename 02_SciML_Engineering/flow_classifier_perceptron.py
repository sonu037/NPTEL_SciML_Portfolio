import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.001, n_iterations=10000):
        self.lr = learning_rate
        self.n_iters = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, 0)
                
                update = self.lr * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# ==========================================
# Engineering Dataset: Fluid Flow Regimes
# X1 = Fluid Velocity, X2 = Fluid Density 
# ==========================================
if __name__ == "__main__":
    # Generating synthetic data for Laminar (Low velocity/density)
    np.random.seed(42)
    laminar_X = np.random.randn(50, 2) * 2 + np.array([5, 5])
    laminar_y = np.zeros(50)

    # Generating synthetic data for Turbulent (High velocity/density)
    turbulent_X = np.random.randn(50, 2) * 2 + np.array([12, 12])
    turbulent_y = np.ones(50)

    # Combine data
    X = np.vstack((laminar_X, turbulent_X))
    y = np.hstack((laminar_y, turbulent_y))

    # Train the Perceptron
    p = Perceptron(learning_rate=0.001, n_iterations=10000)
    p.fit(X, y)

    # ==========================================
    # Plotting for LinkedIn
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.scatter(laminar_X[:, 0], laminar_X[:, 1], color='blue', label='Laminar Flow', alpha=0.7)
    plt.scatter(turbulent_X[:, 0], turbulent_X[:, 1], color='red', label='Turbulent Flow', alpha=0.7)

    # Draw the AI's Decision Boundary
    x0_1 = np.amin(X[:, 0])
    x0_2 = np.amax(X[:, 0])
    x1_1 = (-p.weights[0] * x0_1 - p.bias) / p.weights[1]
    x1_2 = (-p.weights[0] * x0_2 - p.bias) / p.weights[1]
    plt.plot([x0_1, x0_2], [x1_1, x1_2], 'k--', lw=2, label='AI Decision Boundary')

    plt.title("Perceptron Scratch Model: Fluid Flow Regime Classification")
    plt.xlabel("Fluid Velocity (normalized)")
    plt.ylabel("Fluid Density (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the image so i can upload it to LinkedIn!
    plt.savefig('fluid_classifier_graph.png', dpi=300, bbox_inches='tight')
    print("Training Complete! Graph saved as 'fluid_classifier_graph.png'")