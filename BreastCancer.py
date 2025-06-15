"""
Breast Cancer Diagnosis with Logistic Regression - Complete Implementation
=========================================================================

Welcome to this comprehensive implementation of breast cancer diagnosis using 
logistic regression. This project demonstrates how to build a machine learning
model that can help identify whether a breast tumor is malignant (cancerous)
or benign (non-cancerous) based on various features extracted from cell images.

What we're accomplishing here:
- Loading and exploring real medical data about breast tumors
- Building logistic regression from scratch (no black box libraries!)
- Training the model to recognize patterns in the data
- Evaluating how well our model performs on unseen data

The beauty of this approach is that we're implementing everything ourselves,
so you can understand exactly what's happening at each step rather than
relying on pre-built functions that hide the mathematical details.

Dataset: Breast Cancer Wisconsin (Diagnostic) 
Goal: Binary classification - Malignant vs Benign tumors
Method: Logistic regression implemented from scratch
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Starting Breast Cancer Diagnosis Project")
print("=" * 50)

# ============================================================================
# STEP 1: Data Loading and Initial Exploration
# ============================================================================

def load_and_explore_data():
    """
    This function loads our breast cancer dataset and provides an initial overview.
    Think of this as a doctor first reviewing a patient's medical file before
    making any decisions - we need to understand what we're working with.
    """
    print("\nStep 1: Loading the breast cancer dataset...")
    
    # Load the CSV file containing our dataset
    # Make sure you have the 'data.csv' file in your project directory
    data = pd.read_csv("data.csv")
    
    print(f"Dataset loaded successfully!")
    print(f"We have {data.shape[0]} patient records with {data.shape[1]} features each")
    
    # Let's examine the first few rows to understand our data structure
    print("\nFirst 5 rows of our dataset:")
    print(data.head())
    
    # Get detailed information about our dataset
    print(f"\nDataset Overview:")
    print(f"   Total patients: {len(data)}")
    print(f"   Total features: {len(data.columns)}")
    print(f"   Any missing values: {data.isnull().sum().sum()}")
    
    # Analyze the distribution of diagnoses
    if 'diagnosis' in data.columns:
        diagnosis_counts = data['diagnosis'].value_counts()
        print(f"   Benign cases (B): {diagnosis_counts.get('B', 0)}")
        print(f"   Malignant cases (M): {diagnosis_counts.get('M', 0)}")
        
        # Calculate the percentage distribution
        total = len(data)
        benign_pct = (diagnosis_counts.get('B', 0) / total) * 100
        malignant_pct = (diagnosis_counts.get('M', 0) / total) * 100
        print(f"   Distribution: {benign_pct:.1f}% benign, {malignant_pct:.1f}% malignant")
    
    return data

# ============================================================================
# STEP 2: Data Preprocessing and Cleaning
# ============================================================================

def preprocess_data(data):
    """
    Here we clean and prepare our data for machine learning. Medical datasets
    often have some columns we don't need and values that need to be converted
    to numbers that our algorithm can understand.
    """
    print("\nStep 2: Preprocessing and cleaning the data...")
    
    # First, let's see what columns we have
    print("Original columns in our dataset:")
    print(data.columns.tolist())
    
    # Remove columns that don't help with prediction
    # The 'id' column is just a unique identifier - not useful for prediction
    # 'Unnamed: 32' appears to be an empty column that sometimes appears in this dataset
    columns_to_drop = []
    if 'id' in data.columns:
        columns_to_drop.append('id')
    if 'Unnamed: 32' in data.columns:
        columns_to_drop.append('Unnamed: 32')
    
    if columns_to_drop:
        data = data.drop(columns_to_drop, axis=1)
        print(f"Dropped unnecessary columns: {columns_to_drop}")
    
    # Convert diagnosis from letters to numbers
    # M (Malignant) becomes 1, B (Benign) becomes 0
    # This is necessary because our algorithm works with numbers, not letters
    if 'diagnosis' in data.columns:
        original_values = data['diagnosis'].unique()
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
        print(f"Converted diagnosis: {original_values} -> [0=Benign, 1=Malignant]")
    
    # Separate our target variable (what we want to predict) from our features
    y = data['diagnosis'].values  # This is what we want to predict
    x_data = data.drop(['diagnosis'], axis=1)  # These are our input features
    
    print(f"Target variable shape: {y.shape}")
    print(f"Features shape: {x_data.shape}")
    print(f"Feature names: {list(x_data.columns)}")
    
    return x_data, y

# ============================================================================
# STEP 3: Feature Normalization
# ============================================================================

def normalize_features(x_data):
    """
    Normalization is crucial for logistic regression. Different features might
    have very different scales (like age vs income), and we need to put them
    on the same scale so our algorithm treats them fairly.
    
    We use min-max normalization: (value - min) / (max - min)
    This transforms all features to be between 0 and 1.
    """
    print("\nStep 3: Normalizing features...")
    
    # Show the original ranges of our features
    print("Original feature ranges (first 5 features):")
    for col in x_data.columns[:5]:
        print(f"   {col}: {x_data[col].min():.2f} to {x_data[col].max():.2f}")
    
    # Apply min-max normalization
    # Formula: (x - min) / (max - min)
    # This ensures all features are between 0 and 1
    x_normalized = (x_data - x_data.min()) / (x_data.max() - x_data.min())
    
    # Show the new ranges (should all be 0 to 1)
    print("\nAfter normalization (first 5 features):")
    for col in x_normalized.columns[:5]:
        print(f"   {col}: {x_normalized[col].min():.2f} to {x_normalized[col].max():.2f}")
    
    print("All features are now scaled between 0 and 1")
    
    return x_normalized

# ============================================================================
# STEP 4: Split Data for Training and Testing
# ============================================================================

def split_data(x, y, test_size=0.15, random_state=42):
    """
    We need to split our data into two parts:
    - Training set: Used to teach our model
    - Testing set: Used to evaluate how well our model learned
    
    It's like studying for an exam with practice questions, then taking
    a real exam with different questions to see if you truly understand.
    """
    print(f"\nStep 4: Splitting data into training and testing sets...")
    print(f"Using {int((1-test_size)*100)}% for training, {int(test_size*100)}% for testing")
    
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    
    # Transpose the data to match our mathematical formulations
    # Our equations expect features as rows and samples as columns
    x_train = x_train.T
    x_test = x_test.T
    y_train = y_train.reshape(1, -1)  # Make it a row vector
    y_test = y_test.reshape(1, -1)    # Make it a row vector
    
    print(f"Training set: {x_train.shape[1]} samples with {x_train.shape[0]} features")
    print(f"Testing set: {x_test.shape[1]} samples with {x_test.shape[0]} features")
    print(f"Training labels: {y_train.shape}")
    print(f"Testing labels: {y_test.shape}")
    
    return x_train, x_test, y_train, y_test

# ============================================================================
# STEP 5: Initialize Model Parameters
# ============================================================================

def initialize_parameters(dimension):
    """
    Before we can train our model, we need to initialize the weights and bias.
    Think of weights as the importance our model assigns to each feature,
    and bias as a baseline prediction before considering any features.
    
    We start with small random weights and zero bias.
    """
    print(f"\nStep 5: Initializing model parameters...")
    print(f"Creating {dimension} weights (one for each feature) and 1 bias term")
    
    # Initialize weights with small random values
    # We multiply by 0.01 to keep them small initially
    weights = np.random.randn(dimension, 1) * 0.01
    
    # Initialize bias to zero
    bias = 0.0
    
    print(f"Weights initialized with shape: {weights.shape}")
    print(f"Bias initialized to: {bias}")
    print(f"Initial weight range: {weights.min():.4f} to {weights.max():.4f}")
    
    return weights, bias

# ============================================================================
# STEP 6: Define the Sigmoid Function
# ============================================================================

def sigmoid(z):
    """
    The sigmoid function is the heart of logistic regression.
    It takes any real number and squashes it between 0 and 1,
    which we can interpret as a probability.
    
    Formula: sigmoid(z) = 1 / (1 + e^(-z))
    
    When z is large positive: sigmoid ≈ 1 (high probability of class 1)
    When z is large negative: sigmoid ≈ 0 (high probability of class 0)
    When z is 0: sigmoid = 0.5 (uncertain)
    """
    # Clip z to prevent overflow in exponential function
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# ============================================================================
# STEP 7: Forward and Backward Propagation
# ============================================================================

def forward_backward_propagation(weights, bias, x_train, y_train):
    """
    This is where the magic happens! We make predictions and then learn from our mistakes.
    
    Forward propagation: Make predictions based on current weights and bias
    Backward propagation: Calculate how to adjust weights and bias to improve
    """
    # Get the number of training samples
    m = x_train.shape[1]
    
    # FORWARD PROPAGATION
    # Calculate the linear combination: z = weights * x + bias
    z = np.dot(weights.T, x_train) + bias
    
    # Apply sigmoid to get probabilities
    y_pred = sigmoid(z)
    
    # CALCULATE COST (how wrong are we?)
    # We use cross-entropy loss, which heavily penalizes confident wrong predictions
    # Add small epsilon to prevent log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    cost = (-1/m) * np.sum(
        y_train * np.log(y_pred) + (1 - y_train) * np.log(1 - y_pred)
    )
    
    # BACKWARD PROPAGATION
    # Calculate gradients (how much to change each parameter)
    dw = (1/m) * np.dot(x_train, (y_pred - y_train).T)
    db = (1/m) * np.sum(y_pred - y_train)
    
    gradients = {
        "dw": dw,  # Derivative with respect to weights
        "db": db   # Derivative with respect to bias
    }
    
    return cost, gradients

# ============================================================================
# STEP 8: Update Parameters (Learning)
# ============================================================================

def update_parameters(weights, bias, x_train, y_train, learning_rate, num_iterations):
    """
    This is where our model actually learns! We repeatedly:
    1. Make predictions
    2. Calculate how wrong we are
    3. Adjust our weights and bias to be less wrong
    4. Repeat
    
    The learning_rate controls how big steps we take - too big and we might
    overshoot the optimal solution, too small and learning takes forever.
    """
    print(f"\nStep 8: Training the model...")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of iterations: {num_iterations}")
    print("\nTraining progress:")
    
    costs = []  # Keep track of our cost over time
    
    for i in range(num_iterations):
        # Calculate cost and gradients
        cost, gradients = forward_backward_propagation(weights, bias, x_train, y_train)
        
        # Update parameters using gradient descent
        weights = weights - learning_rate * gradients["dw"]
        bias = bias - learning_rate * gradients["db"]
        
        # Print progress every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"   Iteration {i:4d}: Cost = {cost:.6f}")
    
    # Store final parameters
    parameters = {
        "weights": weights,
        "bias": bias
    }
    
    print(f"Training completed! Final cost: {cost:.6f}")
    
    return parameters, costs

# ============================================================================
# STEP 9: Make Predictions
# ============================================================================

def predict(weights, bias, x_test):
    """
    Now that our model is trained, let's use it to make predictions on new data!
    We'll get probabilities and convert them to binary predictions.
    """
    # Calculate probabilities
    z = np.dot(weights.T, x_test) + bias
    probabilities = sigmoid(z)
    
    # Convert probabilities to binary predictions
    # If probability > 0.5, predict malignant (1), otherwise benign (0)
    predictions = (probabilities > 0.5).astype(int)
    
    return predictions, probabilities

# ============================================================================
# STEP 10: Evaluate Model Performance
# ============================================================================

def evaluate_model(y_true, y_pred, dataset_name=""):
    """
    Let's see how well our model performed! We'll calculate accuracy and
    show some other useful metrics.
    """
    # Calculate accuracy
    accuracy = 100 - np.mean(np.abs(y_pred - y_true)) * 100
    
    print(f"{dataset_name} Accuracy: {accuracy:.2f}%")
    
    # Calculate additional metrics manually
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = np.sum((y_true_flat == 1) & (y_pred_flat == 1))
    fp = np.sum((y_true_flat == 0) & (y_pred_flat == 1))
    tn = np.sum((y_true_flat == 0) & (y_pred_flat == 0))
    fn = np.sum((y_true_flat == 1) & (y_pred_flat == 0))
    
    # Calculate precision, recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"{dataset_name} Precision: {precision:.3f}")
    print(f"{dataset_name} Recall: {recall:.3f}")
    print(f"{dataset_name} F1-Score: {f1_score:.3f}")
    
    return accuracy

# ============================================================================
# STEP 11: Complete Logistic Regression Function
# ============================================================================

def logistic_regression_complete(x_train, y_train, x_test, y_test, 
                               learning_rate=0.01, num_iterations=1000):
    """
    This function puts everything together - the complete logistic regression pipeline!
    """
    print("\n" + "="*60)
    print("TRAINING LOGISTIC REGRESSION MODEL")
    print("="*60)
    
    # Get the number of features
    dimension = x_train.shape[0]
    
    # Initialize parameters
    weights, bias = initialize_parameters(dimension)
    
    # Train the model
    parameters, costs = update_parameters(
        weights, bias, x_train, y_train, learning_rate, num_iterations
    )
    
    # Make predictions
    print(f"\nStep 9: Making predictions...")
    train_predictions, train_probs = predict(
        parameters["weights"], parameters["bias"], x_train
    )
    test_predictions, test_probs = predict(
        parameters["weights"], parameters["bias"], x_test
    )
    
    # Evaluate performance
    print(f"\nStep 10: Evaluating model performance...")
    print("-" * 40)
    train_accuracy = evaluate_model(y_train, train_predictions, "Training")
    print("-" * 40)
    test_accuracy = evaluate_model(y_test, test_predictions, "Testing")
    print("-" * 40)
    
    # Check for overfitting
    accuracy_diff = train_accuracy - test_accuracy
    if accuracy_diff > 5:
        print(f"\nWarning: Possible overfitting detected!")
        print(f"Training accuracy is {accuracy_diff:.1f}% higher than test accuracy")
    else:
        print(f"\nGood news: Model generalizes well!")
        print(f"Difference between training and test accuracy: {accuracy_diff:.1f}%")
    
    return parameters, costs, train_accuracy, test_accuracy

# ============================================================================
# STEP 12: Visualization Functions
# ============================================================================

def plot_cost_history(costs, num_iterations):
    """
    Let's visualize how our model learned over time by plotting the cost.
    The cost should generally decrease as the model learns.
    """
    plt.figure(figsize=(10, 6))
    iterations = range(0, num_iterations, 100)
    plt.plot(iterations, costs, 'b-', linewidth=2, marker='o')
    plt.title('Model Learning Progress: Cost vs Iterations', fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Cost (Cross-Entropy Loss)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Cost decreased from {costs[0]:.6f} to {costs[-1]:.6f}")
    print(f"Total reduction: {((costs[0] - costs[-1]) / costs[0] * 100):.1f}%")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function that runs our complete breast cancer diagnosis pipeline.
    """
    print("BREAST CANCER DIAGNOSIS WITH LOGISTIC REGRESSION")
    print("="*60)
    print("This program will build a logistic regression model from scratch")
    print("to diagnose breast cancer as malignant or benign.")
    print("="*60)
    
    try:
        # Step 1: Load and explore data
        data = load_and_explore_data()
        
        # Step 2: Preprocess data
        x_data, y = preprocess_data(data)
        
        # Step 3: Normalize features
        x_normalized = normalize_features(x_data)
        
        # Step 4: Split data
        x_train, x_test, y_train, y_test = split_data(x_normalized, y)
        
        # Step 5-11: Train and evaluate model
        parameters, costs, train_acc, test_acc = logistic_regression_complete(
            x_train, y_train, x_test, y_test, 
            learning_rate=0.01, num_iterations=1000
        )
        
        # Step 12: Visualize results
        print(f"\nStep 11: Visualizing results...")
        plot_cost_history(costs, 1000)
        
        # Final summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Training Accuracy: {train_acc:.2f}%")
        print(f"Testing Accuracy: {test_acc:.2f}%")
        print(f"Model Status: {'Ready for deployment!' if test_acc > 90 else 'Needs improvement'}")
        print("="*60)
        
        # Interpretation
        print("\nWhat do these results mean?")
        if test_acc > 95:
            print("Excellent! This model shows very high accuracy and could be")
            print("a valuable tool for assisting medical professionals.")
        elif test_acc > 90:
            print("Very good! This model shows high accuracy but might benefit")
            print("from additional tuning or more data.")
        elif test_acc > 80:
            print("Good start! This model shows promise but needs improvement")
            print("before it could be used in clinical settings.")
        else:
            print("The model needs significant improvement. Consider:")
            print("- More data, better features, or different algorithms")
        
        return parameters, costs
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check that your data file exists and is properly formatted.")
        return None, None

# Run the complete pipeline
if __name__ == "__main__":
    model_parameters, cost_history = main()
