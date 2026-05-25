from sklearn.metrics import accuracy_score, classification_report, recall_score
import numpy as np

Regularizations = [RidgeLogisticGD, LassoLogisticGD, ElasticNetLogisticGD]
lambdas_ = [0.001,0.01,0.1,1,10,100]
lrs = [0.001,0.01,0.1,1,10]
alphas = np.arange(0.1, 1.1, 0.1)
weights = {}
loss = {}
min_iter = {}

# Initialize tracking variables for each model
best_configs = {
    "RidgeLogisticGD": {"accuracy": 0, "lambda": None, "lr": None, "balanced": False},
    "LassoLogisticGD": {"accuracy": 0, "lambda": None, "lr": None, "balanced": False},
    "ElasticNetLogisticGD": {"accuracy": 0, "lambda": None, "lr": None, "alpha": None, "balanced": False},
}

weights = {}
loss = {}
min_iter = {}

# Function to predict classes using learned weights
def predict(X, w, threshold=0.5):
    probabilities = sigmoid(X @ w)  # Compute probabilities
    return (probabilities >= threshold).astype(int)  # Convert to class labels

# Define threshold for recall balancing
min_recall = 0.3  # Both classes must have recall above this
max_recall_diff = 0.3  # Maximum allowed difference in recall between classes

# Loop through models and hyperparameters
for r in Regularizations:
    for lam in lambdas_:
        for lr in lrs:
            if r.__name__ != "ElasticNetLogisticGD":
                try:
                    weights[r.__name__], loss[r.__name__], min_iter[r.__name__] = r(X_train, y_train, lam, lr)
                except Exception as e:
                    print(f"Error in {r.__name__} with lam={lam}, lr={lr}: {e}")
                    continue
            else:
                for alpha in alphas:
                    try:
                        weights[r.__name__], loss[r.__name__], min_iter[r.__name__] = r(X_train, y_train, lam, alpha, lr)
                    except Exception as e:
                        print(f"Error in {r.__name__} with lam={lam}, lr={lr}, alpha={alpha}: {e}")
                        continue

            # Predict on the test set
            y_pred = predict(X_test, weights[r.__name__])
            
            # Compute accuracy and recall
            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average=None)  # Recall for each class
            
            # Check recall balancing condition
            is_balanced = all(r >= min_recall for r in recall) and abs(recall[0] - recall[1]) <= max_recall_diff
            
            # Update best configuration for the current model
            if acc > best_configs[r.__name__]["accuracy"] and is_balanced:
                best_configs[r.__name__]["accuracy"] = acc
                best_configs[r.__name__]["lambda"] = lam
                best_configs[r.__name__]["lr"] = lr
                best_configs[r.__name__]["balanced"] = True
                if r.__name__ == "ElasticNetLogisticGD":
                    best_configs[r.__name__]["alpha"] = alpha
            
            # Print classification report for current configuration
            print(f"Model: {r.__name__}, lr: {lr}, lambda: {lam}, alpha: {alpha if r.__name__ == 'ElasticNetLogisticGD' else 'N/A'}")
            print(classification_report(y_test, y_pred, target_names=['California', 'Florida'], zero_division=0))
            print(f"Recall: {recall}, Balanced: {is_balanced}\n")

# Print the best configuration for each model
print("\nBest Configurations:")
for model, config in best_configs.items():
    print(f"Model: {model}")
    print(f"  Best Accuracy: {config['accuracy']:.4f}")
    print(f"  Lambda: {config['lambda']}")
    print(f"  Learning Rate: {config['lr']}")
    if model == "ElasticNetLogisticGD":
        print(f"  Alpha: {config['alpha']}")
    print(f"  Balanced Recall: {config['balanced']}")
    print()
