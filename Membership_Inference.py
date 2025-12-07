import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset and separate features from the target label.
# "target_heart" indicates whether an individual has heart disease or not (0 for no, 1 for yes).
data = pd.read_csv('NHANES-heart.csv')
X = data.drop(columns=['target_heart']).values
y = data['target_heart'].values

# 2. Split the data. Cap the training set so the model is more likely to overfit.
X_small, X_rest, y_small, y_rest = train_test_split(X, y, train_size=2000, random_state=42, stratify=y)
X_train, y_train = X_small, y_small
X_test, y_test = X_rest, y_rest

# 3. Train and fit a simple MLP. Use a small model but trained with weak regularization to preserve some memorization.
model = Pipeline([("scaler", StandardScaler()), ("mlp", MLPClassifier(hidden_layer_sizes=(32,), alpha=0.1, max_iter=1000, random_state=0, )),])
model.fit(X_train, y_train)

# 4. Evaluate the model's performance.
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_accuracy = accuracy_score(y_train, train_pred)
test_accuracy = accuracy_score(y_test, test_pred)

print("\n----Model Performance Summary---- ")
print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test  accuracy: {test_accuracy:.3f}")

#--------------------SIMPLE MEMBERSHIP INFERENCE DEMO---------------------------------------

# 5. Define a function for converting a binary label (0 or 1) into a one-hot vector.
def get_target_vector(target: int) -> np.ndarray:
    if target == 0:
        return np.array([[1, 0]])
    return np.array([[0, 1]])

# We then want a sample from both the training and test sets.
# We will pick a training example from the higher-confidence predictions.
# We will pick a testing example from the lower-confidence predictions.
# This is to more consistently demonstrate a strong membership inference attack.

# 6. Get class probabilities for all train and test points, then from each sample, get the max confidence.
train_proba_all = model.predict_proba(X_train)
test_prob_all = model.predict_proba(X_test)
train_conf_all = np.max(train_proba_all, axis=1)
test_conf_all = np.max(test_prob_all, axis=1)

# 7. Pick a random high-confidence training example.
train_conf_threshold = np.quantile(train_conf_all, 0.60)
high_conf_train_indices = np.where(train_conf_all >= train_conf_threshold)[0]
member_index = np.random.choice(high_conf_train_indices) # Member Index

# 8. Pick a random low-confidence testing example.
test_conf_threshold = np.quantile(test_conf_all, 0.40)
low_conf_test_indices = np.where(test_conf_all <= test_conf_threshold)[0]
non_member_index = np.random.choice(low_conf_test_indices) # Non-Member Index

# 9. Get various statistics for the member example.
member_x = X_train[member_index] # Get the actual feature values
member_y = get_target_vector(y_train[member_index]) # Convert its "target_heart" label to a one-hot vector.
member_probabilities = train_proba_all[member_index].reshape(1, -1) # Get the predicted probabilities.
member_pred_class = int(np.argmax(member_probabilities)) # Get the class with the highest predicted probability.
member_confidence = float(np.max(member_probabilities)) # Get the confidence of the prediction.
member_loss = log_loss(member_y, member_probabilities) # Calculate the error (log-loss)

# 10. Get various statistics for the non-member example.
non_member_x = X_test[non_member_index] # Get the actual feature values
non_member_y = get_target_vector(y_test[non_member_index]) # Convert its "target_heart" label to a one-hot vector.
non_member_probabilities = test_prob_all[non_member_index].reshape(1, -1) # Get the predicted probabilities.
non_member_pred_class = int(np.argmax(non_member_probabilities)) # Get the class with the highest predicted probability.
non_member_confidence = float(np.max(non_member_probabilities)) # Get the confidence of the prediction.
non_member_loss = log_loss(non_member_y, non_member_probabilities) # Calculate the error (log-loss)

# 11. Print our results.
print("\n----Membership Inference Demo----")
print("Training Example (Member):")
print(f"  True label:          {y_train[member_index]}") # 0 for no heart disease, and 1 for yes
print(f"  Predicted label:     {member_pred_class}")
print(f"  Class probabilities: [{member_probabilities[0][0]*100:.6f}%, {member_probabilities[0][1]*100:.6f}%]")
print(f"  Max confidence:      {member_confidence:.4f}")
print(f"  Log loss:            {member_loss:.4f}") # Small log loss means the model was very confident and correct. Large means the model was unsure or wrong.

print("\nTest Example (Non-Member):")
print(f"  True label:          {y_test[non_member_index]}")
print(f"  Predicted label:     {non_member_pred_class}")
print(f"  Class probabilities: [{non_member_probabilities[0][0]*100:.6f}%, {non_member_probabilities[0][1]*100:.6f}%]")
print(f"  Max confidence:      {non_member_confidence:.4f}")
print(f"  Log loss:            {non_member_loss:.4f}")


