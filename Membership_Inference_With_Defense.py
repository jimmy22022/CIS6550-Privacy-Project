import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

# 1. Load the dataset and separate features from the target label.
# "target_heart" indicates whether an individual has heart disease or not (0 for no, 1 for yes).
data = pd.read_csv('NHANES-heart.csv')
X = data.drop(columns=['target_heart']).values
y = data['target_heart'].values

# 2. Split the data. Cap the training set to mimic the original attack as closely as possible.
X_small, X_rest, y_small, y_rest = train_test_split(X, y, train_size=2000, random_state=42, stratify=y)
X_train, y_train = X_small, y_small
X_test, y_test = X_rest, y_rest

# 3. Scale the features so that each input dimension has similar magnitude.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Define the model. This is a small Keras MLP that approximates the scikit-learn MLP used in the attack script.
def make_model(input_dim: int) -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(2, activation="softmax"), # Softmax because the output for target_heart is 0 or 1.
    ])
model = make_model(X_train_scaled.shape[1])

# 5. Define the optimizer. We will use DP-SGD defined in Tensorflow Privacy.
BATCH_SIZE = 50 # Number of training examples used in each gradient update.

optimizer = DPKerasSGDOptimizer(
    num_microbatches=25, # Number of smaller batches in the larger batch. Gradient clipping and noise is applied here.
    l2_norm_clip=0.5, # Clip the influence of any noisy examples.
    noise_multiplier=2.0, # Define how much Gaussian noise is added to the gradients.
    learning_rate=0.005, # Define the step size for each update during training.
)

# 6. Define the loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy(
    from_logits=False, # Don't change the model's outputs which are already probabilities.
    reduction=tf.keras.losses.Reduction.NONE, # Output the loss for every example separately.
)

# 7. Compile the model and evaluate its performance.
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"],)
history = model.fit( X_train_scaled, y_train, epochs=50, batch_size=BATCH_SIZE, validation_data=(X_test_scaled, y_test), shuffle=True, verbose=1,)

train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc   = model.evaluate(X_test_scaled, y_test, verbose=0)

print("\n----DP Model Performance Summary----")
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test  accuracy: {test_acc:.3f}")

#--------------------SIMPLE MEMBERSHIP INFERENCE WITH DIFFERENTIAL PRIVACY DEMO---------------------------------------

# 5. Define a function for converting a binary label (0 or 1) into a one-hot vector.
def get_target_vector(target: int) -> np.ndarray:
    if target == 0:
        return np.array([[1, 0]])
    return np.array([[0, 1]])

# 6. Get class probabilities for all train and test points, then from each sample, get the max confidence.
train_proba_all = model.predict(X_train_scaled)
test_prob_all = model.predict(X_test_scaled)
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
print("\n----Membership Inference with Differntial Privacy Demo----")
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









