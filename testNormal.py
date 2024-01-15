import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

# Directory where your .npy files are stored
directory = "."

# Function to find the maximum length of MFCC features in the dataset
def find_max_length(data):
    max_len = 0
    for item in data:
        length = item.shape[0] if len(item.shape) > 1 else len(item)
        if length > max_len:
            max_len = length
    return max_len

# Load and normalize the data
def load_and_normalize_data(directory):
    data = []
    labels = []
    
    # Load each file and extract features and labels
    for filename in os.listdir(directory):
        if filename.startswith("mymfcc") and filename.endswith(".npy"):
            mfcc_features = np.load(os.path.join(directory, filename))
            data.append(mfcc_features)

            # Extract the 'feeling' part of the filename
            feeling = filename.split('_')[1]
            labels.append(feeling)

            # label = filename.split('_')[1]
            # labels.append(label)

    print("Original data shapes:")
    for item in data:
        print(item.shape)

    # Pad the sequences to have the same length
    max_len = find_max_length(data)
    print(f"the max length is {max_len}") #### max_len=499
    padded_data = []
    for item in data:
        if len(item.shape) > 1:
            padding = ((0, max_len - item.shape[0]), (0, 0))
        else:
            padding = ((0, max_len - len(item)),)
        padded_item = np.pad(item, padding, mode='constant', constant_values=0)
        padded_data.append(padded_item)

    print("\nPadded data shapes:")
    for item in padded_data:
        print(item.shape)

    # Reshape the data for normalization
    reshaped_data = [item.flatten() for item in padded_data]
    reshaped_data = np.vstack(reshaped_data)

    print("\nReshaped data shape:", reshaped_data.shape)

    # Normalize the data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(reshaped_data)

    labels = np.array(labels)

    return data_normalized, labels

# Load and normalize data
data, labels = load_and_normalize_data(directory)

# Optionally split the data into training, validation, and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
# Further split training data into training and validation if needed
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)
# Now, train_data, val_data, and test_data are your datasets to be used in the neural network.


all_labels = np.unique(labels)
print(all_labels)
# Convert string labels to integer labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)  # Fit on all unique labels

# Now transform the labels in each dataset
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Neural Network Model
# Adjust the number of unique classes
num_classes = len(all_labels)  # This should be 3 in your case

# Adjust the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Make sure this matches 'num_classes'
])

# Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the Model
history = model.fit(train_data, train_labels, 
                    validation_data=(val_data, val_labels), 
                    epochs=50,  # Adjust the number of epochs
                    batch_size=32)  # Adjust the batch size

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Use the Model for Prediction
predictions = model.predict(test_data)
# You can now use `predictions` to see the model's emotion predictions
# for i, prediction in enumerate(predictions):
#     predicted_index = np.argmax(prediction)
#     predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
#     print(f"Instance {i}: Predicted emotion - {predicted_emotion}")

# Add this code after the model evaluation

def process_single_file(file_path, max_len, scaler):
    # Load the MFCC features from the single file
    mfcc_features = np.load(file_path)

    # Pad the sequence to have the same length
    padding = ((0, max_len - mfcc_features.shape[0]), (0, 0))
    padded_data = np.pad(mfcc_features, padding, mode='constant', constant_values=0)

    # Reshape and normalize the data
    reshaped_data = padded_data.flatten().reshape(1, -1)
    normalized_data = scaler.fit_transform(reshaped_data)
    return normalized_data

# Path to he .npy file you want to predict
# single_file_path = "mymfcc_positive_good_2.npy"
# single_file_path = "mymfcc_neutral_okay_1.npy"
single_file_path = "mymfcc_negative_tired_1.npy"

# Process the single file
max_len=499
scaler = StandardScaler()
single_data = process_single_file(single_file_path, max_len, scaler)

# Predict using the model

single_prediction = model.predict(single_data)
predicted_index = np.argmax(single_prediction)
predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

# Print the prediction

print(f"Predicted emotion for '{single_file_path}': {predicted_emotion}")