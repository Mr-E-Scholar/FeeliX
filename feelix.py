import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder

directory = "."

def find_max_length(data):
    max_len = 0
    for item in data:
        length = item.shape[0] if len(item.shape) > 1 else len(item)
        if length > max_len:
            max_len = length
    return max_len

def load_and_normalize_data(directory):
    data = []
    labels = []
    
    # Load files, get labels
    for filename in os.listdir(directory):
        if filename.startswith("mymfcc") and filename.endswith(".npy"):
            mfcc_features = np.load(os.path.join(directory, filename))
            data.append(mfcc_features)

            # Extract the 'feeling' part of the filename
            feeling = filename.split('_')[1]
            labels.append(feeling)

    print("Original data shapes:")
    for item in data:
        print(item.shape)

    # Padding
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

    # Reshape for normalization
    reshaped_data = [item.flatten() for item in padded_data]
    reshaped_data = np.vstack(reshaped_data)
    print("\nReshaped data shape:", reshaped_data.shape)

    # Normalize
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(reshaped_data)
    labels = np.array(labels)
    return data_normalized, labels

# Load and normalize data
data, labels = load_and_normalize_data(directory)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
# Further split training data into training and validation
train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.25, random_state=42)

all_labels = np.unique(labels)
print(all_labels)
label_encoder = LabelEncoder()
label_encoder.fit_transform(labels)  

# Now transform the labels in each dataset
train_labels = label_encoder.transform(train_labels)
val_labels = label_encoder.transform(val_labels)
test_labels = label_encoder.transform(test_labels)

# Model
num_classes = len(all_labels)  
model = Sequential([
    Dense(128, activation='relu', input_shape=(train_data.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')  # Make sure this matches 'num_classes'
])

# Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(train_data, train_labels, 
                    validation_data=(val_data, val_labels), 
                    epochs=50,  # Adjust the number of epochs
                    batch_size=32)  # Adjust the batch size

# Evaluate
test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# Prediction
predictions = model.predict(test_data)



def process_single_file(file_path, max_len, scaler):
    mfcc_features = np.load(file_path)
    padding = ((0, max_len - mfcc_features.shape[0]), (0, 0))
    padded_data = np.pad(mfcc_features, padding, mode='constant', constant_values=0)
    reshaped_data = padded_data.flatten().reshape(1, -1)
    normalized_data = scaler.fit_transform(reshaped_data)
    return normalized_data

# Prediciton File
single_file_path = "mymfcc_negative_tired_1.npy"
max_len=499
scaler = StandardScaler()
single_data = process_single_file(single_file_path, max_len, scaler)

# Predict using the model
single_prediction = model.predict(single_data)
predicted_index = np.argmax(single_prediction)
predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
print(f"Predicted emotion for '{single_file_path}': {predicted_emotion}")