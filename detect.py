from keras.models import load_model
import librosa
import numpy as np

# Load the pre-trained model
model = load_model('model.h5')

# Function to preprocess audio file
def preprocess_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, mono=True, duration=30)  # Load 30 seconds of audio
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    # Compute the target length for padding
    target_length = 128
    # Resize MFCCs to the target length
    resized_mfccs = np.zeros((20, target_length))
    if mfccs.shape[1] >= target_length:
        resized_mfccs = mfccs[:, :target_length]
    else:
        resized_mfccs[:, :mfccs.shape[1]] = mfccs
    # Reshape for model input
    reshaped_mfccs = resized_mfccs.reshape(1, 20, target_length, 1)
    return reshaped_mfccs




# Path to the audio file you want to classify
audio_file_path = 'Data\genres_original\hiphop\hiphop.00001.wav'

# Preprocess the audio file
preprocessed_audio = preprocess_audio(audio_file_path)

# Your code to make predictions
predictions = model.predict(preprocessed_audio)
# Find the index of the maximum probability
predicted_index = np.argmax(predictions)

print("Expected index:", 7, "Predicted index:", predicted_index)


