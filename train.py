import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
import gdown

# Download and extract dataset
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('data.zip')

# Data loading utilities
def load_video(path: str) -> List[float]: 
    """
    Loads a video from the given path, converts each frame to grayscale,
    crops a specific region from each frame, and normalizes the frames
    to have zero mean and unit standard deviation.

    Parameters:
        path (str): Path to the input video file.

    Returns:
        List[float]: A list of normalized and cropped grayscale frames 
                     as TensorFlow tensors of type float32.
    """
     # Open the video file
    cap = cv2.VideoCapture(path)
    frames = []
 # Iterate through all frames in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
             # Convert the frame from RGB to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
# Crop the frame to the region [190:236, 80:220]
        # Resulting shape: (46, 140, 1)
        frames.append(frame[190:236, 80:220, :])
# Release the video capture object
    cap.release()
 # Compute the mean and standard deviation of all frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
 # Normalize the frames to zero mean and unit variance
    return tf.cast((frames - mean), tf.float32) / std


# vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
# char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

# Define the vocabulary: lowercase letters, apostrophe, question mark, exclamation mark, digits, and space
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

# Create a layer that maps characters (strings) to integer indices
# Any character not in the vocabulary will be mapped to the out-of-vocabulary (OOV) token index
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")

# Create an inverse mapping layer that converts indices back to characters (strings)
# This is useful for decoding model predictions back to readable text
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),  # Get the full vocab including special tokens
    oov_token="",
    invert=True  # Invert to map integers back to strings
)

# def load_alignments(path: str) -> List[str]: 
#     with open(path, 'r') as f: 
#         lines = f.readlines() 
#     tokens = []
#     for line in lines:
#         line = line.split()
#         if line[2] != 'sil': 
#             tokens = [*tokens, ' ', line[2]]
#     return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_alignments(path: str) -> List[str]:
    """
    Loads alignment labels from a given file, extracts non-silence tokens, and
    converts them into a sequence of character indices using `char_to_num`.

    Parameters:
        path (str): Path to the alignment file (usually `.align`), which contains
                    timing and phoneme or word annotations.

    Returns:
        List[str]: A tensor of numeric character indices (excluding the first token),
                   representing the text labels for a video.
    """
    # Open the alignment file and read all lines
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []

    # Process each line to extract phoneme/word tokens
    for line in lines:
        line = line.split()
        # Ignore 'sil' (silence) tokens
        if line[2] != 'sil':
            # Add a space followed by the token to the list
            tokens = [*tokens, ' ', line[2]]

    # Convert list of tokens into a flat sequence of characters
    characters = tf.strings.unicode_split(tokens, input_encoding='UTF-8')

    # Reshape to a flat list of characters
    characters = tf.reshape(characters, (-1,))

    # Map characters to their numeric representation and remove the first index (initial space)
    return char_to_num(characters)[1:]



def load_data(path: str):
    """
    Loads video frames and alignment labels given a path to a data file.

    This function decodes the TensorFlow `tf.Tensor` path (typically passed through a
    `tf.data.Dataset` pipeline), extracts the filename, constructs paths to the corresponding
    `.mpg` video and `.align` label file, and loads both using helper functions.

    Parameters:
        path (str): A TensorFlow string tensor containing the file path (e.g., b'data/s1/sample.align').

    Returns:
        tuple:
            - frames: A list of normalized grayscale video frame tensors from `load_video`.
            - alignments: A tensor of character indices from `load_alignments`.
    """

    # Decode the TensorFlow byte string to a regular Python string
    path = bytes.decode(path.numpy())

    # Extract the base filename (without extension)
    file_name = path.split('/')[-1].split('.')[0]

    # Construct full paths to the video and alignment files
    video_path = os.path.join('data', 's1', f'{file_name}.mpg')
    alignment_path = os.path.join('data', 'alignments', 's1', f'{file_name}.align')

    # Load the video frames and alignment labels
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    # Return both as a tuple
    return frames, alignments

#def mappable_function(path: str) -> List[str]:
 #   result = tf.py_function(load_data, [path], (tf.float32, tf.int64))
 #   return result
def mappable_function(path: str) -> List[str]:
    """
    Wrapper function for mapping over a tf.data.Dataset.
    
    Uses tf.py_function to wrap the Python-based `load_data` function so it can be used 
    in TensorFlow data pipelines.

    Parameters:
        path (str): A TensorFlow string tensor representing the file path 
                    (usually of an `.align` file).

    Returns:
        tuple:
            - A Tensor of type tf.float32: the preprocessed video frames.
            - A Tensor of type tf.int64: the numerical alignment labels.
    """
    # tf.py_function allows wrapping a regular Python function (like load_data)
    # inside a TensorFlow data pipeline. It returns raw TensorFlow tensors.
    result = tf.py_function(
        func=load_data,           # Python function to call
        inp=[path],               # Input to the function
        Tout=(tf.float32, tf.int64)  # Output data types: frames and alignments
    )

    return result

# Data pipeline
#data = tf.data.Dataset.list_files('./data/s1/*.mpg')
#data = data.shuffle(500, reshuffle_each_iteration=False)
#data = data.map(mappable_function)
#data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
#data = data.prefetch(tf.data.AUTOTUNE)

#train = data.take(450)
#test = data.skip(450)

# Step 1: List all .mpg video files in the specified folder
data = tf.data.Dataset.list_files('./data/s1/*.mpg')

# Step 2: Shuffle the dataset with a buffer size of 500 for randomness
# `reshuffle_each_iteration=False` ensures reproducibility (useful for testing/debugging)
data = data.shuffle(500, reshuffle_each_iteration=False)

# Step 3: Map each video file to (frames, alignments) using the custom mapping function
# The function uses tf.py_function to load and preprocess the data
data = data.map(mappable_function)

# Step 4: Pad the variable-length sequences in each batch to fixed shapes
# Padded shapes:
#   - Frames: [75, height, width, channels] — batch of 2 videos, each with max 75 frames
#   - Labels: [40] — max 40 characters per alignment label
data = data.padded_batch(
    batch_size=2,
    padded_shapes=([75, None, None, None], [40])  # Padding video and label to match batch size
)

# Step 5: Prefetch to overlap preprocessing and model execution (for performance)
data = data.prefetch(tf.data.AUTOTUNE)

# Step 6: Split the dataset into training and testing sets
# Take first 450 examples for training
train = data.take(450)

# Use the remaining examples for testing
test = data.skip(450)

# Model definition
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

model = Sequential()
# 1st Conv3D layer: Extracts spatiotemporal features from video frames
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
# MaxPool3D reduces spatial dimensions (not time)
model.add(MaxPool3D((1, 2, 2)))
# 2nd Conv3D layer
model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

# 3rd Conv3D layer
model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))
# Flatten the 3D features per frame into a 1D vector for LSTM input
# Final shape: (batch, time_steps, features) = (None, 75, 75*5*17)
model.add(Reshape((-1, 75 * 5 * 17)))
# ------------------ Bidirectional LSTM Layers ------------------

# LSTM layer to model temporal dependencies in both directions
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
# Second Bidirectional LSTM for deeper temporal modeling
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
# ------------------ Output Layer ------------------

# Final Dense layer with softmax activation to classify each timestep
# Output size is vocabulary size + 1 (for CTC blank token)
model.add(Dense(char_to_num.vocabulary_size() + 1, activation='softmax'))
model.summary()




# Custom loss
def CTCLoss(y_true, y_pred):
    """
    Computes the Connectionist Temporal Classification (CTC) loss.

    This loss is used when the alignment between input (video/audio frames)
    and output (character sequences) is unknown, such as in lip reading.

    Parameters:
        y_true (Tensor): Ground truth labels, shape (batch_size, max_label_len)
                         — padded label sequences.
        y_pred (Tensor): Predicted output, shape (batch_size, time_steps, vocab_size)
                         — softmax probabilities over vocabulary for each frame.

    Returns:
        Tensor: A tensor representing the CTC loss value for the batch.
    """
     # Get batch size
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
     # Input sequence length (time steps from the model output)
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
      # Label sequence length (ground truth sequence length)
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
      # Create tensor of shape (batch_size, 1) for input length
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    # Create tensor of shape (batch_size, 1) for label length
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    # Compute CTC loss using TensorFlow's built-in function
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)



# Learning rate scheduler
def scheduler(epoch, lr):
    """
    Learning rate scheduler function.

    Keeps the learning rate constant for the first 30 epochs,
    then applies exponential decay after epoch 30.

    Parameters:
        epoch (int): Current epoch number.
        lr (float): Current learning rate.

    Returns:
        float: Updated learning rate.
    """
    if epoch < 30:
        # No change in LR for the first 30 epochs
        return lr
    else:
        # Apply exponential decay after epoch 30
        return lr * tf.math.exp(-0.1)

# Compile and train
model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
# Save weights callback (save only weights, not full model)
checkpoint_callback = ModelCheckpoint(os.path.join('models', 'checkpoint.weights.h5'), monitor='loss', save_weights_only=True)
# Learning rate scheduler callback
schedule_callback = LearningRateScheduler(scheduler)
# Train the model with callbacks
model.fit(train, validation_data=test, epochs=100, callbacks=[checkpoint_callback, schedule_callback])
