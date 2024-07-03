import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
import tkinter as tk
from tkinter import filedialog
from PIL import Image

def greeting():
    """
    Displays the neural network dashboard and gets the user input.
    
    Returns:
        int: user_choice
    """

    print("This is your neural network dashboard.")
    print("[1] - Prepare data")
    print("[2] - Build model")
    print("[3] - Train model")
    print("[4] - Load file")
    print("[5] - Predict loaded file")
    print("[6] - Exit")

    while True:
        try:
            answer = int(input("Enter your choice:"))
            if answer in [1,2,3,4,5,6]:
                return answer
            else:
                print("Please enter a valid number.")
        except ValueError:
            print("Invalid input: Please enter a valid number.")

def prepare_data():
    #Load data (mnist digit dataset)
    #x refers to images, y refers to their labels
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    #Normalize the data (between 0 - 1)
    train_images = train_images.reshape((60000, 784)).astype('float32') / 255
    test_images = test_images.reshape((10000, 784)).astype('float32') / 255
    train_labels = to_categorical(train_labels) #Prepares the labels
    test_labels = to_categorical(test_labels)

    print('Data loaded!')
    return train_images, train_labels, test_images, test_labels

def build_model(train_images, train_labels, test_images, test_labels):
    #Outlines the architecture of the neural network
    #Returns: neural net model
    input_shape = train_images.shape[1:]

    model = Sequential()
    #model.add(Input(shape=input_shape))
    model.add(Dense(128, input_shape=(28 * 28,)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    print('Model built!')
    print(f"Model summary: {model.summary()}")
    return model

def train_model(train_images, train_labels, test_images, test_labels, model):
    model.fit(train_images, train_labels, epochs=4, batch_size=512, shuffle=True, validation_split=0.2, verbose=1)
    results = model.evaluate(test_images, test_labels)
    
    print('Model trained!')
    return results

def load_and_process_image(train_images):
    file_path = select_file()
    image = Image.open(file_path).convert('L') #Convert to grayscale
    image = image.resize((28, 28)) #Resizes to 28x28 (two brackets to pass a tuple as an argument)
    image_array = np.array(image) #Converts to numpy array
    image_array = image_array / 255.0 #Normalizes the pixel value to be between 0 and 1
    image_array = 1 - image_array #Inverts the picture so white=0
    image_array = image_array.flatten()
    #image_array= np.expand_dims(image_array, axis=-1) #Adds channel dimension

    print('Image processed!')
    return image_array

def select_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        filetypes=[("PNG files", "*.png")],
        title="Select a PNG file"
    )
    return file_path

def predict(image_array, model):
    prediction = model.predict(np.expand_dims(image_array, axis=0)) # Adds batch dimension
    max_index = np.argmax(prediction) # Find the index of the maximum prediction
    max_value = np.max(prediction) # Find the maximum prediction value

    # Print the prediction results
    print(f'Prediction: {prediction}')
    print(f'The digit is {max_index}, with a confidence of {max_value * 100:.2f}%')

def main():
    user_choice = 0
    while user_choice != 6:
        user_choice = greeting()

        if user_choice == 1:
            train_images, train_labels, test_images, test_labels = prepare_data()
        if user_choice == 2:
            model = build_model(train_images, train_labels, test_images, test_labels)
        if user_choice == 3:
            results = train_model(train_images, train_labels, test_images, test_labels, model)
        if user_choice == 4:
            image_array = load_and_process_image(train_images)
        if user_choice == 5:
            prediction = predict(image_array, model)     


if __name__ == "__main__":
    main()