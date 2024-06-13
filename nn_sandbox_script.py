import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
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
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    #Normalize the data (between 0 - 1)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print('Data loaded!')
    return x_train, y_train, x_test, y_test

def build_model(x_train):
    #Outlines the architecture of the neural network
    #Returns: neural net model
    input_shape = x_train.shape[1:]

    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print('Model built!')
    return model

def train_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    
    print('Model trained!')
    return results

def load_and_process_image():
    file_path = select_file()
    image = Image.open(file_path).convert('L') #Convert to grayscale
    image = image.resize((28,28)) #Resizes to 28x28 (two brackets to pass a tuple as an argument)
    image_array = np.array(image) #Converts to numoy array
    image_array = image_array / 255.0 #Normalizes the pixel value to be between 0 and 1

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

def predict(image_vector, model):
   # Add a batch dimension to the image vector
    prediction = model.predict(np.expand_dims(image_vector, axis=0))
    # Find the index of the maximum prediction
    max_index = np.argmax(prediction)
    # Find the maximum prediction value
    max_value = np.max(prediction)
    # Print the prediction results
    print(f'Prediction: {prediction}')
    print(f'The digit is {max_index}, with a confidence of {max_value * 100:.2f}%')

def main():
    user_choice = 0
    while user_choice != 6:
        user_choice = greeting()

        if user_choice == 1:
            x_train, y_train, x_test, y_test = prepare_data()
        if user_choice == 2:
            model = build_model(x_train)
        if user_choice == 3:
            results = train_model(x_train, y_train, x_test, y_test, model)
        if user_choice == 4:
            image_vector = load_and_process_image()
        if user_choice == 5:
            prediction = predict(image_vector, model)
        
    


if __name__ == "__main__":
    main()