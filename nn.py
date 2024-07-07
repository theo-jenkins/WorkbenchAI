import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import psycopg2
from psycopg2 import sql
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

def greeting():
    """
    Displays the neural network dashboard and gets the user input.
    
    Returns:
        int: user_choice
    """

    print("This is your neural network dashboard.")
    print("[1] - Load data")
    print("[2] - Save to db")
    print("[3] - Process data")
    print("[4] - Build model")
    print("[5] - Train model")
    print("[6] - Make prediction")
    print("[7] - Exit")

    while True:
        try:
            answer = int(input("Enter your choice:"))
            if answer in [1,2,3,4,5,6,7]:
                return answer
            else:
                print("Please enter a valid number.")
        except ValueError:
            print("Invalid input: Please enter a valid number.")

def upload_data():
    #This function will allow the user to select .csv file
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
    filetypes=[("PNG files", "*.png")],
    title="Select a PNG file"
    )
    return file_path

def save_to_db():
    #This function will allow the user to save the data to the db
    return db

def process_data():
    #This function will process the dataset into nn features
    return db

def build_model():
    #Outlines the architecture of the neural network
    #Returns: neural net model
    model = Sequential()

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

def predict(test_data, model):
    #This function will make a prediction

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