import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

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
    print("[4] - Exit")

    while True:
        try:
            answer = int(input("Enter your choice:"))
            if answer in [1,2,3,4]:
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

    print(f"Data prepared: {x_train[1]}, label: {y_train[1]}")
    return x_train, y_train, x_test, y_test

def build_model(x_train):
    #Outlines the architecture of the neural network
    #Returns: neural net model
    input_shape = x_train.shape[1]

    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    print(f"Model built")
    return model

def train_model(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(f"results")
    return results

def main():
    user_choice = 0
    while user_choice != 4:
        user_choice = greeting()

        if user_choice == 1:
            x_train, y_train, x_test, y_test = prepare_data()
        if user_choice == 2:
            model = build_model(x_train)
        if user_choice == 3:
            train_model(x_train, y_train, x_test, y_test, model)
        
    


if __name__ == "__main__":
    main()
