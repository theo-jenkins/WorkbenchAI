from keras import layers, models
import numpy as np

def greeting():
    """
    Displays the neural network dashboard and gets the user input.
    
    Returns:
        int: user_choice
    """

    print("This is your neural network dashboard.")
    print("[1] - Train keras NN")
    print("[2] - Exit")

    while True:
        try:
            answer = int(input("Enter your choice:"))
            if answer in [1,2]:
                return answer
            else:
                print("Please enter 1 or 2")
        except ValueError:
            print("Invalid input: Please enter a valid number")

def keras_function():
    from keras.datasets import boston_housing

    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std
    
    test_data -= mean
    test_data /= std

    model = build_model()
    model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print(f"test_mse_score = {test_mse_score}, test_mae_score = {test_mae_score}")

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1])))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

    return model

def main():
    user_choice = 0
    while user_choice != 2:
        user_choice = greeting()
    
    if user_choice == 1:
        keras_function()

if __name__ == "__main__":
    main()
