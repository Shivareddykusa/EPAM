from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt

import pandas as pd

class Annmodel:
    def __init__(self):
        self.model = self.simple_model()

    def simple_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28, 1)))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_data(self):
        train_data = pd.read_csv('/Users/manvithchandra/Desktop/DeepLearning/LAB/fashion_mnist/fashion-mnist_test.csv')
        test_data = pd.read_csv('/Users/manvithchandra/Desktop/DeepLearning/LAB/fashion_mnist/fashion-mnist_test.csv')

        train_images = train_data.iloc[:, 1:].values.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        train_labels = to_categorical(train_data.iloc[:, 0].values)

        test_images = test_data.iloc[:, 1:].values.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
        test_labels = to_categorical(test_data.iloc[:, 0].values)

        return train_images, train_labels, test_images, test_labels

    def train(self, train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2):
        history = self.model.fit(
            train_images, train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )
        return history

    def evaluate(self, test_images, test_labels):
        # Evaluate the model on the test set
        test_loss, test_accuracy = self.model.evaluate(test_images, test_labels)
        print(f'Test Accuracy: {test_accuracy}')
        print(f'Test Loss: {test_loss}')

        return test_loss, test_accuracy

    def visualize(self, test_images, test_labels, num_images=5):
        # Visualize predictions on a few test images
        predictions = self.model.predict(test_images[:num_images])
        for i in range(num_images):
            plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
            plt.title(f'Actual: {test_labels[i]}, Predicted: {predictions[i].argmax()}')
            plt.show()

    def plot_history(self, history):
        # Plot training history (accuracy and loss)
        plt.figure(figsize=(12, 6))

        # Plot training & validation accuracy values
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')

        # Plot training & validation loss values
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.tight_layout()
        plt.show()
