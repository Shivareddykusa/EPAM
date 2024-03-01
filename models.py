import matplotlib.pyplot as plt
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

class DeepANN():
    def simple_model(self, input_shape=(124, 124, 3), optimizer='sgd'):
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(2, activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        return model

    def train_model(self, model_instance, train_generator, validate_generator, epochs=5):
        mhistory = model_instance.fit(train_generator, validation_data=validate_generator, epochs=epochs)
        return mhistory

    def compare_models(self, models, train_generator, validate_generator, epochs=5):
        histories = []
        plt.figure(figsize=(10, 6))

        for i, model in enumerate(models):
            history = self.train_model(model, train_generator, validate_generator, epochs=epochs)
            histories.append(history)

            # Plot training accuracy
            plt.subplot(2, 1, 1)
            plt.plot(history.history['accuracy'], label=f'Model {i + 1} Training Accuracy')

            # Plot validation accuracy
            plt.subplot(2, 1, 2)
            plt.plot(history.history['val_accuracy'], label=f'Model {i + 1} Validation Accuracy')

        plt.subplot(2, 1, 1)
        plt.title('Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.title('Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()


