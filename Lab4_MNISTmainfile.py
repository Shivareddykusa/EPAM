from Lab4_annclassfile import Annmodel


def main():
    model_instance = Annmodel()

    train_images, train_labels, test_images, test_labels = model_instance.load_data()

    history = model_instance.train(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

    test_loss, test_accuracy = model_instance.evaluate(test_images, test_labels)

    model_instance.visualize(test_images, test_labels, num_images=5)

    model_instance.plot_history(history)


if __name__ == "__main__":
    main()
