from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_test)

def main():
    print('main')


if __name__ == "__main__":
    main()