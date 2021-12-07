import random
import json
import sys
import matplotlib.pyplot as plt
from cv2 import cv2
from time import sleep
import numpy as np
import mnist_loader


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """
        Возвращает стоимость, связанную с выходом `а`, и желаемый результат.
        """
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Возвращает ошибку из выходного слоя"""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        """
        Возвращает стоимость, связанную с выходом `а`, и желаемый результат.
        """
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        """
        Верните дельту ошибки из выходного слоя.
        Параметр `z` не используется методом.  Он включен в
        параметры метода для того, чтобы сделать интерфейс
        согласуется с методом дельта для других классов затрат.
        """
        return (a - y)


class Network(object):

    def __init__(self, sizes, cost=CrossEntropyCost):
        """
        Список `размеры` содержит количество нейронов в
        соответствующих слоях сети.  Например, если список
        было [2, 3, 1], то это была бы трехслойная сеть,
        первый слой которой содержал бы 2 нейрона, второй слой - 3 нейрона
        и третий слой - 1 нейрон.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        self.cost = cost

    def default_weight_initializer(self):
        """
        Инициализация весов и смещений
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        """
        Переинициализация для повторного использования
        """
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a):
        """
        Возвращает вывод сети при а на входе
        """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, alpha,
            lmbda=0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """
        Мини-пакетный градиентный спуск
        """
        training_data = list(training_data)
        n = len(training_data)

        if evaluation_data:
            evaluation_data = list(evaluation_data)
            n_data = len(evaluation_data)

        # early stopping functionality:
        best_accuracy = 0
        no_accuracy_change = 0

        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch, alpha, lmbda, len(training_data))

            print("Epoch %s training complete" % j)

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("Accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=False)
                evaluation_cost.append(cost)
                print("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data, True)
                evaluation_accuracy.append(accuracy)
                print("Accuracy on evaluation data: {} / {}".format(accuracy, n_data))

        return evaluation_cost, evaluation_accuracy, \
               training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, alpha, lmbda, n):
        """
        Обновление весов и смещений сети с помощью испрльзования обратного распространения ошибки
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - alpha * (lmbda / n)) * w - (alpha / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (alpha / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Возвращает кортеж изменений весов и смещений
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def accuracy(self, data, convert=False):
        """
        Проверяет кол-во совпадений в данных и при использовании нейросети
        """
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                       for (x, y) in data]

        result_accuracy = sum(int(x == y) for (x, y) in results)
        return result_accuracy

    def total_cost(self, data, lmbda, convert=False):
        """
        Возвращет значение стоимости ошибки
        """
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            # if convert: y = vectorized_result(y)
            if convert: y = sigmoid(y)
            cost += self.cost.fn(a, y) / len(data)
            cost += 0.5 * (lmbda / len(data)) * sum(
                np.linalg.norm(w) ** 2 for w in self.weights)  # '**' - to the power of.
        return cost

    def save(self, filename):
        """
        Сохранить в файл
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()


def load(filename):
    """
    Загрузить из файла
    """
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


def vectorized_result(j):
    """
    Возвращает вектор показывающий используемое число, случай MNIST
    [0, 1, 0, ...] - число 1
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def sigmoid(z):
    """
    Сигмоида
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    Производная сигмоиды
    """
    return sigmoid(z) * (1 - sigmoid(z))


def generate_poly(a, n, noise, filename, size=100):
    x = 2 * np.random.rand(size, 1) - 1
    y = np.zeros((size, 1))
    print(np.shape(x))
    print(np.shape(y))
    if len(a) != (n + 1):
        print(f'ERROR: Length of polynomial coefficients ({len(a)}) must be the same as polynomial degree {n + 1}')
        return
    for i in range(0, n + 1):
        y = y + a[i] * np.power(x, i) + noise * (np.random.rand(size, 1) - 0.5)
    print(np.shape(x))
    data = np.hstack((x, y))
    print(1)
    np.savetxt(filename, data, delimiter=',')


def show_j_models(training_data, evaluating_data):
    epoch = 30
    mini_batch = 10
    alpha = 0.01
    net = Network([1, 100, 1], cost=QuadraticCost)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               lmbda=6,
                                               evaluation_data=None,
                                               monitor_evaluation_cost=False,
                                               monitor_training_cost=True,
                                               monitor_training_accuracy=False)


    evaluation_cost_1, evaluation_accuracy_1, \
    training_cost_1, training_accuracy_1 = net.SGD(evaluating_data, epoch, mini_batch, alpha,
                                                   lmbda=6,
                                                   evaluation_data=None,
                                                   monitor_evaluation_cost=False,
                                                   monitor_training_cost=True,
                                                   monitor_training_accuracy=False)

    plt.xlabel("Epochs")
    plt.ylabel("J(Theta)")
    plt.title(f"J model")
    plt.plot(range(epoch), training_cost, "b.", label='model train')
    plt.plot(range(epoch), training_cost_1, "y.", label='evaluation train')
    plt.legend()
    plt.show()
    plt.close()


def show_J_neurons(training_data):
    epoch = 50
    mini_batch = 10
    alpha = 0.01
    net = Network([1, 10, 1], cost=QuadraticCost)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               lmbda=5,
                                               evaluation_data=None,
                                               monitor_evaluation_cost=False,
                                               monitor_training_cost=True,
                                               monitor_training_accuracy=False)
    net_1 = Network([1, 40, 1], cost=QuadraticCost)
    _, _, \
    training_cost_1, _ = net_1.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_2 = Network([1, 80, 1], cost=QuadraticCost)
    _, _, \
    training_cost_2, _ = net_2.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_3 = Network([1, 120, 1], cost=QuadraticCost)
    _, _, \
    training_cost_3, _ = net_3.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_4 = Network([1, 10, 10, 1], cost=QuadraticCost)
    _, _, \
    training_cost_4, _ = net_4.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    plt.xlabel("Epochs")
    plt.ylabel("J(Theta)")
    plt.title(f"J model")
    plt.plot(range(epoch), training_cost, "b", label='neuron = 10')
    plt.plot(range(epoch), training_cost_1, "y", label='neuron = 40')
    plt.plot(range(epoch), training_cost_2, "g", label='neuron = 80')
    plt.plot(range(epoch), training_cost_3, "r", label='neuron = 120')
    plt.legend()
    plt.show()
    plt.close()


def show_J_layer(training_data):
    epoch = 50
    mini_batch = 10
    alpha = 0.01
    net = Network([1, 10, 1], cost=QuadraticCost)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               lmbda=5,
                                               evaluation_data=None,
                                               monitor_evaluation_cost=False,
                                               monitor_training_cost=True,
                                               monitor_training_accuracy=False)
    net_1 = Network([1, 10, 10, 1], cost=QuadraticCost)
    _, _, \
    training_cost_1, _ = net_1.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_2 = Network([1, 5, 1], cost=QuadraticCost)
    _, _, \
    training_cost_2, _ = net_2.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_3 = Network([1, 5, 5, 1], cost=QuadraticCost)
    _, _, \
    training_cost_3, _ = net_3.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    net_4 = Network([1, 10, 10, 10, 1], cost=QuadraticCost)
    _, _, \
    training_cost_4, _ = net_4.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    plt.xlabel("Epochs")
    plt.ylabel("J(Theta)")
    plt.title(f"J model")
    plt.plot(range(epoch), training_cost, "b", label='layer/neuron = 2/10')
    plt.plot(range(epoch), training_cost_1, "y", label='layer/neuron = 2/10, 3/10')
    plt.plot(range(epoch), training_cost_3, "r", label='layer/neuron = 2/10, 3/10, 4/10')
    plt.plot(range(epoch), training_cost_2, "g", label='layer/neuron = 2/5, 3/5')
    plt.legend()
    plt.show()
    plt.close()


def show_j_any(training_data):
    epoch = 1000
    mini_batch = 10
    alpha = 0.0001
    net = Network([1, 100, 1], cost=QuadraticCost)
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               lmbda=5,
                                               evaluation_data=None,
                                               monitor_evaluation_cost=False,
                                               monitor_training_cost=True,
                                               monitor_training_accuracy=False)
    net_1 = Network([1, 100, 1], cost=CrossEntropyCost)
    _, _, \
    training_cost_1, _ = net_1.SGD(training_data, epoch, mini_batch, alpha,
                                   lmbda=5,
                                   evaluation_data=None,
                                   monitor_evaluation_cost=False,
                                   monitor_training_cost=True,
                                   monitor_training_accuracy=False)

    plt.xlabel("Epochs")
    plt.ylabel("J(Theta)")
    plt.title(f"J model")
    plt.plot(range(epoch), training_cost, "b", label='QuadraticCost')
    plt.plot(range(epoch), training_cost_1, "y", label='CrossEntropyCost')
    plt.legend()
    plt.show()
    plt.close()


def main_poly():
    # degree = 3
    # min_coeff = 1
    # max_coeff = 10
    # array_coefficients = [1] * (degree + 1)
    # for coeff in range(degree + 1):
    #     array_coefficients[coeff] = random.randint(min_coeff, max_coeff)
    # generate_poly(array_coefficients, degree, 0.5, "polynomial.csv")

    with open("polynomial.csv", 'r') as _file:
        data = np.loadtxt(_file, delimiter=',')
    X, y = data[:, 0], data[:, 1]
    percent = 60
    X_train = X[:percent]
    Y_train = y[:percent]
    X_validation = X[percent:]
    Y_validation = y[percent:]
    X_train = X_train.reshape(len(X_train), 1)[0]
    Y_train = Y_train.reshape(len(Y_train), 1)[0]
    X_validation = X_validation.reshape(len(X_validation), 1)[0]
    Y_validation = Y_validation.reshape(len(Y_validation), 1)[0]
    training_data = [[np.array([X_train]), np.array([Y_train])]]
    evaluating_data = [[np.array([X_validation]), np.array([Y_validation])]]
    # test_data = [X_validation, Y_validation]

    # show_j_models(training_data, evaluating_data)
    # show_J_neurons(training_data)
    # show_J_layer(training_data)
    show_j_any(training_data)


def main_num():
    pixel = 784
    layer_2 = 30
    layer_3 = 10
    net = Network([pixel, layer_2, layer_3], cost=CrossEntropyCost)

    # max 50`000
    size_data = 10000
    parity = True
    training_data, validation_data, test_data, test_data_show = \
        mnist_loader.load_data_wrapper(size_data, parity=parity)

    epoch = 30
    mini_batch = 10
    alpha = 0.1
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha)

    test_data = list(test_data)
    results = [(net.feedforward(x), y) for (x, y) in test_data]
    i = 0
    for x, y in results:
        if x[0][0] > 0.5:
            print('Четное!')
        else:
            print('Не четное!')

        first_image = np.array(test_data_show[0][i], dtype='float')
        i += 1
        pixels = first_image.reshape((28, 28))
        plt.imshow(pixels, cmap='gray')
        plt.show()
        plt.close()

def main():

    pixel = 784
    layer_2 = 30
    layer_3 = 10
    net = Network([pixel, layer_2, layer_3], cost=CrossEntropyCost)

    # max 50`000
    size_data = 1000
    parity = False
    training_data, validation_data, test_data, test_data_show = \
        mnist_loader.load_data_wrapper(size_data, parity=parity)

    epoch = 30
    mini_batch = 10
    alpha = 4
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               evaluation_data=test_data,
                                               monitor_training_cost=True,
                                               monitor_evaluation_cost=True,
                                               monitor_training_accuracy=True,
                                               monitor_evaluation_accuracy=True)

    plt.xlabel("Epochs")
    plt.ylabel("J(Theta)")
    plt.title(f"J model")
    plt.plot(range(epoch), training_cost, "b", label='model train')
    plt.plot(range(epoch), evaluation_cost, "r", label='evaluation_accuracy')
    plt.legend()
    plt.show()
    plt.close()

    for index, accuracy in enumerate(training_accuracy):
        training_accuracy[index] = accuracy / size_data
    for index, accuracy in enumerate(evaluation_accuracy):
        evaluation_accuracy[index] = accuracy / size_data
    plt.xlabel("Epochs")
    plt.ylabel("accuracy")
    plt.title(f"training accuracy")
    plt.plot(range(epoch), training_accuracy, "b", label='accuracy train')
    plt.plot(range(epoch),  evaluation_accuracy, "r", label='evaluation_accuracy')
    plt.legend()
    plt.show()
    plt.close()


def main_numbers():
    pixel = 784
    layer_2 = 30
    layer_3 = 1
    net = Network([pixel, layer_2, layer_3], cost=CrossEntropyCost)

    # max 50`000
    size_data = 1000
    parity = True
    training_data, validation_data, test_data, test_data_show = \
        mnist_loader.load_data_wrapper(size_data, parity=parity)

    epoch = 30
    mini_batch = 10
    alpha = 3
    evaluation_cost, evaluation_accuracy, \
    training_cost, training_accuracy = net.SGD(training_data, epoch, mini_batch, alpha,
                                               evaluation_data=None,
                                               monitor_evaluation_accuracy=False,
                                               monitor_training_cost=False,
                                               monitor_training_accuracy=False)

    # -----------------------------------------------
    letters = letters_extract("paint.bmp")
    print_mass = np.zeros(784)
    ind = 5 * 28 + 4
    for index_h, row_h in enumerate(letters[-1][2]):
        for index_w, row_w in enumerate(row_h):
            row_w_f = abs(1 - row_w / 255)  # fixed
            print_mass[ind] = row_w_f
            ind += 1
        ind += 8
    # -----------------------------------------------

    results = net.feedforward(print_mass.reshape(-1, 1))
    if results[0][0] > 0.5:
        print('Четное!')
    else:
        print('Не четное!')
    #
    # print(results[0][0])
    # cv2.imshow("0", letters[0][2])
    # cv2.imshow("1", letters[1][2])
    # показать последнее число
    first_image = np.array(print_mass, dtype='float')
    pixels = first_image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    plt.close()


def letters_extract(image_file: str, out_size=28):
    """
    Нахождение и приобразование ихображения
    """
    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    # Get contours
    contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    output = img.copy()

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            cv2.rectangle(output, (x, y), (x + w, y + h), (70, 0, 0), 1)
            letter_crop = gray[y:y + h, x:x + w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop
            letters.append((x, w, cv2.resize(letter_square, (out_size-8, out_size-8), interpolation=cv2.INTER_AREA)))
    letters.sort(key=lambda x: x[0], reverse=False)

    return letters


if __name__ == "__main__":
    main()
    # main_poly()
    # main_num()
    # main_numbers()
