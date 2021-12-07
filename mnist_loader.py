import copy
import pickle
import gzip
import numpy as np


def load_data():
    """
    Возвращает данные MNIST в виде кортежа, содержащего обучающие данные, данные
    проверки и тестовые данные.
    `training_data` возвращается в виде кортежа с двумя записями.
    Первая запись содержит фактические обучающие изображения. Это
    numpy ndarray с 50 000 записями. Каждая запись, в свою очередь, представляет собой
    numpy ndarray с 784 значениями, представляющими 28 * 28 = 784
    пикселей в одном MNIST-изображении.
    Вторая запись в кортеже `training_data` представляет собой numpy ndarray
    , содержащий 50 000 записей. Эти записи - всего лишь цифра
    значения (0...9) для соответствующих изображений, содержащихся в первом
    вход в кортеж.
    `validation_data` и `test_data` похожи, за исключением
    каждый содержит всего 10 000 изображений.
    """
    f = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper(size, parity=False):
    """
    Возвращает кортеж, содержащий `(training_data, validation_data,
    test_data)`. На основе `load_data`, но формат более
    удобен для использования в нашей реализации нейронных сетей.
    В частности, `training_data` - это список, содержащий 50 000 данных
    2 - кортежи `(x, y)`. "x` - это 784-мерный numpy.ndarray
    , содержащий входное изображение.
    "y" - зависит от параметра parity

    parity: False - данные будут проверяться на соответстиве числу (0, 1, 2, ..., 9)
            True - данные проверяются на четное, нечетное
    size: ограничение размера данных
    """
    # tr_d, va_d, te_d = load_data()
    tr_d_load, va_d_load, te_d_load = load_data()
    tr_d = []
    va_d = []
    te_d = []
    tr_d.append(tr_d_load[0][:size])
    tr_d.append(tr_d_load[1][:size])
    va_d.append(va_d_load[0][:size])
    va_d.append(va_d_load[1][:size])
    te_d.append(te_d_load[0][:size])
    te_d.append(te_d_load[1][:size])
    tests_data_show = copy.deepcopy(te_d)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y, parity) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_result = [vectorized_result(y, parity) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_result)

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_result = [vectorized_result(y, parity) for y in te_d[1]]
    test_data = zip(test_inputs, test_result)
    return training_data, validation_data, test_data, tests_data_show


def vectorized_result(j, parity):
    """
    parity: False - данные будут проверяться на соответстиве числу (0, 1, 2, ..., 9)
            True - данные проверяются на четное, нечетное
    Пример для чисел:
        2 на входе
        [0, 0, 1, 0, ..., 0] на выходе
    """
    if parity:
        e = np.zeros((1, 1))
        if j % 2:
            # нечетное - 0
            # четное - 1
            # e[1] = 1.0
            e[0] = 0
        else:
            e[0] = 1.0
            # e[1] = 0
    else:
        e = np.zeros((10, 1))
        e[j] = 1.0
    return e
