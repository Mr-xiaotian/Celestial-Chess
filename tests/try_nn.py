import numpy as np
import numpy.random as nr
import scipy.special as sp
from mnist import MNIST
from CelestialVault.instances import NeuralNetwork, ActivationLayer, DenseLayer, SequentialNetwork

# 加载MNIST数据
mndata = MNIST(r'G:\Project\Celestial-Chess\tests\mnist')
images, labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

# 将数据转换为 numpy 数组并进行标准化
images = np.array(images) / 255.0
test_images = np.array(test_images) / 255.0

# 将标签转换为独热编码
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

labels = one_hot_encode(labels)
test_labels = one_hot_encode(test_labels)

net = SequentialNetwork()

net.add(DenseLayer(784, 392))
net.add(ActivationLayer(392))
net.add(DenseLayer(392, 196))
net.add(ActivationLayer(196))
net.add(DenseLayer(196, 10))
net.add(ActivationLayer(10))

# 将训练数据和标签打包
train_data = list(zip(images, labels))
test_data = list(zip(test_images, test_labels))

# 设置训练参数
epochs = 10
mini_batch_size = 32
learning_rate = 0.1

# 训练神经网络
net.train(train_data[:1000], epochs, mini_batch_size, learning_rate, test_data[:100])