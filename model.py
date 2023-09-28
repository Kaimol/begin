import numpy as np


# neural network class
class neuralNetwork():

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):
        """
        The network consists of three layers: input layer, hidden layer and output layer.
        Here defined these layers.
        :param input_nodes: dimension of input
        :param hidden_nodes: dimension of hidden nodes
        :param output_nodes: dimension of output
        :param learning_rate: the learning rate of neural network
        """
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate
        self.hidden_layer = np.random.random((1, hidden_nodes))
        self.output_layer = np.random.random((hidden_nodes, output_nodes))
        self.synapse_0 = 2 * np.random.random((self.input_nodes, self.hidden_nodes)) - 1
        self.synapse_1 = 2 * np.random.random((self.hidden_nodes, self.output_nodes)) - 1
        self.bias_hidden = np.random.randn(1, self.hidden_nodes)
        self.bias_output = np.random.randn(1, self.output_nodes)
        self.input = np.random.random((1, input_nodes))




    def sigmoid(self, x):
        x_sigmoid = 1 / (1 + np.exp((-1) * x))
        return x_sigmoid


    def forward(self, input_feature):
        """
        Forward the neural network
        :param input_feature: single input image, flattened [784, ]
        """
        self.input = input_feature.reshape(1, -1)
        hidden_in = np.dot(self.input, self.synapse_0) + self.bias_hidden
        self.hidden_layer = self.sigmoid(hidden_in).reshape(1, -1)
        output_in = np.dot(self.hidden_layer, self.synapse_1) + self.bias_output
        self.output_layer = self.sigmoid(output_in).reshape(1, -1)


    def backpropagation(self, targets_list):
        targets_list = np.array(targets_list).reshape(1, -1)
        # 计算损失
        loss_total = 0.5 * np.sum((targets_list - self.output_layer) ** 2)

        # 计算输出层和隐藏层的delta
        output_layer_delta = (self.output_layer - targets_list) * (self.output_layer * (1 - self.output_layer))
        hidden_layer_delta = output_layer_delta.dot(self.synapse_1.T) * (self.hidden_layer * (1 - self.hidden_layer))

        # 更新权重和偏置
        self.synapse_1 -= self.learning_rate * self.hidden_layer.T.dot(output_layer_delta)
        self.synapse_0 -= self.learning_rate * self.input.T.dot(hidden_layer_delta)

        self.bias_hidden -= self.learning_rate * np.sum(hidden_layer_delta, axis=0, keepdims=True)
        self.bias_output -= self.learning_rate * np.sum(output_layer_delta, axis=0, keepdims=True)

        return loss_total
