from Model import *

if __name__ == '__main__':
    model = Model()
    layer1 = Dense((5, 4), Relu(), XavierInit())
    model.add_layer(layer1)
    softmax = Softmax()
    layer2 = Dense((4, 3), softmax, XavierInit())
    model.add_layer(layer2)
    model.add_loss(CrossEntropyLossWithLogits(softmax))

    x = np.asarray([1, 2, 3, 4, 5])
    y = [1, 0, 0]
    model.train(x, y)
    print(y)
    print(sum(y))
