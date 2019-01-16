from Model import *

if __name__ == '__main__':
    layer1 = Dense((3, 2), Relu(), XavierInit())
    print(layer1.W)
    print(layer1.b)

    x = [1, 2, 3]
    x = np.asarray(x)
    print(x)
    print(layer1.W @ x)
    print(layer1.forward_pass(x))