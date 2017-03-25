def main(amount_of_additional_data=2000):
    import numpy as np
    from lasagne.layers import InputLayer, DenseLayer
    from lasagne.nonlinearities import softmax, sigmoid  # , rectify
    from lasagne.updates import adam  # , nesterov_momentum
    from lasagne.objectives import categorical_crossentropy
    from nolearn.lasagne import NeuralNet

    X = np.zeros((4, 2))
    X[2:, 0] = X[::2, 1] = 1

    if 0 < amount_of_additional_data:
        np.random.seed(42)
        random_data = np.random.random_integers(0, 1, amount_of_additional_data * 2).reshape((-1, 2))
        X = np.concatenate((X, random_data)).astype(np.float32, copy=False)
        del random_data

    y = np.logical_xor(X[:, 0], X[:, 1]).ravel().astype(np.int32, copy=False)

    print('X:', X.shape, X.dtype, 'y:', y.shape, y.dtype)

    layers = [
        (InputLayer, dict(shape=(None, *X.shape[1:]))),
        (DenseLayer, dict(num_units=2, nonlinearity=sigmoid)),  # rectify, sigmoid
        (DenseLayer, dict(num_units=2, nonlinearity=softmax))
    ]

    net = NeuralNet(
        layers=layers,
        max_epochs=700,
        update=adam,  # nesterov_momentum, adam
        update_learning_rate=0.1,
        objective_loss_function=categorical_crossentropy,
        verbose=3
    )

    net.fit(X, y)


if '__main__' == __name__:
    main(amount_of_additional_data=2000)
