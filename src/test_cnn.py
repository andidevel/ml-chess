import chess_nn
from data_config import DataConfig


if __name__ == '__main__':
    sub_set = 100
    print('Reading data samples...')
    data = DataConfig('data.config')
    train_data, train_labels = data.get_train()
    test_data, test_labels = data.get_test()
    data.h5data.close()
    #test_labels = test_labels[:, 0].astype(int).reshape(-1,1)
    #train_labels = train_labels[:, 0].astype(int).reshape(-1,1)
    print('train_data shape: ', train_data.shape)
    print('test_data shape: ', test_data.shape)
    print('Getting a train samples subset ({} samples)...'.format(sub_set))
    x_train, y_train = chess_nn.random_sampling(train_data, train_labels, sub_set)
    print('x_train shape: ', x_train.shape)
    print('y_train shape: ', y_train.shape)
    print('Creating CNN model...')
    chess_model = chess_nn.make_model()
    print('Training...')
    history = chess_nn.train(chess_model, x_train, y_train, test_data, test_labels)
    print('Evaluating...')
    score = chess_nn.score(chess_model, test_data, test_labels)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Done.')
