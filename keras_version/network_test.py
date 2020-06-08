from keras.layers import Input
from keras.models import Model, load_model
from attention import *
from transformer import *
import numpy as np

def test_layernorm():
    BATCH_SIZE = 64
    DATA_DIM = 128

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = LayerNormalization(name='LayerNorm')(input_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)
    print(x[0, :2, :6])
    print(y[0, :2, :6])

def test_ScaledDotProductAttention():
    BATCH_SIZE = 64
    DATA_DIM = 128

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = ScaledDotProductAttention(True, True, 0.1)([input_tensor, input_tensor, input_tensor])

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)

def test_PositionEncoding():
    BATCH_SIZE = 64
    DATA_DIM = 128

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = PositionEncoding(name='pos')(input_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)

def test_PositionWiseFeedForward():
    BATCH_SIZE = 64
    DATA_DIM = 128
    INNER_DIM = 256

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = PositionWiseFeedForward(DATA_DIM, INNER_DIM, True)(input_tensor)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)

def test_MultiheadAttention():
    BATCH_SIZE = 64
    DATA_DIM = 128

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = MultiheadAttention(8, 16, 0.1, True, False, True)([input_tensor, input_tensor, input_tensor])

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)
    print(x[0, :2, :6])
    print(y[0, :2, :6])

def test_Transformer():
    BATCH_SIZE = 64
    DATA_DIM = 128
    MODEL_DIM = DATA_DIM

    transformer = Transformer(model_dim=MODEL_DIM)

    input_tensor = Input((None, DATA_DIM), name='input')
    output_tensor = transformer([input_tensor, input_tensor])

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    x = np.random.uniform(0, 1, (BATCH_SIZE, 100, DATA_DIM))
    y = model.predict(x)
    print(x.shape, '->', y.shape)
    print(x[0, :2, :6])
    print(y[0, :2, :6])


def train_on_MultiheadAttention():
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    from keras.utils import to_categorical
    from keras.layers import Embedding, Add, Dropout, Dense, GlobalAveragePooling1D
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    EPOCHS = 10
    BATCH_SIZE = 64
    MAX_LEN = 256
    VOCAB_SIZE = 5000
    LATENT_DIM = 128
    NUM_HEAD = 2

    input_tensor = Input((None, ), name='input')
    embeddings = Embedding(VOCAB_SIZE + 1, LATENT_DIM)(input_tensor)
    encodings = PositionEncoding(name='pos')(embeddings)
    encodings = Add()([embeddings, encodings])
    x = MultiheadAttention(NUM_HEAD, LATENT_DIM // NUM_HEAD, 0.1, True, False, True)([encodings, encodings, encodings])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    output_tensor = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacklist = [EarlyStopping(patience=3)]


    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=MAX_LEN, num_words=VOCAB_SIZE)
    x_train = sequence.pad_sequences(x_train, MAX_LEN)
    x_test = sequence.pad_sequences(x_test, MAX_LEN)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=callbacklist)

    test_metrics = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print("test loss=%.4f acc=%.4f" % (test_metrics[0], test_metrics[1]))


def train_on_Transformer():
    from keras.datasets import imdb
    from keras.preprocessing import sequence
    from keras.utils import to_categorical
    from keras.layers import Embedding, Add, Dropout, Dense, GlobalAveragePooling1D
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping

    EPOCHS = 10
    BATCH_SIZE = 64
    MAX_LEN = 256
    VOCAB_SIZE = 5000
    LATENT_DIM = 128
    FF_UNITS = 256

    input_tensor = Input((MAX_LEN, ), name='input')
    embeddings = Embedding(VOCAB_SIZE + 1, LATENT_DIM)(input_tensor)

    x = Transformer(model_dim=LATENT_DIM, n_heads=2, encoder_stack=2, decoder_stack=2, feed_forward_size=FF_UNITS)([embeddings, embeddings])
    x = GlobalAveragePooling1D()(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='relu')(x)
    output_tensor = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    print(model.summary())

    model.compile(optimizer=Adam(beta_1=0.9, beta_2=0.98, epsilon=1e-9),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    callbacklist = [EarlyStopping(patience=3)]


    (x_train, y_train), (x_test, y_test) = imdb.load_data(maxlen=MAX_LEN, num_words=VOCAB_SIZE)
    x_train = sequence.pad_sequences(x_train, MAX_LEN)
    x_test = sequence.pad_sequences(x_test, MAX_LEN)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.2, callbacks=callbacklist)

    test_metrics = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=1)
    print("test loss=%.4f acc=%.4f" % (test_metrics[0], test_metrics[1]))


if __name__ == '__main__':
    # train on IMDB dataset (MultiHead)
    train_on_MultiheadAttention()

    # train on IMDB dataset (Transformer)
    train_on_Transformer()
