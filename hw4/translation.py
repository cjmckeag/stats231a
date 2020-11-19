from abc import ABC

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

# Download the file
path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"


class GetData:
    """
    Class for downloading and cleaning data.

    Methods
    -------
    unicode_to_ascii(s):
        Converts any unicode characters to ASCII.
    preprocess_sentence(self, w):
        Converts any unicode to ascii, creates a space between a word and the punctuation following it, replaces
         everything with space except (a-z, A-Z, ".", "?", "!", ","), and adds a start and an end token to the sentence.
    create_dataset(self, path, num_examples):
        Opens path to file, preprocesses all lines in file.
    tokenize(lang):
        Returns tokenizer for a specific language.
    load_dataset(self, num_examples=None):
        Creates dataset and tokenizes it.
    """

    def __init__(self, path_to_file):
        """
        Constructs necessary parameter for GetData.
        :param path_to_file: string of os path to data file
        """
        self.path_to_file = path_to_file

    def __call__(self):
        """
        Loads dataset and splits into training set.
        :return: input_tensor_train, target_tensor_train, max_length_targ, max_length_inp, inp_lang, targ_lang
        """
        # Try experimenting with the size of that dataset
        num_examples = 30000
        input_tensor, target_tensor, inp_lang, targ_lang = self.load_dataset(num_examples)

        # Calculate max_length of the target tensors
        max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

        # Creating training and validation sets using an 80-20 split
        input_tensor_train, _, target_tensor_train, _ = train_test_split(input_tensor, target_tensor, test_size=0.2)

        return input_tensor_train, target_tensor_train, max_length_targ, max_length_inp, inp_lang, targ_lang

    # Converts the unicode file to ascii
    @staticmethod
    def unicode_to_ascii(s):
        """
        Converts any unicode characters to ASCII.
        :param s: string to search
        :return: s with unicode replaced w/ ascii
        """
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        """
        Converts any unicode to ascii, creates a space between a word and the punctuation following it, replaces
         everything with space except (a-z, A-Z, ".", "?", "!", ","), and adds a start and an end token to the sentence.
        :param w: sentence to preprocess
        :return: cleaned sentence
        """
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    # 1. Remove the accents
    # 2. Clean the sentences
    # 3. Return word pairs in the format: [ENGLISH, SPANISH]
    def create_dataset(self, path, num_examples):
        """
        Opens path to file, preprocesses all lines in file.
        :param path: path to file
        :param num_examples: number of lines to read
        :return: word pairs
        """
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]

        return zip(*word_pairs)

    @staticmethod
    def tokenize(lang):
        """
        Returns tokenizer for a specific language.
        :param lang: which language
        :return: tensor, lang_tokenizer
        """
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='')
        lang_tokenizer.fit_on_texts(lang)

        tensor = lang_tokenizer.texts_to_sequences(lang)

        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, num_examples=None):
        """
        Creates dataset and tokenizes it.
        :param num_examples: number of lines to read
        :return: input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
        """
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(self.path_to_file, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


class Encoder(tf.keras.Model, ABC):
    """
    Builds the encoder part of the model. The encoder is applied to the source language. It is an RNN that uses zero
     vectors as its starting states. The encoder is used to build a "thought" vector, a sequence of numbers that
      represents the sentence meaning.
    Methods
    -------
    """

    def __init__(self, vocab_size, embedding_dim=256, enc_units=1024, batch_size=64):
        """
        Constructs necessary parameters for Encoder.
        :param vocab_size: size of vocabulary.
        :param embedding_dim: dimension of layer embedding
        :param enc_units: dimension of output space
        :param batch_size: size of each training batch
        """
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, x, hidden):
        """
        Performs building of encoder.
        :param x: input
        :param hidden: encoder hidden layer
        :return: output, state
        """
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        """
        Initializes hidden state
        :return: vector of zeros
        """
        return tf.zeros((self.batch_size, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
    """
    Implements attention mechanism which establishes direct short-cut connections between the target and the source by
     paying attention to relevant source content as we translate. Returns context_vector, attention_weights. Inherits
     keras layer.
    """

    def __init__(self, units):
        """
        Constructs necessary parameters for BahdanauAttention.
        :param units: num units for dense layers
        """
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values):
        """
        Broadcasts addition along the time axis, calculates the score, computes attention_weights, computes context
         vector.
        :param query: inputted hidden layer
        :param values: encoder output
        :return: context_vector, attention_weights
        """
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model, ABC):
    """
    Builds the decoder of the model. The decoder processes the sentence vector to emit a translation. It is an RNN. It
     processes the target sentence while predicting the next words.
    """

    def __init__(self, vocab_size, embedding_dim=256, dec_units=1024, batch_size=64):
        """
        Constructs necessary parameters for Decoder.
        :param vocab_size: size of vocabulary
        :param embedding_dim: dimension of embedding layer
        :param dec_units: dimension of output space
        :param batch_size: size of training batch
        """
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def __call__(self, x, hidden, enc_output):
        """
        Gets context vector and attention weights, passes input through embedding, passes concatenated vector to the
         GRU, and passes it through dense layer.
        :param x: decoder input
        :param hidden: decoder hidden layer
        :param enc_output: encoder output
        :return: decoded x, state, attention_weights
        """
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class Translation:
    """
    Trains a sequence to sequence model for Spanish to English translation.

    Methods
    -------
    loss_function(self, real, pred):
        Provides sparse categorical cross-entropy loss given target and predictions.
    train_step(self, inp, targ, enc_hidden):
        Performs single step of training. Gets encoder output and hidden layer, feeds target as the next input, passes
         encoder output to decoder, computes loss, then updates optimizer with gradients.
    train(self):
        Performs entire training of encoder-decoder model.
    evaluate(self, sentence):
        Similar to the training loop, except we don't use teacher forcing here. The input to the decoder at each time
         step is its previous predictions along with the hidden state and the encoder output.
    plot_attention(attention, sentence, predicted_sentence):
        Utility function that plots the attention weights.
    translate(self, sentence):
        Translates a sentence from target to source and plots attention plot.
    """

    def __init__(self, path_to_file, train_size=5000, epochs=10, units=1024, batch_size=64):
        """
        Constructs necessary parameters for Translation.
        :param path_to_file:
        :param train_size: number of examples to train on
        :param epochs: number of training epochs
        :param units: dim of hidden layer
        :param batch_size: size of training batches
        """
        # get data
        get_data = GetData(path_to_file)
        input_tensor_train, target_tensor_train, self.max_length_targ, \
        self.max_length_inp, self.inp_lang, self.targ_lang = get_data()

        input_tensor_train = input_tensor_train[:train_size]
        target_tensor_train = target_tensor_train[:train_size]

        buffer_size = len(input_tensor_train)
        self.batch_size = batch_size
        self.steps_per_epoch = len(input_tensor_train) // self.batch_size
        self.vocab_inp_size = len(self.inp_lang.word_index) + 1
        self.vocab_tar_size = len(self.targ_lang.word_index) + 1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(buffer_size)
        self.dataset = dataset.batch(self.batch_size, drop_remainder=True)
        # training params
        self.epochs = epochs
        self.units = units
        # initialize models
        self.encoder = Encoder(self.vocab_inp_size, enc_units=self.units)
        self.decoder = Decoder(self.vocab_tar_size, dec_units=self.units)
        # optimizer and loss
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def loss_function(self, real, pred):
        """
        Provides sparse categorical cross-entropy loss given target and predictions.
        :param real: true target
        :param pred: predictions
        :return: loss
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = self.loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        """
        Performs single step of training. Gets encoder output and hidden layer, feeds target as the next input, passes
         encoder output to decoder, computes loss, then updates optimizer with gradients.
        :param inp: example from dataset
        :param targ: true target
        :param enc_hidden: encoder hidden layer
        :return: batch loss
        """
        loss = 0

        with tf.GradientTape() as tape:
            # pass the input through the encoder, which returns encoder output and encoder hidden state
            enc_output, enc_hidden = self.encoder(inp, enc_hidden)

            dec_hidden = enc_hidden

            dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']] * self.batch_size, 1)

            # Teacher forcing - feeding the target as the next input
            for t in range(1, targ.shape[1]):
                # passing enc_output to the decoder, returns predictions and decoder hidden state
                predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

                loss += self.loss_function(targ[:, t], predictions)

                # using teacher forcing to decide the next input to the decoder
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables

        # calculate the gradients
        gradients = tape.gradient(loss, variables)

        # apply gradients to optimizer and backpropagate
        self.optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    def train(self):
        """
        Performs entire training of encoder-decoder model.
        :return: none
        """
        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0

            for (batch, (inp, targ)) in enumerate(self.dataset.take(self.steps_per_epoch)):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 10 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                                 batch,
                                                                 batch_loss.numpy()))

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / self.steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    def evaluate(self, sentence):
        """
        Similar to the training loop, except we don't use teacher forcing here. The input to the decoder at each time
         step is its previous predictions along with the hidden state and the encoder output.
        :param sentence:
        :return:
        """
        data_gen = GetData(path_to_file)
        attention_plot = np.zeros((self.max_length_targ, self.max_length_inp))

        sentence = data_gen.preprocess_sentence(sentence)

        inputs = [self.inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                               maxlen=self.max_length_inp,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)

        result = ''

        hidden = [tf.zeros((1, self.units))]
        enc_out, enc_hidden = self.encoder(inputs, hidden)

        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.targ_lang.word_index['<start>']], 0)

        for t in range(self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input,
                                                                      dec_hidden,
                                                                      enc_out)

            # storing the attention weights to plot later on
            attention_weights = tf.reshape(attention_weights, (-1,))
            # store the attention weights for every time step
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()

            result += self.targ_lang.index_word[predicted_id] + ' '

            # stop predicting when the model predicts the end token
            if self.targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence, attention_plot

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims([predicted_id], 0)

        return result, sentence, attention_plot

    # function for plotting the attention weights
    @staticmethod
    def plot_attention(attention, sentence, predicted_sentence):
        """
        Utility function that plots the attention weights.
        :param attention: attention weights
        :param sentence: true sentence
        :param predicted_sentence: predicted sentence
        :return: plot
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        ax.matshow(attention, cmap='viridis')

        fontdict = {'fontsize': 14}

        ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        plt.show()

    def translate(self, sentence):
        """
        Translates a sentence from target to source and plots attention plot.
        :param sentence: sentence to translate
        :return: translated sentence and attention plot
        """
        result, sentence, attention_plot = self.evaluate(sentence)

        print('Input: %s' % sentence)
        print('Predicted translation: {}'.format(result))

        attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
        self.plot_attention(attention_plot, sentence.split(' '), result.split(' '))
