import contractions
import pandas as pd
import numpy as np
from keras.layers import Bidirectional, Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Attention
from keras.models import Model
from keras import backend as K
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Clearing any existing Keras session
K.clear_session()
# Loading the dataset
data = pd.read_csv('data/final.csv', encoding='latin-1')
# Creating a set of English stopwords
StopWords = set(stopwords.words('english'))


def preprocess(text):
    # Ensure the text is a string
    if not isinstance(text, str):
        return ""
    # Convert text to lower case and remove text within parentheses
    new_text = text.lower()
    new_text = re.sub(r'\([^)]*\)', '', new_text)
    # Remove quotation marks
    new_text = re.sub('"', '', new_text)
    # Expand contractions
    new_text = contractions.fix(new_text)
    # Remove possessive endings
    new_text = re.sub(r"'s\b", "", new_text)
    # Keep only alphabetic characters
    new_text = re.sub("[^a-zA-Z]", " ", new_text)
    # Remove stopwords and short words (assuming StopWords is already defined)
    new_text = ' '.join([word for word in new_text.split() if word not in StopWords and len(word) >= 3])
    return new_text


# Applying the preprocessing function to the dataset
data['processed_text'] = data['text'].apply(preprocess)
data['processed_summary'] = data['summary'].apply(preprocess)

# Ensure all values are strings after preprocessing
data['processed_text'] = data['processed_text'].astype(str)
data['processed_summary'] = data['processed_summary'].astype(str)

# Adding START and END tokens to summaries
data['processed_summary'] = data['processed_summary'].apply(lambda x: '<START> ' + x + ' <END>')

# Filter out rows where processed_text or processed_summary are effectively empty
data = data[data['processed_text'].str.strip() != '']
data = data[data['processed_summary'].str.strip() != '<START>  <END>']

# Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['processed_text'], data['processed_summary'], test_size=0.2,
                                                    random_state=0)

# Determining the maximum length for text and summaries based on the 95th percentile of their lengths
max_len_text = int(np.quantile([len(t.split()) for t in X_train], 0.95))
max_len_summary = int(np.quantile([len(s.split()) for s in y_train], 0.95))

# Ensuring that y_train and y_test are strings (this is crucial)
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Tokenizing the text and summaries
text_tokenizer = Tokenizer()
headline_tokenizer = Tokenizer(filters='')
text_tokenizer.fit_on_texts(list(X_train))
headline_tokenizer.fit_on_texts(list(y_train) + list(y_test))

# Adding special tokens to the headline tokenizer if not already present
special_tokens = ['<START>', '<END>']
for token in special_tokens:
    if token not in headline_tokenizer.word_index:
        headline_tokenizer.word_index[token] = len(headline_tokenizer.word_index) + 1

# Padding sequences to ensure uniform length
x_train_pad = pad_sequences(text_tokenizer.texts_to_sequences(X_train), maxlen=max_len_text, padding='post')
x_test_pad = pad_sequences(text_tokenizer.texts_to_sequences(X_test), maxlen=max_len_text, padding='post')
y_train_pad = pad_sequences(headline_tokenizer.texts_to_sequences(y_train), maxlen=max_len_summary, padding='post')
y_test_pad = pad_sequences(headline_tokenizer.texts_to_sequences(y_test), maxlen=max_len_summary, padding='post')

# Calculating vocabulary sizes for the text and summaries
news_vocab_size = len(text_tokenizer.word_index) + 1
headline_vocab_size = len(headline_tokenizer.word_index) + 1

# Setting model parameters
embedding_dim = 250
latent_dim = 250

# Building the encoder part of the model
encoder_inputs = Input(shape=(max_len_text,))
encoder_embedding = Embedding(news_vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_state=True, return_sequences=True, dropout=0.4))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Building the decoder part of the model
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(headline_vocab_size, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim * 2, return_sequences=True, return_state=True, dropout=0.4)
decoder_lstm_output, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
# Adding the attention layer
attn_layer = Attention()
attn_out = attn_layer([decoder_lstm_output, encoder_outputs])
# Concatenating the output of the decoder LSTM and the attention layer
decoder_concat = Concatenate(axis=-1)([decoder_lstm_output, attn_out])
# Adding a time-distributed dense layer for generating the final output
decoder_dense = TimeDistributed(Dense(headline_vocab_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat)
# Compiling the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Function to check if any label is out of the range of the vocabulary size
def check_label_range(data, vocab_size):
    if np.max(data) >= vocab_size:
        raise ValueError("Label index exceeds vocabulary size. Check tokenization and vocabulary size.")


# Running the check on the training and test sets
check_label_range(y_train_pad, headline_vocab_size)
check_label_range(y_test_pad, headline_vocab_size)

# Training the model
history = model.fit([x_train_pad, y_train_pad[:, :-1]],
                    y_train_pad.reshape(y_train_pad.shape[0], y_train_pad.shape[1], 1)[:, 1:],
                    epochs=50, batch_size=256,
                    validation_data=([x_test_pad, y_test_pad[:, :-1]],
                                     y_test_pad.reshape(y_test_pad.shape[0], y_test_pad.shape[1], 1)[:, 1:]))

# Building the inference models for encoder and decoder
# Encoder model
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])
# Decoder model for inference
decoder_embedding_inf = Embedding(headline_vocab_size, embedding_dim)  # Separate embedding layer for inference
decoder_inputs_inf = Input(shape=(None,))
decoder_emb_inf = decoder_embedding_inf(decoder_inputs_inf)
decoder_states_inputs = [Input(shape=(latent_dim * 2,)), Input(shape=(latent_dim * 2,))]
decoder_out, state_h_inf, state_c_inf = decoder_lstm(decoder_emb_inf, initial_state=decoder_states_inputs)
decoder_states = [state_h_inf, state_c_inf]

# Attention layer for inference
decoder_hidden_state_input = Input(shape=(max_len_text, latent_dim * 2))
attn_out_inf = attn_layer([decoder_out, decoder_hidden_state_input])
decoder_inf_concat = Concatenate(axis=-1)([decoder_out, attn_out_inf])
# Final output layer for inference
decoder_outputs_inf = decoder_dense(decoder_inf_concat)
# Compiling the inference model
decoder_model = Model(
    [decoder_inputs_inf, decoder_hidden_state_input] + decoder_states_inputs,
    [decoder_outputs_inf] + decoder_states
)


# Function to decode a sequence into a summary
def decode_sequence(input_seq):
    # Get the encoder's output and states
    encoder_out, state_h, state_c = encoder_model.predict(input_seq)
    states_value = [state_h, state_c]
    # Initialize the sequence with the '<START>' token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = headline_tokenizer.word_index['<START>']
    # Decoding loop
    decoded_sentence = ''
    stop_condition = False
    while not stop_condition:
        # Predict the next word
        output_tokens, h, c = decoder_model.predict([target_seq, encoder_out] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = headline_tokenizer.index_word.get(sampled_token_index, '')
        # Check for the stop condition
        if sampled_word == '<END>' or len(decoded_sentence) > max_len_summary:
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_word
            # Update the target sequence and states
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = sampled_token_index
            states_value = [h, c]
    return decoded_sentence.strip()


# Testing the model's predictions
for i in range(10):
    print('Actual Summary:', y_test.iloc[i])
    print('Predicted Summary:', decode_sequence(x_test_pad[i].reshape(1, max_len_text)))
