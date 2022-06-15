import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1
MAX_SEQUENCE = 10
us_columns = ['Fun','Intellectual','Sport','Relax','Extreme','Calm','Creative','Romantic']

def create_model():
    alg_model = tf.keras.Sequential()
    alg_model.add(tf.keras.layers.Input(shape=(None, 8)))
    
    alg_model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, return_sequences=False, dropout=0.1, recurrent_dropout=0.1)
    ))
    
    alg_model.add(tf.keras.layers.Dense(128, activation='linear'))
    alg_model.add(tf.keras.layers.ELU())

    alg_model.add(tf.keras.layers.Dropout(0.2))

    alg_model.add(tf.keras.layers.Dense(64, activation='linear'))
    alg_model.add(tf.keras.layers.ELU())

    alg_model.add(tf.keras.layers.Dropout(0.1))

    alg_model.add(tf.keras.layers.Dense(5, activation='linear'))
    alg_model.add(tf.keras.layers.Activation('sigmoid'))
    
    alg_model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=['accuracy'])

    return alg_model
    
def load_training_data():
    upt_columns = ['O','C','E','A','N']

    us_key = 'UserId'
    upt_key = 'UserID'
    
    userselection_df = pd.read_csv('Users-SupplierFirms-Moods-List.csv')
    user_pt_df = pd.read_csv('Users-PersonalityTrait-List.csv')

    dataset_inp = [ userselection_df[userselection_df[us_key] == user_id][us_columns].to_numpy()
        for user_id in user_pt_df[upt_key].tolist()]
    dataset_inp = tf.keras.preprocessing.sequence.pad_sequences(dataset_inp, maxlen=MAX_SEQUENCE, padding='pre', truncating='post', dtype=np.float64)
    dataset_inp = np.array(dataset_inp).astype(np.float64)

    dataset_out = [ user_pt_df[user_pt_df[upt_key] == user_id][upt_columns].iloc[0].to_numpy()
        for user_id in user_pt_df[upt_key].tolist()]
    dataset_out = np.array(dataset_out).astype(np.float64)

    return dataset_inp, dataset_out

# use this to create model
def train_save_model(file_name):
    dataset_inp, dataset_out = load_training_data()
    train_inp, test_inp, train_out, test_out = train_test_split(dataset_inp, dataset_out, test_size=0.2, random_state=13)

    pers_model = create_model()

    pres_train_hist = pers_model.fit(
        train_inp, train_out, batch_size=BATCH_SIZE, epochs=40,
        verbose=0, validation_data=(test_inp, test_out))
    pers_model.save(file_name)

    return pers_model

# use this to load created model
def load_model(file_name):
    return tf.keras.models.load_model(file_name)

# use this to obtain user personality trait table
def model_predict(model, user_df):
    model_inp = tf.keras.preprocessing.sequence.pad_sequences(user_df[us_columns].to_numpy().astype(np.float64),
        maxlen=MAX_SEQUENCE, padding='pre', truncating='post', dtype=np.float64)
    prediction = pers_model.predict(model_inp)[0]

    us_sum = np.sum(pers_traits_seq, axis=0)
    us_sum = us_sum / np.amax(us_sum)

    return np.outer(pers_traits_seq, us_sum)
