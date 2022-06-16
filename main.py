from urllib import response

import pymssql
from flask import Flask, jsonify
from flask import request

import pandas as pd
from flask_cors import cross_origin, CORS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app)

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BATCH_SIZE = 1
MAX_SEQUENCE = 10
us_columns = ['Fun', 'Intellectual', 'Sport', 'Relax', 'Extreme', 'Calm', 'Creative', 'Romantic']
MODEL_NAME = "ml-0001"


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
    upt_columns = ['O', 'C', 'E', 'A', 'N']

    us_key = 'UserId'
    upt_key = 'UserID'

    userselection_df = pd.read_csv('Users-SupplierFirms-Moods-List.csv')
    user_pt_df = pd.read_csv('Users-PersonalityTrait-List.csv')

    dataset_inp = [userselection_df[userselection_df[us_key] == user_id][us_columns].to_numpy()
                   for user_id in user_pt_df[upt_key].tolist()]
    dataset_inp = tf.keras.preprocessing.sequence.pad_sequences(dataset_inp, maxlen=MAX_SEQUENCE, padding='pre',
                                                                truncating='post', dtype=np.float64)
    dataset_inp = np.array(dataset_inp).astype(np.float64)

    dataset_out = [user_pt_df[user_pt_df[upt_key] == user_id][upt_columns].iloc[0].to_numpy()
                   for user_id in user_pt_df[upt_key].tolist()]
    dataset_out = np.array(dataset_out).astype(np.float64)

    return dataset_inp, dataset_out


# use this to create model
def train_save_model(file_name):
    dataset_inp, dataset_out = load_training_data()
    train_inp, test_inp, train_out, test_out = train_test_split(dataset_inp, dataset_out, test_size=0.2,
                                                                random_state=13)

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
    sel_sequences = user_df[us_columns].to_numpy()
    model_inp = tf.keras.preprocessing.sequence.pad_sequences(np.array([sel_sequences]).astype(np.float64),
                                                              maxlen=MAX_SEQUENCE, padding='pre', truncating='post',
                                                              dtype=np.float64)
    prediction = model.predict(model_inp)[0]

    us_sum = np.sum(sel_sequences, axis=0)
    us_sum = us_sum / np.amax(us_sum)

    return np.outer(prediction, us_sum).tolist()


def get_user_data(user_id):

    # Some other example server values are
    # server = 'localhost\sqlexpress' # for a named instance
    # server = 'myserver,port' # to specify an alternate port
    server = 'entertainment-planner.database.windows.net'
    database = 'EntertainmentPlanner'
    username = 'nodus'
    password = 'Azure1.5.3'
    # cnxn = pyodbc.connect(
    #     'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cnxn = pymssql.connect(host=server, user=username, password=password, database=database)

    cursor = cnxn.cursor()
    # select 26 rows from SQL table to insert in dataframe.
    query = "exec Get_User_Suppliers_Mood " + str(user_id) + ""
    df = pd.read_sql(query, cnxn)
    print(df.head())

    cursor.close()
    cnxn.close()
    return df
    # Table DATA:
    # 0 UserId
    # 1 SupplierFirmId
    # 2 Fun
    # 3 Intellectual
    # 4 Sport
    # 5 Relax
    # 6 Extreme
    # 7 Calm
    # 8 Creative
    # 9 Romantic
    #

    # row = cur.fetchone()
    # while row:
    #     print("ID=%d, Name=%s" % (row[0], row[1]))
    #     row = cur.fetchone()
    #
    # # if you call execute() with one argument, you can use % sign as usual
    # # (it loses its special meaning).
    # cur.execute("SELECT * FROM persons WHERE salesrep LIKE 'J%'")



@app.route('/users/portrait/<user_id>', methods = ['GET', 'POST', 'DELETE'])
@cross_origin(origin='*',headers='*')
def user(user_id):
    if request.method == 'GET':
        """return the information for <user_id>"""
        pass
    if request.method == 'POST':
        """modify/update the information for <user_id>"""
        # you can use <user_id>, which is a str but could
        # changed to be int or whatever you want, along
        # with your lxml knowledge to make the required
        # changes
        data = request.form # a multidict containing POST data
        pass
    if request.method == 'DELETE':
        """delete user with ID <user_id>"""
        pass


    df_user_suppliers_moods = get_user_data(user_id)

    # CALL TO THE NERUAL NETWORK
    model = create_model()
    #
    x_t, y_t = load_training_data()
    current_model = train_save_model(MODEL_NAME)
    #current_model = load_model(MODEL_NAME)
    
    model_result = model_predict(current_model, df_user_suppliers_moods)


    psy_portrait_matrix = [

        [1, 1, 1, 0, 0.5, 0, 1, 1], # Ideas
        [1, 1, 1, 0, 0.5, 0, 1, 1], # Fantasy
        [1, 1, 1, 0, 0.5, 0, 1, 1], # Aesthetic
        [1, 1, 1, 0, 0.5, 0, 1, 1], #Actions
        [1, 1, 1, 0, 0.5, 0, 1, 1], # Feelings
        [1, 1, 1, 0, 0.5, 0, 1, 1], #Values

        [0, 1, 1, 0, 0.5, 1, 0.5, 0.5],  # Competence
        [0, 1, 0, 0, 0, 1, 0, 0],  # Order
        [0, 1, 0, 0, 0, 1, 0, 0],  # Dutifulness
        [0, 1, 0, 0, 0, 1, 1, 0],  # Achieve
        [0, 1, 0, 0, 0, 1, 1, 0],  # SElf-disc
        [0, 1, 0, 1, 0, 1, 1, 0],  # Deliberation

        [1, 0.5, 1, 0, 1, 0, 0, 1],  # Gregarious
        [1, 0.5, 1, 0, 1, 0, 0, 1],  # Assertivness
        [1, 0.5, 1, 0, 1, 0, 0, 1],  # Activity
        [1, 0.5, 1, 0, 1, 0, 1, 1],  # Excetement seeking
        [1, 0.5, 1, 0, 1, 0, 0, 1],  # Positive emotions
        [1, 0.5, 1, 0, 1, 0, 0, 1],  # Warmth

        [0, 1, 1, 0, 0, 1, 0, 0],  # Trust
        [0, 1, 0, 0, 0, 1, 1, 1],  # Straghtforwardness
        [0, 1, 0, 0, 0, 1, 1, 1],  # Altruis
        [0, 1, 0, 0, 0, 1, 1, 1],  # Compliance
        [0, 1, 0, 0, 0, 1, 1, 1],  # Modesty
        [0, 1, 0, 1, 0, 1, 1, 1],  # Tender mindedness

        [0, 0, 0, 1, 0, 1, 0, 0],  # Anxiety
        [0, 0, 1, 0, 1, 0, 0, 0],  # Angry hostiliyu
        [0, 0, 0, 0, 0.5, 1, 0, 0],  # Depression
        [0, 1, 0, 1, 0.5, 1, 1, 0],  # SElf-consciousness
        [0, 0, 1, 0, 1, 0, 1, 0],  # Impulsiveness
        [0, 0, 0, 1, 0.5, 1, 1, 0],  # Vulnerability
    ]


    Openess = 0
    Openess_arr = []

    Consciousness = 0
    Consciousness_arr = []

    Extraversion = 0
    Extraversion_arr = []

    Awareness = 0
    Awareness_arr = []

    Neurotism = 0
    Neurotism_arr = []

    outer_cnt = 0
    mx = -1000
    for i in range(len(psy_portrait_matrix)):
        if (i % 6 == 0):
            outer_cnt += 1
        sum = 0
        for j in psy_portrait_matrix[i]:
            sum += j
        mx = max(mx, sum)
        if (outer_cnt == 1):
            Openess_arr.append(sum)
            Openess += sum
        if (outer_cnt == 2):
            Consciousness_arr.append(sum)
            Consciousness += sum
        if (outer_cnt == 3):
            Extraversion_arr.append(sum)
            Extraversion += sum
        if (outer_cnt == 4):
            Awareness_arr.append(sum)
            Awareness += sum
        if (outer_cnt == 5):
            Neurotism_arr.append(sum)
            Neurotism += sum

    # 1 SupplierFirmId
    # 0 Fun
    # 1 Intellectual
    # 2 Sport
    # 3 Relax
    # 4 Extreme
    # 5 Calm
    # 6 Creative
    # 7 Romantic
    #
    Openess /= mx
    Consciousness /= mx
    Extraversion /= mx
    Awareness /= mx
    Neurotism /= mx

    Openess_arr[:] = [x / mx for x in Openess_arr]
    Consciousness_arr[:] = [x / mx for x in Consciousness_arr]
    Extraversion_arr[:] = [x / mx for x in Extraversion_arr]
    Awareness_arr[:] = [x / mx for x in Awareness_arr]
    Neurotism_arr[:] = [x / mx for x in Neurotism_arr]

    print(Openess_arr)
    CHUNK_SIZE = 6
    result =  {
     "O": {
    "value": Openess / CHUNK_SIZE,
      "child": Openess_arr
    },

     "C": {
    "value": Consciousness / CHUNK_SIZE,
      "child": Consciousness_arr
    },

     "E": {
    "value": Extraversion / CHUNK_SIZE,
      "child": Extraversion_arr
    },

     "A": {
    "value": Awareness / CHUNK_SIZE,
      "child": Awareness_arr
    },

     "N": {
    "value": Neurotism / CHUNK_SIZE,
      "child": Neurotism_arr
    }
    }
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return jsonify(result)

@app.route('/')
@cross_origin()
def index():
    """ Displays the index page accessible at '/'
    """
    return Flask.render_template('index.html')


if __name__ == '__main__':
    get_user_data(103)
    app.debug=True
    app.run()
