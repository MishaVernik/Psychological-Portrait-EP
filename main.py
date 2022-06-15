from flask import Flask, jsonify
from flask import request
import pymssql

app = Flask(__name__)


def get_user_data(user_id):
    import pyodbc
    import pandas as pd
    # Some other example server values are
    # server = 'localhost\sqlexpress' # for a named instance
    # server = 'myserver,port' # to specify an alternate port
    server = 'entertainment-planner.database.windows.net'
    database = 'EntertainmentPlanner'
    username = 'nodus'
    password = 'Azure1.5.3'
    cnxn = pyodbc.connect(
        'DRIVER={SQL Server};SERVER=' + server + ';DATABASE=' + database + ';UID=' + username + ';PWD=' + password)
    cursor = cnxn.cursor()
    # select 26 rows from SQL table to insert in dataframe.
    query = "exec Get_User_Suppliers_Mood " + str(user_id) + ""
    df = pd.read_sql(query, cnxn)
    print(df.head())

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

    # result = '' - table OCEAN

    return jsonify('result')

@app.route('/')
def index():
    """ Displays the index page accessible at '/'
    """
    return Flask.render_template('index.html')


if __name__ == '__main__':

    app.debug=True
    app.run()