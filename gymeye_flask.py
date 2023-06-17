# Store this code in 'app.py' file

from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import json
import MySQLdb.cursors
import re
import jwt

app = Flask(__name__)


app.secret_key = 'your secret key'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Mm24681012#'
app.config['MYSQL_DB'] = 'gymeye'

mysql = MySQL(app)

@app.route('/')

@app.route('/login', methods=['POST'])
def login():
    try:
        data = json.loads(request.data.decode('utf-8'))
        email = data.get("email")
        password = data.get("password")

        if email and password:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE email = % s AND password = % s', (email, password))
            account = cursor.fetchone()

            if account:
                user_info = {
                    'id': account['id'],
                    'name': account['name'],
                    'email': account['email'],
                    'gender': account['gender'],
                    'age': account['age']
                }

                token = jwt.encode(user_info, app.config['SECRET_KEY'], algorithm='HS256')

                return jsonify({'token': token}),200
            else:
                return jsonify({'msg': 'Incorrect email or password'}),401
        else:
            return jsonify({'msg': 'Please provide email and password'}),400
    except: 
        return jsonify({'msg': "server error"}),502
@app.route('/logout')
def logout():
	session.pop('loggedin', None)
	session.pop('id', None)
	session.pop('username', None)
	return redirect(url_for('login'))

import re
import json
import MySQLdb.cursors

@app.route('/register', methods=['POST'])
def register():

        data = json.loads(request.data.decode('utf-8'))

        name = data.get('name')
        password = data.get('password')
        email = data.get('email')
        age = data.get('age')
        gender = data.get('gender')
        code=200
        if name and password and email and age and gender:
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM users WHERE name = %s', (name,))
            account = cursor.fetchone()
            if account:
                msg = 'Account already exists!'
                code=401
            elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
                msg = 'Invalid email address!'
                code=401
            elif not re.match(r'[A-Za-z0-9]+', name):
                msg = 'Username must contain only characters and numbers!'
                code=401
            else:
                cursor.execute('INSERT INTO users VALUES ( null,%s, %s, %s, %s, %s)', (name, email, password, gender,age ))
                mysql.connection.commit()
                msg = 'You have successfully registered!'
        else:
            msg = 'Please fill out the form!'

        return jsonify({'msg': msg}),code


if __name__ == '__main__':
    app.run(host='192.168.8.108',port=5000)