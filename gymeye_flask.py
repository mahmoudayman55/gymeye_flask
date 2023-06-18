# Store this code in 'app.py' file
import biceps_detection_function as mpa 
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
@app.route('/get_all_exercise', methods=['GET'])
def get_all_exercise():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT id, name, cover_url FROM exercise')
    exercises = cursor.fetchall()
    cursor.close()

    return jsonify(exercises),200

@app.route('/get_by_id/<int:exercise_id>', methods=['GET'])
@app.route('/get_by_id/<int:exercise_id>', methods=['GET'])
def get_by_id(exercise_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Get exercise information
    cursor.execute('SELECT id, name, description, video_url, cover_url FROM exercise WHERE id = %s', (exercise_id,))
    exercise = cursor.fetchone()
    
    if not exercise:
        cursor.close()
        return jsonify({'error': 'Exercise not found'}), 404
    
    # Get images linked to the exercise
    cursor.execute('SELECT image_url FROM exercise_images WHERE exercise_id = %s', (exercise_id,))
    images = cursor.fetchall()
    
    # Get errors associated with the exercise
    cursor.execute('SELECT name, description FROM exercise_error WHERE exercise_id = %s', (exercise_id,))
    errors = cursor.fetchall()
    
    cursor.close()

    exercise['images'] = [image['image_url'] for image in images]
    exercise['errors'] = errors

    return jsonify(exercise)
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



import re
import json
import MySQLdb.cursors



@app.route('/analyze_biceps', methods=['POST']) 
def analyze_video():
    # Get the video from the request 
    video = request.files['video']
    user_id= int(request.form["user_id"])
    exercise_id=int(request.form["exercise_id"])
    # Save the video to a file
    video.save('video.mp4')  
    
    # Analyze the video 
    left_arm_count, right_arm_count, left_arm_errors, right_arm_errors, prediction_probability = mpa.analyze_video('video.mp4')
#     {
#     "Lean_too_far_back": 12.743972445464976,
#     "left_arm_count": 13,
#     "left_arm_errors": {
#         "LOOSE_UPPER_ARM": 0,
#         "PEAK_CONTRACTION": 0
#     },
#     "right_arm_count": 0,
#     "right_arm_errors": {
#         "LOOSE_UPPER_ARM": 0,
#         "PEAK_CONTRACTION": 0
#     }
# }
    
    import datetime

    current_time = datetime.datetime.now()

    LooseUpperleft=left_arm_errors["LOOSE_UPPER_ARM"]
    LooseUpperRight=right_arm_errors["LOOSE_UPPER_ARM"]
    PeekRight=right_arm_errors["PEAK_CONTRACTION"]
    PeekLeft=left_arm_errors["PEAK_CONTRACTION"]
    lb= 100-prediction_probability

#save evaluation result
###   user_id,leftcounter,right counter,exercise_id
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_eveluation_results VALUES ( null,%s, %s, %s, %s, %s)', (user_id, exercise_id, left_arm_count, right_arm_count,current_time ))
    
    mysql.connection.commit()
        
#get id of inserted evaluation
    eval_id=cursor.lastrowid
    print("eval",eval_id)
#get errors of exercise from exercise_errors table
    # cursor.execute('SELECT id FROM exercise_error WHERE exercise_id = %s', (exercise_id,))
    # exercise_errors = cursor.fetchall()
#for each error insert into evaluation error result table (error_id,error_value)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (1, eval_id, LooseUpperRight,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (2, eval_id, LooseUpperleft,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (3, eval_id, PeekRight,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (4, eval_id, PeekLeft,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (5, eval_id, lb,"" ))
    mysql.connection.commit()



    # Return the analysis results 
    results = {
        'left_arm_count': left_arm_count,
        'right_arm_count': right_arm_count,
        'left_arm_errors': left_arm_errors,
        'right_arm_errors': right_arm_errors,
        'Lean_too_far_back': 100-prediction_probability
    }
    
    return jsonify(results)

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
            cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
            account = cursor.fetchone()
            if account:
                print("exist")
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
            code=401

        return jsonify({'msg': msg}),code


if __name__ == '__main__':
    app.run(host='192.168.1.55',port=5000)