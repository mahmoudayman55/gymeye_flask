# Store this code in 'app.py' file
import biceps.biceps_detection_function as mpa 
import barbell_row.detection_function as barbell
import squat.squat_detection as squat
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import json
import MySQLdb.cursors
import re
import jwt
import re
import json
import MySQLdb.cursors
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



###########################BMI##################
#############save bmi
# Create a new route to handle BMI data storage
@app.route('/bmi', methods=['POST'])
def bmi():
    # Get the data from the POST request
    data = json.loads(request.data.decode('utf-8'))

    # Extract the data and store it in variables
    user_id = data['user_id']
    age = data    ['age']
    weight = data ['weight']
    water = data['water']
    protein = data['protein']
    fat = data['fat']
    gender = data['gender']
    daily_activity_level = data['daily_activity_level']
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    # Create a cursor object to interact with the database

    # Execute the SQL query to insert the data into the table
    sql_query = "INSERT INTO bmi (user_id, age, weight, water, protein, fat, daily_activity_level,gender) VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"
    values = (user_id, age, weight, water, protein, fat, daily_activity_level,gender)
    cursor.execute(sql_query, values)

    # Commit the changes to the database
    mysql.connection.commit()

    # Close the cursor and return a success message
    cursor.close()
    return jsonify({'message': 'BMI data saved successfully'}), 200



############get all bmi
@app.route('/get_all_bmi', methods=['GET'])
def get_all_bmi():
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT id, user_id, age, weight, water, protein, fat, daily_activity_level, gender FROM bmi ')
    bmis = cursor.fetchall()
    cursor.close()

    return jsonify(bmis),200



############get bmi by id
@app.route('/bmi/<int:bmi_id>', methods=['GET'])
def get_bmi(bmi_id):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    # Get BMI information
    cursor.execute('SELECT id, user_id, age, weight, water, protein, fat, daily_activity_level FROM bmi WHERE id = %s', (bmi_id,))
    bmi = cursor.fetchone()
    
    if not bmi:
        cursor.close()
        return jsonify({'error': 'BMI not found'}), 404
    
    cursor.close()

    return jsonify(bmi)


########################################



@app.route('/evaluate_video', methods=['Post'])
def test():
    id=int(request.form["exercise_id"])
    if id ==1:
        return analyze_biceps()
    if id ==2:
      return  analyze_barbell_row()
    if id==3:
       return analyze_squat()


@app.route('/analyze_barbell_row', methods=['POST']) 
def analyze_barbell_row():
    # Get the video from the request 
    video = request.files['video']
    user_id= int(request.form["user_id"])
    exercise_id=int(request.form["exercise_id"])
    # Save the video to a file
    video.save('video.mp4')  
    
    # Analyze the video 
    left_counter,right_counter,l_ratio_percent,t_ratio_percent = barbell.analyzeBarbellRow('video.mp4')

    
    import datetime

    current_time = datetime.datetime.now()



#save evaluation result
###   user_id,leftcounter,right counter,exercise_id
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_eveluation_results VALUES ( null,%s, %s, %s, %s, %s)', (user_id, exercise_id, left_counter, right_counter,current_time ))
    
    mysql.connection.commit()
        
#get id of inserted evaluation
    eval_id=cursor.lastrowid
    print("eval",eval_id)
#get errors of exercise from exercise_errors table
    # cursor.execute('SELECT id FROM exercise_error WHERE exercise_id = %s', (exercise_id,))
    # exercise_errors = cursor.fetchall()
#for each error insert into evaluation error result table (error_id,error_value)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (6, eval_id, l_ratio_percent,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (7, eval_id, t_ratio_percent,"" ))

    mysql.connection.commit()

    # Return the analysis results 
    results = {
        "id":eval_id,
        "exercise_id":exercise_id,
        "exercise_name":"Barbell Row",
        "date":current_time,
        'left_count': left_counter,
        'right_count': right_counter,
   "errors": 
       [ 
        {"name":"l_ratio",'value': l_ratio_percent,"screenshot":""},
        {"name":"t_ratio",'value': t_ratio_percent,"screenshot":""},
        
        ],

    }
    
    return jsonify(results)


@app.route('/analyze_squat', methods=['POST']) 
def analyze_squat():
    # Get the video from the request 
    video = request.files['video']
    user_id= int(request.form["user_id"])
    exercise_id=int(request.form["exercise_id"])
    # Save the video to a file
    video.save('video.mp4')  
    
    # Analyze the video 
    knees_inward_error_ratio_percent,knees_forward_errors_ratio_percent,left_counter,right_counter = squat.analyze_squat('video.mp4')

    
    import datetime

    current_time = datetime.datetime.now()



#save evaluation result
###   user_id,leftcounter,right counter,exercise_id
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_eveluation_results VALUES ( null,%s, %s, %s, %s, %s)', (user_id, exercise_id, left_counter, right_counter,current_time ))
    
    mysql.connection.commit()
        
#get id of inserted evaluation
    eval_id=cursor.lastrowid
    print("eval",eval_id)
#get errors of exercise from exercise_errors table
    # cursor.execute('SELECT id FROM exercise_error WHERE exercise_id = %s', (exercise_id,))
    # exercise_errors = cursor.fetchall()
#for each error insert into evaluation error result table (error_id,error_value)
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (8, eval_id, knees_forward_errors_ratio_percent,"" ))
    cursor.execute('INSERT INTO exercise_error_results VALUES ( null,%s, %s, %s,%s )', (9, eval_id, knees_inward_error_ratio_percent,"" ))

    mysql.connection.commit()


    # Return the analysis results 
    results ={
        "id":eval_id,
        "exercise_id":exercise_id,
        "exercise_name":"Squat",
        "date":current_time,
        'left_count': left_counter,
        'right_count': right_counter,
   "errors": 
       [ 
        {"name":"Knees Forward",'value': knees_forward_errors_ratio_percent,"screenshot":""},
        {"name":"Knees Inward",'value': knees_inward_error_ratio_percent,"screenshot":""},
        
        ],

    }
    return jsonify(results)


@app.route('/analyze_biceps', methods=['POST']) 
def analyze_biceps():
    # Get the video from the request 
    video = request.files['video']
    user_id= int(request.form["user_id"])
    exercise_id=int(request.form["exercise_id"])
    # Save the video to a file
    video.save('video.mp4')  
    
    # Analyze the video 
    left_arm_count, right_arm_count, left_arm_errors, right_arm_errors, prediction_probability = mpa.analyze_video('video.mp4')

    
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
    app.run(host='192.168.8.108',port=5000)