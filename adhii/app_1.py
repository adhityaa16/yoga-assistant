from datetime import datetime, timedelta
from flask import Flask, render_template, Response, request, flash, redirect, url_for, session, current_app
from flask import jsonify
from sqlalchemy import exc, func, desc
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, time
import cv2
import os
import time
import math
import mediapipe as mp
import pyttsx3
import threading
import requests
import queue
import calendar

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///yoga.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    workouts = db.relationship('Workout', backref='user', lazy=True)
    
    def get_workout_stats(self):
        """Get user's workout statistics"""
        total_workouts = Workout.query.filter_by(user_id=self.id).count()
        total_sequences = db.session.query(func.sum(Workout.sequences_completed))\
                            .filter(Workout.user_id == self.id).scalar() or 0
        
        # Get streak (consecutive days with workouts)
        streak = 0
        current_date = datetime.utcnow().date()
        
        # Check backwards from today
        check_date = current_date
        while True:
            day_workouts = Workout.query.filter(
                Workout.user_id == self.id,
                func.date(Workout.date) == check_date
            ).first()
            
            if day_workouts:
                streak += 1
                check_date = check_date - timedelta(days=1)
            else:
                break
                
        return {
            'total_workouts': total_workouts,
            'total_sequences': total_sequences,
            'current_streak': streak
        }

class Workout(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    sequences_completed = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Initialize Database
with app.app_context():
    db.create_all()

# Camera Configuration
SAVE_FOLDER = "captured_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def init_camera():
    """Initialize camera with multiple index attempts"""
    for index in range(3):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera initialized at index {index}")
            return cap
    raise RuntimeError("No camera detected")

# Initialize camera with app context
def get_camera():
    with app.app_context():
        return init_camera()

try:
    camera = get_camera()
except Exception as e:
    print(f"Camera error: {e}")
    camera = None

# MediaPipe Setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Text-to-Speech Engine
tts_queue = queue.Queue()
tts_lock = threading.Lock()
tts_engine = None
tts_thread_active = False

def tts_worker():
    """Background thread that processes TTS queue"""
    global tts_engine, tts_thread_active
    
    try:
        with tts_lock:
            if tts_engine is None:
                tts_engine = pyttsx3.init()
                tts_engine.setProperty("rate", 150)
                tts_engine.setProperty("volume", 0.9)
        
        while tts_thread_active:
            try:
                # Get message with 0.5 second timeout
                message = tts_queue.get(timeout=0.5)
                
                # Process the message
                with tts_lock:
                    if tts_engine:
                        tts_engine.say(message)
                        tts_engine.runAndWait()
                
                # Mark task as done
                tts_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, just continue
                pass
            except Exception as e:
                print(f"TTS worker error: {e}")
                # Recreate engine on failure
                with tts_lock:
                    tts_engine = None
                    
    except Exception as e:
        print(f"TTS thread error: {e}")
    finally:
        with tts_lock:
            if tts_engine:
                try:
                    tts_engine.stop()
                except:
                    pass
                tts_engine = None

def start_tts_thread():
    """Start the TTS worker thread"""
    global tts_thread_active
    if not tts_thread_active:
        tts_thread_active = True
        threading.Thread(target=tts_worker, daemon=True).start()

def stop_tts_thread():
    """Stop the TTS worker thread"""
    global tts_thread_active
    tts_thread_active = False

def speak_text(text):
    """Add text to the TTS queue"""
    if not tts_thread_active:
        start_tts_thread()
    tts_queue.put(text)

# Start TTS thread when app starts
start_tts_thread()

# Pose sequence
pose_sequence = [
    "Pranamasana (Prayer Pose)",
    "Hasta padasana (second)",
    "Third",
    "Fourth",
    "Fifth Pose",
    "Sixth Pose",
    "Seventh Pose",
    "Eighth Pose",
    "Fourth",
    "Third",
    "Hasta padasana (second)",
    "Pranamasana (Prayer Pose)",
]

# Global State with Thread Safety
current_pose_index = 0
pose_start_time = None
poses_completed = 0
sequences_completed = 0
is_running = False
control_lock = threading.Lock()

# Angle calculation functions
def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: The first landmark containing the x,y and z coordinates.
        landmark2: The second landmark containing the x,y and z coordinates.
        landmark3: The third landmark containing the x,y and z coordinates.
    Returns:
        angle: The calculated angle between the three landmarks.
    '''
    # Get the required landmarks coordinates.
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

# Added alias for compatibility
def calculate_angle(a, b, c):
    """Calculate joint angle between three landmarks (alias function)"""
    return calculateAngle(a, b, c)

def normalize_angle(angle):
    """Normalize angle to 0-180 range"""
    if angle > 180:
        return 360 - angle
    return angle

def detect_pose(image):
    """Detect body landmarks using MediaPipe"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    landmarks = []
    
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,0,0), thickness=2)
        )
        height, width, _ = image.shape
        landmarks = [
            (int(l.x * width), int(l.y * height), int(l.z * width))
            for l in results.pose_landmarks.landmark
        ]
    
    return image, landmarks

def get_landmark_angles(landmarks):
    """Calculate all required joint angles with normalization"""
    angles = {}
    
    # Elbow angles
    angles['left_elbow'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    )
    
    angles['right_elbow'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    )

    # Shoulder angles
    angles['left_shoulder'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    )
    
    angles['right_shoulder'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    )

    # Knee angles
    angles['left_knee'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    )
    
    angles['right_knee'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    )

    # Hip angles
    angles['left_hip'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    )
    
    angles['right_hip'] = calculate_angle(
        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    )

    # Normalize all angles to 0-180 range
    for joint in angles:
        angles[joint] = normalize_angle(angles[joint])

    return angles

def classify_pose(landmarks, image):
    """Classify yoga poses using joint angles"""
    global current_pose_index, pose_start_time, poses_completed, sequences_completed
    
    with control_lock:
        if not is_running:
            cv2.putText(image, "PAUSED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return image, "Paused"

        label = "Unknown Pose"
        color = (0, 0, 255)

        if landmarks:
            try:
                angles = get_landmark_angles(landmarks)
                current_pose = pose_sequence[current_pose_index]

                # Get all angles
                le = angles['left_elbow']
                re = angles['right_elbow']
                ls = angles['left_shoulder']
                rs = angles['right_shoulder']
                lk = angles['left_knee']
                rk = angles['right_knee']
                lh = angles['left_hip']
                rh = angles['right_hip']
                
                # Display angles on image for debugging
                angle_texts = [
                    f"LE: {le:.1f}°", f"RE: {re:.1f}°",
                    f"LS: {ls:.1f}°", f"RS: {rs:.1f}°",
                    f"LK: {lk:.1f}°", f"RK: {rk:.1f}°",
                    f"LH: {lh:.1f}°", f"RH: {rh:.1f}°"
                ]
                
                for i, text in enumerate(angle_texts):
                    cv2.putText(
                        image, text, 
                        (10, 90 + i*30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 
                        1
                    )

                # Pose classification logic
                if current_pose == "Pranamasana (Prayer Pose)":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f}")
                    if (30 < le < 90 and 30 < re < 90 and
                        0 < ls < 50 and 0 < rs < 50 and
                        170 < lk < 190 and 170 < rk < 190):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Hasta padasana (second)":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (140 < le < 180 and 140 < re < 180 and
                        130 < ls < 180 and 130 < rs < 180 and
                        170 < lk < 190 and 170 < rk < 190 and
                        150 < lh < 180 and 150 < rh < 180):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Third":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (140 < le < 180 and 140 < re < 180 and
                        80 < ls < 150 and 80 < rs < 150 and
                        160 < lk < 180 and 160 < rk < 180 and
                        20 < lh < 100 and 20 < rh < 100):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Fourth":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (140 < le < 180 and 140 < re < 180 and
                        20 < ls < 80 and 20 < rs < 80 and
                        40 < lk < 100 and 120 < rk < 180 and
                        20 < lh < 60 and 150 < rh < 180):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Fifth Pose":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (140 < le < 180 and 140 < re < 180 and
                        50 < ls < 90 and 50 < rs < 90 and
                        160 < lk < 180 and 160 < rk < 180 and
                        135 < lh < 180 and 135 < rh < 180):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Sixth Pose":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (10 < le < 80 and 10 < re < 80 and
                        0 < ls < 30 and 0 < rs < 30 and
                        120 < lk < 170 and 120 < rk < 170 and
                        110 < lh < 170 and 110 < rh < 170):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Seventh Pose":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (10 < le < 180 and 10 < re < 180 and
                        0 < ls < 50 and 0 < rs < 50 and
                        150 < lk < 180 and 150 < rk < 180 and
                        110 < lh < 150 and 110 < rh < 150):
                        label = current_pose
                        color = (0, 255, 0)

                elif current_pose == "Eighth Pose":
                    print(f"Elbows: {le:.1f}, {re:.1f} | Shoulders: {ls:.1f}, {rs:.1f} | Knees: {lk:.1f}, {rk:.1f} | Hips: {lh:.1f}, {rh:.1f}")
                    if (140 < le < 180 and 140 < re < 180 and
                        145 < ls < 180 and 145 < rs < 180 and
                        150 < lk < 180 and 150 < rk < 180 and
                        60 < lh < 150 and 60 < rh < 150):
                        label = current_pose
                        color = (0, 255, 0)


                # Pose timing and progression logic
                if label == current_pose:
                    if pose_start_time is None:
                        pose_start_time = time.time()
                        speak_text(f"Hold {label} for 5 seconds")
                    
                    if time.time() - pose_start_time >= 5:
                        poses_completed += 1
                        current_pose_index += 1
                        pose_start_time = None
                        
                        # Save pose snapshot
                        filename = os.path.join(SAVE_FOLDER, 
                            f"seq_{sequences_completed}_pose_{poses_completed}.jpg")
                        cv2.imwrite(filename, image)
                        speak_text(f"{label} completed!")

                # Sequence completion check
                if current_pose_index >= len(pose_sequence):
                    sequences_completed += 1
                    current_pose_index = 0
                    speak_text("Sequence completed! Starting over.")
                    
                    # Save workout data
                    if 'user_id' in session:
                        try:
                            # Direct database save instead of HTTP request
                            workout = Workout(
                                sequences_completed=1,  # Each save is for one sequence
                                user_id=session['user_id']
                            )
                            db.session.add(workout)
                            db.session.commit()
                        except Exception as e:
                            print(f"Workout save error: {e}")

            except Exception as e:
                print(f"Classification error: {e}")

        # Update display text
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, f"Poses: {poses_completed} | Sequences: {sequences_completed}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        return image, label


def process_frame():
    """Video streaming generator function"""
    while True:
        if camera is None:
            break
            
        success, frame = camera.read()
        if not success:
            continue
            
        try:
            if is_running:
                frame, landmarks = detect_pose(frame)
                frame, _ = classify_pose(landmarks, frame)
            else:
                frame, _ = classify_pose(None, frame)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Frame error: {e}")

# Helper functions for workout history
def get_monthly_workouts(user_id, year, month):
    """Get all workouts for a specific month and year"""
    # Create date range for the month
    last_day = calendar.monthrange(year, month)[1]
    start_date = datetime(year, month, 1)
    end_date = datetime(year, month, last_day, 23, 59, 59)
    
    # Query workouts in date range
    workouts = Workout.query.filter(
        Workout.user_id == user_id,
        Workout.date >= start_date,
        Workout.date <= end_date
    ).all()
    
    # Format for calendar display
    calendar_data = {}
    for workout in workouts:
        day = workout.date.day
        if day not in calendar_data:
            calendar_data[day] = {
                'count': 0,
                'sequences': 0
            }
        calendar_data[day]['count'] += 1
        calendar_data[day]['sequences'] += workout.sequences_completed
    
    return calendar_data

def get_recent_workouts(user_id, limit=5):
    """Get user's most recent workouts"""
    workouts = Workout.query.filter_by(user_id=user_id)\
                .order_by(desc(Workout.date))\
                .limit(limit).all()
    return workouts

# Flask Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    user = User.query.get(session['user_id'])
    
    # Get recent workouts for display
    recent_workouts = get_recent_workouts(user.id, 3)
    
    # Get stats
    stats = user.get_workout_stats()
    
    return render_template('index.html', 
                          current_user=user, 
                          recent_workouts=recent_workouts,
                          stats=stats)

@app.route('/start', methods=['POST'])
def start_detection():
    with control_lock:
        global is_running
        is_running = True
    return '', 204

@app.route('/pause', methods=['POST'])
def pause_detection():
    with control_lock:
        global is_running
        is_running = False
    return '', 204

@app.route('/reset', methods=['POST'])
def reset_detection():
    with control_lock:
        global is_running, current_pose_index, pose_start_time, poses_completed, sequences_completed
        is_running = False
        current_pose_index = 0
        pose_start_time = None
        poses_completed = 0
        sequences_completed = 0
    return '', 204


@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/save_workout', methods=['POST'])
def save_workout():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    workout = Workout(
        sequences_completed=request.json['sequences'],
        user_id=session['user_id']
    )
    db.session.add(workout)
    db.session.commit()
    return jsonify({'message': 'Workout saved'}), 200


@app.route('/workout_history')
def workout_history():
    """Display workout history page with calendar"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user = User.query.get(session['user_id'])
    
    # Get year and month from query params, default to current date
    current_date = datetime.utcnow()
    
    # Fixed code for handling year parameter
    year_param = request.args.get('year', '')
    try:
        year = int(year_param) if year_param else current_date.year
    except ValueError:
        year = current_date.year
    
    # Fixed code for handling month parameter
    month_param = request.args.get('month', '')
    try:
        month = int(month_param) if month_param else current_date.month
    except ValueError:
        month = current_date.month
    
    # Get calendar data
    cal_data = get_monthly_workouts(user.id, year, month)
    
    # Create calendar HTML
    cal = calendar.monthcalendar(year, month)
    month_name = calendar.month_name[month]
    
    # Navigation links for previous/next month
    prev_month = month - 1
    prev_year = year
    if prev_month == 0:
        prev_month = 12
        prev_year -= 1
        
    next_month = month + 1
    next_year = year
    if next_month == 13:
        next_month = 1
        next_year += 1
    
    # Get workout stats
    stats = user.get_workout_stats()
    
    return render_template('workout_history.html',
                          current_user=user,
                          year=year,
                          month=month,
                          month_name=month_name,
                          calendar=cal,
                          workout_data=cal_data,
                          prev_month=prev_month,
                          prev_year=prev_year,
                          next_month=next_month,
                          next_year=next_year,
                          stats=stats)

@app.route('/api/progress')
def get_progress():
    return jsonify({
        'current_pose': pose_sequence[current_pose_index] if current_pose_index < len(pose_sequence) else 'Completed',
        'current_pose_index': current_pose_index,
        'poses_completed': poses_completed,
        'sequences_completed': sequences_completed
    }) 

@app.route('/api/workout-details/<date>')
def workout_details(date):
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Store user_id locally to avoid repeated session access
    user_id = session.get('user_id')
    
    try:
        # Parse the date
        year, month, day = map(int, date.split('-'))
        
        # Create datetime objects for start and end of the day
        start_date = datetime.combine(datetime(year, month, day), time.min)
        end_date = datetime.combine(datetime(year, month, day), time.max)
        
        # Use a separate session object for this query to avoid conflicts
        # with the camera processing
        workouts = Workout.query.filter(
            Workout.user_id == user_id,
            Workout.date >= start_date,
            Workout.date <= end_date
        ).order_by(Workout.date).all()
        
        # Calculate total sequences without using sum() to improve performance
        total_sequences = 0
        workout_list = []
        
        for workout in workouts:
            total_sequences += workout.sequences_completed
            workout_list.append({
                'time': workout.date.strftime('%I:%M %p'),
                'sequences': workout.sequences_completed,
                'date': workout.date.strftime('%Y-%m-%d')
            })
        
        # Build response object
        response = {
            'workouts': workout_list,
            'total_sequences': total_sequences,
            'date': date
        }
        
        return jsonify(response)
        
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400
    except Exception as e:
        print(f"Workout details error: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500    
          

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            flash('Logged in successfully!', 'success')
            return redirect(url_for('index'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            
            if User.query.filter_by(username=username).first():
                flash('Username already exists', 'danger')
                return redirect(url_for('signup'))
                
            hashed_password = generate_password_hash(password)
            new_user = User(
                username=username,
                email=email,
                password_hash=hashed_password
            )
            db.session.add(new_user)
            db.session.commit()
            flash('Account created! Please login', 'success')
            return redirect(url_for('login'))
            
        except exc.IntegrityError:
            db.session.rollback()
            flash('Email already exists', 'danger')
        except Exception as e:
            db.session.rollback()
            print(f"Signup error: {e}")
            flash('Error creating account', 'danger')
            
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if camera:
            camera.release()
        stop_tts_thread()
