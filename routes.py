from flask import json, json, render_template, request, jsonify, send_file
from flask_login import login_user, logout_user, current_user
from models import Course, Users
from werkzeug.security import generate_password_hash, check_password_hash
import AI_API, os, pandas
import jwt
import datetime
from functools import wraps
import uuid
from flask import request, jsonify
from AI_API import generate_adaptive_exam, update_relevance_map
EXAM_SESSIONS = {} 

SECRET_KEY = "your_super_secret_key" # Keep this in environment variables!
IMAGE_STORAGE_DIR = "IMAGE_STORAGE_DIR"
UPLOAD_DIR = 'user_data_cache'
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)



def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({"message": "Token is missing"}), 401
        try:
            data = jwt.decode(token.replace("Bearer ", ""), SECRET_KEY, algorithms=["HS256"])
            current_user_id = data['user_id']
        except Exception:
            return jsonify({"message": "Token is invalid"}), 401
        return f(current_user_id, *args, **kwargs)
    return decorated


def register_routes(app, db):
    @app.route('/people', methods=['GET', 'POST'])
    def index():
        pass

    @app.route('/upload', methods=['POST'])
    @token_required
    def upload_file(user_id):
        if 'files' not in request.files:
            return jsonify({"success": False, "message": "No file field found"}), 400

        course = request.form.get('course')
        files = request.files.getlist('files')

        if not course:
            return jsonify({"success": False, "message": "Course name is required"}), 400

        all_data = []

        for file in files:
            if file.filename == '':
                continue

            if file.filename.lower().endswith('.pdf'):
                all_data.extend(AI_API.process_pdf_and_save_images(file, course))
            elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(AI_API.IMAGE_STORAGE_DIR, f"{course}_{file.filename}")
                file.save(img_path)
                all_data.append({'type': 'image', 'content': img_path})
                extracted_text = AI_API.extract_text_from_image(img_path)
                if extracted_text:
                    chunks = AI_API.split_text(extracted_text)
                    for chunk in chunks:
                        all_data.append({'type': 'text', 'content': f"[Extracted from image '{file.filename}']: {chunk}"})

        if not all_data:
            return jsonify({"success": False, "message": "No valid files processed"}), 400

        victoria = AI_API.get_vectors(all_data)
        file_path = os.path.join(UPLOAD_DIR, f"{course}.pkl")
        victoria.to_pickle(file_path)

        existing_course = Course.query.filter_by(name=course, user_id=user_id).first()
        if not existing_course:
            new_course = Course(name=course, user_id=user_id)
            db.session.add(new_course)
            db.session.commit()

        return jsonify({"success": True, "message": f"Processed {course}"}), 200

    @app.route('/')
    @token_required
    def user(user_id):
        if current_user.is_authenticated:
            return str(current_user.username)
        return 'no user is logged in'

    @app.route('/login', methods=['POST'])
    def login():
        data = request.get_json(force=True)
        user = Users.query.filter(Users.username == data.get('username')).first()

        if not user:
            return "no user"
        if user and check_password_hash(user.password, data.get('password')):
            token = jwt.encode({
                'user_id': user.id,
                'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, SECRET_KEY, algorithm="HS256")
            return jsonify({
                "status": "success",
                "token": token,
                "courses": [c.name for c in user.courses]
            })
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    @app.route('/logout/<uid>')
    @token_required     
    def logout(uid):
        logout_user()
        return "success"

    @app.route('/signup', methods=['POST', 'GET'])
    def signup():
        if request.method == 'GET':
            return render_template('signup.html')
        if request.method == 'POST':
            data = request.get_json(force=True)
            if data and data.get('username') and data.get('password'):
                name = data.get('username')
                past = data.get('password')
                email = data.get('email')
                password = generate_password_hash(past)
                user = Users(username=name, password=password, email=email)
                db.session.add(user)
                db.session.commit()
                login_user(user)
                return 'success'
            return "enter username and password"

    @app.route('/get_question', methods=['POST'])
    @token_required
    def get_question(user_id):
        data = request.get_json(force=True)
        question = data.get('question')
        file_path = os.path.join(UPLOAD_DIR, "f.pkl")
        victoria = pandas.read_pickle(file_path)
        vicky = AI_API.find_best_passage(question, victoria)
        cleaned_text = " ".join(vicky.split())
        return jsonify({"answer": cleaned_text})

    @app.route('/get_topic', methods=['POST'])
    @token_required
    def getttopic(user_id):
        data = request.get_json(force=True)
        course = data.get('course')

        csv_path = os.path.join(UPLOAD_DIR, f"{course}topics.csv")
        pkl_path = os.path.join(UPLOAD_DIR, f"{course}.pkl")

        if not os.path.exists(csv_path):
            victoria = pandas.read_pickle(pkl_path)
            som = AI_API.gettopic(victoria, course)
            x = pandas.DataFrame(som)
            x.to_csv(csv_path, index=False)
        else:
            som = pandas.read_csv(csv_path)
            return jsonify(som['topic'].tolist())

        if som['topic'] is list:
            return jsonify(som['topic'])
        else:
            return jsonify(som['topic'].tolist())

    @app.route('/study', methods=['POST'])
    @token_required
    def study(user_id):
        data = request.get_json(force=True)
        course = data.get('course')
        topic_value = data.get('topic')

        file_path = os.path.join(UPLOAD_DIR, f"{course}topics.csv")
        victoria = pandas.read_csv(file_path)
        result = victoria[victoria['topic'] == topic_value]

        file = os.path.join(UPLOAD_DIR, f"{course}topic.csv")
        vick = pandas.read_csv(file)
        merged = vick.merge(result, on='module_id')
        chunks = merged['content'].tolist()
        v = AI_API.study(topic_value, chunks)
        print(v)
        return jsonify(v)

    @app.route('/ask_ai', methods=['POST'])
    @token_required
    def ask_ai(user_id):
        data = request.get_json(force=True)
        question = data.get('question')
        course = data.get('course')
        history = data.get('history')

        file_path = os.path.join(UPLOAD_DIR, f"{course}.pkl")
        vick = pandas.read_pickle(file_path)
        a = AI_API.find_best_passage(question, vick)
        answer = AI_API.ask(question=question, chunks=a, history=history)
        return jsonify({"answer": answer}), 200

    # Add this to your routes.py


    @app.route('/generate_exam', methods=['POST'])
    @token_required
    def generate_exam(user_id):
        data = request.get_json(force=True)
        course_name = data.get('course')
        
        if not course_name:
            return jsonify({"status": "error", "message": "Course name is required"}), 400

        try:
            # Call the production AI function
            # Ensure UPLOAD_DIR matches the path where your files are actually saved
            questions_data = generate_adaptive_exam(course_name, upload_dir=UPLOAD_DIR)
            EXAM_SESSIONS[user_id] = questions_data 
            
            return jsonify({
                "status": "success",
                "questions": questions_data
            })
            
        except FileNotFoundError as e:
            # Triggered if the course material hasn't been uploaded yet
            return jsonify({"status": "error", "message": str(e)}), 404
            
        except ValueError as e:
            # Triggered if Gemini returns a bad JSON format
            return jsonify({"status": "error", "message": str(e)}), 500
            
        except Exception as e:
            # Catch-all for API connection issues, etc.
            return jsonify({"status": "error", "message": f"An unexpected error occurred: {str(e)}"}), 500
        

    @app.route('/upload_past_questions', methods=['POST'])
    @token_required
    def upload_past_questions(user_id):
        course_name = request.form.get('course')
        
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400

        if file and course_name:
            try:
                # 1. Extract Text and Images directly from the file stream using AI_API
                data_list = AI_API.process_pdf_and_save_images(file, course_name)
                
                # 2. Compile all content into a single massive string for analysis
                past_questions_text = ""
                
                for item in data_list:
                    if item['type'] == 'text':
                        past_questions_text += item['content'] + "\n"
                    elif item['type'] == 'image':
                        # Use Gemini OCR to read text out of diagrams, scanned pages, or math equations
                        ocr_text = AI_API.extract_text_from_image(item['content'])
                        if ocr_text:
                            past_questions_text += f"\n[Extracted from image/diagram]: {ocr_text}\n"
                
                # Security check: Make sure we actually found something
                if not past_questions_text.strip():
                    return jsonify({"status": "error", "message": "No readable text or images found in the document."}), 400
                
                # 3. Pass the combined text to generate the relevance map
                relevance_map = AI_API.update_relevance_map(
                    course_name=course_name, 
                    past_questions_text=past_questions_text, 
                    upload_dir=AI_API.UPLOAD_DIR
                )
                
                return jsonify({
                    "status": "success", 
                    "message": "Past questions analyzed and relevance map updated!",
                    "relevance": relevance_map
                })
                
            except FileNotFoundError as e:
                return jsonify({"status": "error", "message": str(e)}), 404
            except Exception as e:
                # Catch-all for API issues, invalid PDFs, etc.
                return jsonify({"status": "error", "message": f"Processing failed: {str(e)}"}), 500

    @app.route('/submit_exam', methods=['POST'])
    @token_required
    def submit_exam(user_id):
        data = request.get_json(force=True)
        user_answers = data.get('answers') # Expected: { "0": "answer...", "1": "answer..." }
        
        # 1. Retrieve the original questions
        original_exam = EXAM_SESSIONS.get(user_id)
        if not original_exam:
            return jsonify({"message": "No active exam session found"}), 400
        
        # 2. Grade all answers in ONE single API call
        graded_batch = AI_API.grade_exam_batch(original_exam, user_answers)
        
        # Create a lookup map by index in case the AI returns items out of order
        results_map = {item['index']: item for item in graded_batch}
        
        graded_items = []
        total_score = 0
        
        # 3. Build the response structure for your Flutter frontend
        # 3. Build the response structure for your Flutter frontend
        for i, question_data in enumerate(original_exam):
            ans = user_answers.get(str(i), "")
            
            # Fetch the grade from our AI batch result map
            result = results_map.get(i, {"score": 0, "feedback": "Grading missed for this item."})
            
            score = result.get('score', 0)
            feedback = result.get('feedback', 'No feedback provided.')
            
            graded_items.append({
                "question": question_data['question'],
                "userAnswer": ans,
                # Add the rubric here so Flutter can display what the answer SHOULD have been:
                "correctAnswer": question_data.get('rubric', 'No specific rubric provided.'),
                "score": score,
                "feedback": feedback
            })
            
            total_score += score
        
        # 4. Clear session and return results
        EXAM_SESSIONS.pop(user_id, None)
        
        return jsonify({
            "score": total_score,
            "total": len(original_exam) * 10,
            "items": graded_items
        })


    @app.route('/get_relevance/<course_name>', methods=['GET'])
    @token_required
    def get_relevance(user_id, course_name):
        relevance_file = os.path.join(AI_API.UPLOAD_DIR, f"{course_name}_relevance.json")
        
        if not os.path.exists(relevance_file):
            return jsonify({"status": "error", "message": "No relevance map found."}), 404
            
        with open(relevance_file, 'r') as f:
            data = json.load(f)
            
        return jsonify({"status": "success", "relevance": data})

    @app.route('/clarify_item', methods=['POST'])
    @token_required
    def clarify_item(user_id):
        data = request.get_json(force=True)
        
        question = data.get('question', '')
        user_answer = data.get('userAnswer', '')
        correct_answer = data.get('correctAnswer', '')
        feedback = data.get('feedback', '')
        user_query = data.get('query', '')
        
        if not user_query:
            return jsonify({"status": "error", "message": "Query cannot be empty"}), 400
            
        explanation = AI_API.clarify_exam_item(
            question=question,
            user_answer=user_answer,
            correct_answer=correct_answer,
            feedback=feedback,
            user_query=user_query
        )
        
        return jsonify({"status": "success", "answer": explanation}), 200
