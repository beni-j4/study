import fitz  # PyMuPDF
import json
import os
import textwrap
import uuid
import numpy as np
import pandas as p
from dotenv import load_dotenv
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap

# Legacy SDK
import google.generativeai as genai

# New SDK (Cleanly aliased to prevent namespace conflicts)
from google import genai as google_genai
from google.genai import types

# 1. Load variables from the .env file into the system environment
load_dotenv()

# 2. Retrieve the API key securely
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("No GEMINI_API_KEY found. Please set it in your .env file!")

# Configure the legacy Gemini SDK
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-2.5-flash')

UPLOAD_DIR = 'user_data_cache'
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMAGE_STORAGE_DIR = 'IMAGE_STORAGE_DIR'
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)

def reduce(merged_data):
    # 1. Prepare contents list for Gemini
    contents = []
    text_buffer = ""

    for item in merged_data:
        # If it's a file path (image)
        if isinstance(item, str) and os.path.exists(item):
            # If we have accumulated text, add it first
            if text_buffer:
                contents.append(text_buffer)
                text_buffer = ""
            # Add the opened PIL Image object
            contents.append(Image.open(item))
        else:
            # Accumulate text
            text_buffer += f"\n{item}"
    
    # Add final text chunk if exists
    if text_buffer:
        contents.append(text_buffer)

    # 2. Add the prompt to the contents
    prompt = "Task: Analyze these documents and images to identify the core topic. Respond with only the topic name (max 10 words)."
    contents.append(prompt)

    # 3. Generate content using the global 'model' defined at the top of your file
    response = model.generate_content(contents)
    return response.text

def create_conv(chunk):
    client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        config=types.GenerateContentConfig(system_instruction=chunk),
        contents="Hello there"
    )

    chat = client.chats.create(model="gemini-3-flash-preview")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f'role - {message.role}', end=": ")
        print(message.parts[0].text)
    print(response.text)

def embed_fn(item):
    model_name = 'gemini-embedding-2'
    
    if item['type'] == 'image':
        content_to_embed = extract_text_from_image(item['content'])
        if not content_to_embed:
            content_to_embed = "Image with no readable text"
    else:
        content_to_embed = item['content']
        
    response = genai.embed_content(
        model=model_name,
        content=content_to_embed,
        task_type="retrieval_document"
    )
    return response['embedding']

def process_pdf(file_stream):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    data_list = []
    
    # Text extraction
    for page in doc:
        text = page.get_text()
        if text.strip():
            data_list.append({'type': 'text', 'content': text})
    
    # Image extraction
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            data_list.append({'type': 'image', 'content': pix.tobytes()})
    return data_list

def split_text(text):
    """Splits extracted PDF text into manageable chunks for AI processing."""
    chunks = textwrap.wrap(text, width=200)
    return chunks

def pdf_Reader(pdf_path):
    doc = fitz.open(stream=pdf_path.read(), filetype="pdf")
    data_list = []
    
    # 1. Extract Text
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks = textwrap.wrap(text, width=500)
            for chunk in chunks:
                data_list.append({'type': 'text', 'content': chunk})
    
    # 2. Extract Images
    for page_index in range(len(doc)):
        image_list = doc.get_page_images(page_index)
        for img in image_list:
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_bytes = pix.tobytes()
            data_list.append({'type': 'image', 'content': img_bytes})
            
    return data_list

def get_vectors(data_list):
    df = p.DataFrame(data_list) 
    df['Embeddings'] = df.apply(lambda row: embed_fn({'type': row['type'], 'content': row['content']}), axis=1)
    return df
    
def find_best_passage(query, dataframe):
    model_name = "gemini-embedding-2"
    
    query_response = genai.embed_content(
        model=model_name,
        content=query,
        task_type="question_answering"
    )
    query_embedding = query_response['embedding']

    dot_products = np.dot(
        np.stack(dataframe['Embeddings']),
        query_embedding
    )
    idx = np.argmax(dot_products)
    score = dot_products[idx]
    print(score)
    if score > 0.75:
        return str(dataframe.iloc[idx]['content'])
    else:
        return 'No information on this topic'
 
def gettopic(df, course):
    is_small_dataset = len(df) < 8
    
    if not is_small_dataset:
        reducer = umap.UMAP(n_components=5, n_neighbors=min(len(df), 9), metric='cosine', random_state=42)
        embeddings_reduced = reducer.fit_transform(np.stack(df['Embeddings'].values))
        kmeans = KMeans(n_clusters=8, random_state=42)
        df = df.copy()
        df['module_id'] = kmeans.fit_predict(embeddings_reduced)
    else:
        df = df.copy()
        df['module_id'] = 1 

    file_path = os.path.join(UPLOAD_DIR, f"{course}topic.csv")
    df.to_csv(file_path)
    
    some = []
    vicky = []
    module_chunk = {'topic': some, 'module_id': vicky}
    
    unique_modules = df['module_id'].unique()
    
    for each in unique_modules:
        dc = np.argwhere(df['module_id'].values == each).flatten()
        merged = []
        
        for i in dc:
            for j in dc:
                if i > j:
                    emb_i = np.array(df.iloc[i]['Embeddings']).reshape(1, -1)
                    emb_j = np.array(df.iloc[j]['Embeddings']).reshape(1, -1)
                    
                    dot_product = cosine_similarity(emb_i, emb_j)[0][0]
                    
                    if 0.85 < dot_product < 1:
                        if df.iloc[i]['content'] not in merged:
                            merged.append(df.iloc[i]['content']) 
                        if df.iloc[j]['content'] not in merged:
                            merged.append(df.iloc[j]['content']) 
        
        if not merged and len(dc) > 0:
            merged = df.iloc[dc]['content'].tolist()
            
        merged = list(set(merged))
        if merged:
            some.append(str(reduce(merged)))
            vicky.append(each)

    return module_chunk

def study(topic, chunks):
    prompt = f"""System/Context: You are an expert educator specializing in simplifying complex topics for students. 
    Task: Create a comprehensive study guide based on the provided topic and text chunks or images. Input 
    Topic: {topic} Input Text Chunks: {chunks} Please structure the output as follows: 
    Summary: Provide a 30-sentence high-level overview of the topic. Key Concepts: Create a bulleted list of the most important terms/concepts 
    found in the text and explain them briefly in 200 words, in plain language. 
    The Why it Matters Section: Explain why this topic is relevant to the broader subject or real-world application. 
    Study Quiz: Provide 15 multiple-choice questions (with one correct answer and two distractors each) to test understanding. 
    Advanced Prompt: Create a single, open-ended thought question that would require a student to analyze the text deeply. 
    Constraints: Keep the tone encouraging and academic but accessible. Use clear, concise language.
    Return your response strictly as a valid JSON dictionary with this exact structure:
    {{
    "content": "return the initial chunk of text you were given, with complete information"
    "summary": "...",
    "key_concepts": [{{"term": "...", "definition": "...", "overview and relevance": "..." }}],
    "why_it_matters": "...",
    "study_quiz": [{{"question": "...", "answers": [{{"text": "...", "correct": true}}, ...]}}],
    }}
    """
    response = model.generate_content(prompt)
    return response.text

def ask(question, chunks, history):
    client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    prompt = f"Answer this: {question} based on: {chunks}, keep your answer short and concise, Around 100 to 200 words"
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

def extract_text_from_image(image_path):
    """Uses Gemini to extract text from a saved image file."""
    try:
        img = Image.open(image_path)
        prompt = "Extract all readable text from this image exactly as written. Do not summarize. If there is no text, return exactly 'NO_TEXT_FOUND'."
        response = model.generate_content([img, prompt])
        
        if not response.text or "NO_TEXT_FOUND" in response.text:
            return None
            
        return response.text
    except Exception as e:
        print(f"OCR Error on {image_path}: {e}")
        return None

def process_pdf_and_save_images(file_stream, course_name):
    doc = fitz.open(stream=file_stream.read(), filetype="pdf")
    data_list = []
    
    for page in doc:
        text = page.get_text()
        if text.strip():
            chunks = split_text(text)
            for chunk in chunks:
                data_list.append({'type': 'text', 'content': chunk})
            
    for page_idx, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            
            img_filename = f"{course_name}_p{page_idx}_i{img_idx}.png"
            img_path = os.path.join(IMAGE_STORAGE_DIR, img_filename)
            pix.save(img_path)
            
            data_list.append({'type': 'image', 'content': img_path})
            
    return data_list

def generate_adaptive_exam(course_name, upload_dir="uploads"):
    topics_file = os.path.join(upload_dir, f"{course_name}topics.csv")
    if not os.path.exists(topics_file):
        raise FileNotFoundError(f"Course material for '{course_name}' not found. Please upload materials first.")
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        course_content = f.read()

    relevance_file = os.path.join(upload_dir, f"{course_name}_relevance.json")
    relevance_data = "No past question data available. Distribute questions evenly across topics."
    
    if os.path.exists(relevance_file):
        with open(relevance_file, 'r') as f:
            relevance_data = f.read()

    prompt = f"""
    You are an expert university professor. Create exactly 20 exam questions for the course: '{course_name}'.
    
    Course material: {course_content}
    Historical relevance: {relevance_data}
    
    CRITICAL CONSTRAINTS:
    1. You must return EXACTLY 20 exam questions.
    2. You must include 15 'objective' (multiple choice/short answer) and 5 'essay' (long form) questions.
    3. Return ONLY a valid JSON array. Do not include any intro/outro text.
    
    Schema for each:
    {{
        "id": "uuid",
        "question": "The question text",
        "rubric": "Grading criteria",
        "type": "essay" OR "objective"
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        questions = json.loads(raw_text)
        
        for q in questions:
            if 'id' not in q or len(q['id']) < 5:
                q['id'] = str(uuid.uuid4())
                
        return questions
    except json.JSONDecodeError:
        print(f"Failed to parse AI output: {response.text}")
        raise ValueError("The AI generated an invalid exam format. Please try again.")

def update_relevance_map(course_name, past_questions_text, upload_dir="uploads"):
    topics_file = os.path.join(upload_dir, f"{course_name}topics.csv")
    
    if not os.path.exists(topics_file):
        raise FileNotFoundError("Course topics must be uploaded before past questions.")
        
    with open(topics_file, 'r', encoding='utf-8') as f:
        course_topics = f.read()

    prompt = f"""
    You are an expert AI teaching assistant. 
    
    Here is the course syllabus/topics:
    {course_topics}
    
    Here is a set of past exam questions for this course:
    {past_questions_text}
    
    Task: Analyze the past questions and map them to the course topics. 
    Determine the relevance of each topic based on how frequently it appears in the past questions.
    
    Return ONLY a valid JSON object where the keys are the exact topic names, and the values are either "High", "Medium", or "Low".
    Do not use markdown blocks like ```json.
    
    Example output format:
    {{
        "Introduction to Biology": "High",
        "Cell Structure": "Medium",
        "Plant Anatomy": "Low"
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        relevance_data = json.loads(raw_text)
        
        relevance_file = os.path.join(upload_dir, f"{course_name}_relevance.json")
        with open(relevance_file, 'w', encoding='utf-8') as f:
            json.dump(relevance_data, f, indent=4)
            
        return relevance_data
        
    except json.JSONDecodeError:
        raise ValueError("Failed to parse AI relevance map.")

def grade_essay(question, student_response, rubric):
    prompt = f"""
    You are an expert examiner. 
    Question: {question}
    Student Response: {student_response}
    Grading Rubric: {rubric}
    
    Provide a score out of 10 and a brief justification.
    Return ONLY JSON: {{"score": 8, "feedback": "Your explanation was correct but missed..."}}
    """
    
    response = model.generate_content(prompt)
    raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
    try:
        return json.loads(raw_text)
    except:
        return {"score": 0, "feedback": "Error processing answer."}

def grade_exam_batch(original_exam, user_answers):
    batch_payload = []
    for i, question_data in enumerate(original_exam):
        batch_payload.append({
            "index": i,
            "question": question_data['question'],
            "rubric": question_data.get('rubric', 'Grade out of 10 based on accuracy.'),
            "student_response": user_answers.get(str(i), "")
        })
        
    prompt = f"""
    You are an expert university examiner. You have been given a batch of exam questions, the grading rubric for each, and the student's submitted responses.
    
    Evaluate every student response in the batch and assign a score out of 10, along with brief, constructive feedback.
    
    Input Batch:
    {json.dumps(batch_payload, indent=2)}
    
    CRITICAL CONSTRAINTS:
    1. You MUST grade every single item in the batch.
    2. Return ONLY a valid JSON array of objects. Do not include any intro/outro text or markdown formatting.
    3. Each object in your returned array must follow this exact schema:
    [
      {{
        "index": 0,
        "score": 8,
        "feedback": "Your explanation was correct but missed..."
      }},
      ...
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        return json.loads(raw_text)
    except Exception as e:
        print(f"Batch grading failed: {e}")
        return [{"index": i, "score": 0, "feedback": "Error processing grading for this item."} for i in range(len(original_exam))]
    
def clarify_exam_item(question, user_answer, correct_answer, feedback, user_query):
    prompt = f"""
    You are an encouraging and insightful university tutor helping a student review their exam results.
    
    Context of the exam item:
    - Question Asked: "{question}"
    - Student's Answer: "{user_answer}"
    - Expected Answer / Rubric: "{correct_answer}"
    - Examiner's Initial Feedback: "{feedback}"
    
    Student's Follow-up Question/Doubt: "{user_query}"
    
    Task: Answer the student's follow-up question directly. Explain clearly and gently why their original answer received the score it did, or help them understand the correct concept based on the rubric. 
    Keep your explanation concise, friendly, and under 150 words.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Clarification error: {e}")
        return "I'm having trouble analyzing this question right now. Please try again in a moment."
