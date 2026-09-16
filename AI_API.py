import fitz  # PyMuPDF
import json
import os
import re
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

# Configure the legacy Gemini SDK using environment variable
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-3.1-flash-lite')

UPLOAD_DIR = 'user_data_cache'
os.makedirs(UPLOAD_DIR, exist_ok=True)
IMAGE_STORAGE_DIR = 'IMAGE_STORAGE_DIR'
os.makedirs(IMAGE_STORAGE_DIR, exist_ok=True)

def extract_json(text):
    """Safely extracts JSON from model response text even if wrapped in markdown."""
    try:
        # Strip markdown code blocks if present
        cleaned_text = re.sub(r'```(?:json)?\s*|\s*```', '', text, flags=re.MULTILINE).strip()
        return json.loads(cleaned_text)
    except Exception:
        # Fallback regex search for JSON object or array
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        raise ValueError("Could not extract valid JSON from response.")

def reduce(merged_data):
    contents = []
    text_buffer = ""

    for item in merged_data:
        if isinstance(item, str) and os.path.exists(item):
            if text_buffer:
                contents.append(text_buffer)
                text_buffer = ""
            contents.append(Image.open(item))
        else:
            text_buffer += f"\n{item}"
    
    if text_buffer:
        contents.append(text_buffer)

    prompt = "Task: Analyze these documents and images to identify the core topic. Respond with only the topic name (max 10 words)."
    contents.append(prompt)

    response = model.generate_content(contents)
    return response.text

def create_conv(chunk):
    client = google_genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(system_instruction=chunk),
        contents="Hello there"
    )

    chat = client.chats.create(model="gemini-2.5-flash")

    response = chat.send_message("I have 2 dogs in my house.")
    print(response.text)

    response = chat.send_message("How many paws are in my house?")
    print(response.text)

    for message in chat.get_history():
        print(f'role - {message.role}', end=": ")
        print(message.parts[0].text)

def embed_fn(item):
    model_name = 'models/gemini-embedding-2'
    
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
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            data_list.append({'type': 'image', 'content': pix.tobytes("png")})
    return data_list

def split_text(text):
    """Splits extracted PDF text into manageable chunks for AI processing."""
    return textwrap.wrap(text, width=200)

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
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            img_bytes = pix.tobytes("png")
            data_list.append({'type': 'image', 'content': img_bytes})
            
    return data_list

def get_vectors(data_list):
    df = p.DataFrame(data_list) 
    df['Embeddings'] = df.apply(lambda row: embed_fn({'type': row['type'], 'content': row['content']}), axis=1)
    return df
    
def find_best_passage(query, dataframe):
    model_name = "models/text-embedding-004"
    
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
        kmeans = KMeans(n_clusters=min(8, len(df)), random_state=42)
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
    items_to_process = chunks if isinstance(chunks, list) else [chunks]
    text_contents = []

    for item in items_to_process:
        if isinstance(item, dict):
            # Handle structured dictionaries like {'type': 'image', 'content': 'path'}
            if item.get('type') == 'image' and 'content' in item:
                ocr_text = extract_text_from_image(item['content'])
                if ocr_text:
                    text_contents.append(ocr_text)
            elif 'content' in item:
                text_contents.append(str(item['content']))
            else:
                text_contents.append(str(item))
        elif isinstance(item, str):
            # Check if the string is a file path pointing to a saved image on disk
            if os.path.exists(item) and any(item.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']):
                ocr_text = extract_text_from_image(item)
                if ocr_text:
                    text_contents.append(ocr_text)
            else:
                text_contents.append(item)
        else:
            text_contents.append(str(item))

    # Join all extracted text into a single clean string (only text is retained!)
    content_str = "\n\n".join(text_contents).strip()
    if not content_str:
        content_str = "No readable content or text could be extracted."

    prompt = f"""System/Context: You are an expert university professor specializing in breaking down complex topics, mathematical concepts, derivations, and theory for students. 

    Task: Create a comprehensive study guide based on the provided topic and text chunks or image paths. 
    Input Topic: {topic} 
    Input Content / Chunks: {chunks} 
    
    Please structure the output as follows: 
    1. Summary: Provide a clear, intuitive overview of the topic, its foundational principles, and practical significance (around 10 to 15 sentences). 
    2. Key Formulas & Concepts: Create a list of the core theorems, concepts, equations, or key terms found in the text. If an equation exists, provide its LaTeX representation. If no math formula applies, set "formula" to null.
    3. Worked Examples: Provide 5 step-by-step solved problems or case studies demonstrating clear logical progression from question to final answer.
    4. Why it Matters: Explain real-world engineering, industry, or scientific applications of these principles. 
    5. Study Quiz: Provide 10 multiple-choice questions (with 1 correct answer and 2 distractors each). At least five question must require the student to calculate an answer. Include a solution explanation for each question. 
    6. Never use ASCII symbols like * for multiplication in math equations; always use LaTeX commands like \times or \cdot. ALWAYS wrap every formula or variable expression in Markdown dollar signs ($ ... $), even inside normal text explanations.
    CRITICAL CONSTRAINTS:
    - Keep the tone encouraging, structured, and academically rigorous.
    - JSON & LaTeX Safety: If you use LaTeX for formulas, you MUST double-escape all backslashes (e.g., write `\\\\frac{{a}}{{b}}` instead of `\\frac{{a}}{{b}}` or `\\\\lambda` instead of `\\lambda`) so the output remains valid JSON.
    - Return your response strictly as a valid JSON object without any markdown code blocks (like ```json).
    
    Exact JSON Schema:
    {{
        "content": "return a clean version of: {content_str}, with laTeX formulas, wrap every formula or variable expression in Markdown dollar signs ($ ... $), even inside normal text explanations. Preserve all text",
        "summary": "...",
        "key_concepts": [
            {{
                "term": "Name of concept, theorem, or equation", 
                "formula": "LaTeX formula string (double-escaped) OR null if non-mathematical", no ASCII, 
                "definition": "Plain language explanation and variable breakdown", 
                "overview_and_relevance": "How and when to apply this concept"
            }}
        ],
        "worked_examples": [
            {{
                "problem_statement": "Clear problem question or analytical scenario...",
                "step_by_step_solution": [
                    "Step 1: Identify given parameters or context...",
                    "Step 2: Apply core principle or formula...",
                    "Step 3: Calculate or derive solution...",
                    "Step 4: Arrive at final result..."
                ],
                "final_answer": "Final numeric, algebraic, or theoretical answer"
            }}
        ],
        "why_it_matters": "...",
        "study_quiz": [
            {{
                "question": "...", 
                "answers": [
                    {{"text": "...", "correct": true}}, 
                    {{"text": "...", "correct": false}}, 
                    {{"text": "...", "correct": false}}
                ],
                "solution_explanation": "Brief explanation proving why the correct answer is right."
            }}
        ]
    }}
    """

    response = model.generate_content(prompt)
    
    try:
        return extract_json(response.text)
    except Exception as e:
        print(f"Failed to parse study guide JSON: {e}")
        return {
            "content": content_str,
            "summary": "Error generating structured summary. Please retry.",
            "key_formulas_and_concepts": [],
            "worked_examples": [],
            "why_it_matters": "An error occurred during AI processing.",
            "study_quiz": []
        }

def study2(topic, chunks):
    items_to_process = chunks if isinstance(chunks, list) else [chunks]
    text_contents = []

    for item in items_to_process:
        if isinstance(item, dict):
            # Handle structured dictionaries like {'type': 'image', 'content': 'path'}
            if item.get('type') == 'image' and 'content' in item:
                ocr_text = extract_text_from_image(item['content'])
                if ocr_text:
                    text_contents.append(ocr_text)
            elif 'content' in item:
                text_contents.append(str(item['content']))
            else:
                text_contents.append(str(item))
        elif isinstance(item, str):
            # Check if the string is a file path pointing to a saved image on disk
            if os.path.exists(item) and any(item.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']):
                ocr_text = extract_text_from_image(item)
                if ocr_text:
                    text_contents.append(ocr_text)
            else:
                text_contents.append(item)
        else:
            text_contents.append(str(item))

    # Join all extracted text into a single clean string (only text is retained!)
    content_str = "\n\n".join(text_contents).strip()
    if not content_str:
        content_str = "No readable content or text could be extracted."
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
    Exact JSON Schema:
        {{
            "content": "return a clean version of: {content_str}",
            "summary": "...",
            "key_concepts": [
                {{
                    "term": "Name of concept, theorem, or equation", 
                   
                    "definition": "Plain language explanation and variable breakdown", 
                    "overview_and_relevance": "How and when to apply this concept"
                }}
            ],
            
            
            "why_it_matters": "...",
            "study_quiz": [
                {{
                    "question": "...", 
                    "answers": [
                        {{"text": "...", "correct": true}}, 
                        {{"text": "...", "correct": false}}, 
                        {{"text": "...", "correct": false}}
                    ],
                    "solution_explanation": "Brief explanation proving why the correct answer is right."
                }}
            ]
        }}
        """
    response = model.generate_content(prompt)
    return extract_json(response.text)

def ask(question, chunks, history=None):
    client = google_genai.Client(api_key=api_key)
    prompt = f"Answer this question: {question} based on: {chunks}. Keep your answer short and concise (100 to 200 words)."
    
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
            if pix.n - pix.alpha >= 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            
            img_filename = f"{course_name}_p{page_idx}_i{img_idx}.png"
            img_path = os.path.join(IMAGE_STORAGE_DIR, img_filename)
            pix.save(img_path)
            
            data_list.append({'type': 'image', 'content': img_path})
            
    return data_list

def update_relevance_map(course_name, past_questions_text, upload_dir=UPLOAD_DIR):
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
    
    Task:
    1. Analyze the past questions and map them to the course topics.
    2. Determine the relevance of each topic ("High", "Medium", or "Low") based on frequency.
    3. Extract all past questions so they can be added to generated exams.
    
    Return ONLY a valid JSON object matching this exact schema:
    {{
        "topic_relevance": {{
            "Topic Name 1": "High",
            "Topic Name 2": "Medium"
        }},
        "past_questions": [
            {{
                "question": "Full past question text",
                "rubric": "Expected answer key or grading criteria",
                "type": "objective"
            }}
        ]
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        relevance_data = extract_json(response.text)
        relevance_file = os.path.join(upload_dir, f"{course_name}_relevance.json")
        with open(relevance_file, 'w', encoding='utf-8') as f:
            json.dump(relevance_data, f, indent=4)
            
        return relevance_data
        
    except Exception as e:
        raise ValueError(f"Failed to parse AI relevance map: {e}")

def generate_adaptive_exam(course_name, upload_dir=UPLOAD_DIR):
    topics_file = os.path.join(upload_dir, f"{course_name}topics.csv")
    if not os.path.exists(topics_file):
        raise FileNotFoundError(f"Course material for '{course_name}' not found. Please upload materials first.")
    
    with open(topics_file, 'r', encoding='utf-8') as f:
        course_content = f.read()

    relevance_file = os.path.join(upload_dir, f"{course_name}_relevance.json")
    relevance_data = "No past question data available. Distribute questions evenly across topics."
    
    if os.path.exists(relevance_file):
        with open(relevance_file, 'r', encoding='utf-8') as f:
            relevance_data = f.read()

    prompt = f"""
    You are an expert university professor. Create an adaptive exam for the course: '{course_name}'.
    
    Course material: 
    {course_content}
    
    Historical Relevance & Past Questions Data: 
    {relevance_data}
    
    CRITICAL INSTRUCTIONS:
    1. Directly include all extracted questions found under 'past_questions' in the historical relevance data.
    2. Use those past questions as templates for style and difficulty level.
    3. Generate new questions to make up the total count, prioritizing topics with 'High' and 'Medium' relevance.
    4. You MUST return EXACTLY 20 exam questions in total (15 'objective', 5 'essay').
    5. Return ONLY a valid JSON array.
    6. Never use ASCII symbols like * for multiplication in math equations; always use LaTeX commands like \times or \cdot. ALWAYS wrap every formula or variable expression in Markdown dollar signs ($ ... $), even inside normal text explanations.

    Schema for each question:
    {{
        "id": "uuid",
        "question": "The question text",
        "rubric": "Grading criteria or expected answer points and correct answer",
        "type": "essay" OR "objective"
    }}
    """
    
    response = model.generate_content(prompt)
    
    try:
        questions = extract_json(response.text)
        
        for q in questions:
            if 'id' not in q or len(str(q['id'])) < 5:
                q['id'] = str(uuid.uuid4())
                
        return questions
    except Exception as e:
        print(f"Failed to parse AI exam output: {e}")
        raise ValueError("The AI generated an invalid exam format. Please try again.")

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
    try:
        return extract_json(response.text)
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
    2. Return ONLY a valid JSON array of objects following this schema:
    [
      {{
        "index": 0,
        "score": 8,
        "feedback": "Your explanation was correct but missed..."
      }}
    ]
    """
    
    try:
        response = model.generate_content(prompt)
        return extract_json(response.text)
    except Exception as e:
        print(f"Batch grading failed: {e}")
        return [{"index": i, "score": 0, "feedback": "Error processing grading for this item."} for i in range(len(original_exam))]

def clarify_exam_item(question, user_answer, correct_answer, feedback, user_query):
    """Provides targeted tutoring to explain why a student lost points on a specific exam question."""
    prompt = f"""
    You are an encouraging and insightful university tutor helping a student review their exam results.
    
    Context of the exam item:
    - Question Asked: "{question}"
    - Student's Answer: "{user_answer}"
    - Expected Answer / Rubric: "{correct_answer}"
    - Examiner's Initial Feedback: "{feedback}"
    
    Student's Follow-up Question/Doubt: "{user_query}"
    
    Task: Answer the student's follow-up question directly. Explain clearly and gently why their original answer received the score it did.
    Keep your explanation concise, friendly, and under 150 words.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Clarification error: {e}")
        return "I'm having trouble analyzing this question right now. Please try again in a moment."

# Assuming you have an initialized AI client (e.g., OpenAI, Gemini, etc.) 
# replace `ai_model_generate` with your actual AI calling method.

def get_semantic_definition(highlighted_text):
    """
    Provides a clear, definition for a specific excerpt of text.
    """
    prompt = (
        f"You are an intelligent assistant. Please provide a clear, concise, "
        f"and contextually accurate definition or explanation for the following text excerpt:\n\n"
        f"\"{highlighted_text}\"\n\n"
        f"Keep the explanation easy to understand for a reader."
        f"return the definition and possible use cases in a single paragraph without any additional commentary."
    )
    
    # Example using a generic generation call:
    # response = ai_client.generate(prompt)
    # return response.text
    
    response = model.generate_content(prompt) 
    return response.text.strip()


def answer_document_question(highlighted_text, question, pdf_name, history):
    """
    Answers a specific user query based ONLY on the provided highlighted text,
    maintaining conversational context from the history.
    """
    client = google_genai.Client(api_key=api_key)
    
    # 1. Define the system instruction to anchor the AI's persona and context
    system_instruction = (
        f"You are an expert, friendly AI Study Assistant embedded inside a PDF reading application.\n"
        f"Document Title: '{pdf_name}'\n"
        f"Highlighted Snippet: \"{highlighted_text}\"\n\n"
        f"Guidelines:\n"
        f"1. Base answers primarily on the highlighted text and document context.\n"
        f"2. Pay close attention to the provided chat history to understand follow-up questions.\n"
        f"3. Keep explanations clear, encouraging, and educational."
    )
    
    # 2. Build the contents array for the Gemini API
    formatted_contents = []
    
    if history:
        for msg in history:
            # Map frontend roles ('user' or 'ai') to Gemini roles ('user' or 'model')
            role = "user" if msg.get('role') == "user" else "model"
            formatted_contents.append({"role": role, "parts": [{"text": msg.get('content', '')}]})
            
    # 3. Append the current user question
    formatted_contents.append({"role": "user", "parts": [{"text": question}]})
    
    # 4. Generate the response
    response = model.generate_content(formatted_contents)
    return response.text.strip()
