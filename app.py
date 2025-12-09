from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from llm_inference.llm import LLM_Inference
from labelling_scores.llm import LLM
from labelling_scores.image_to_url import local_image_to_data_url

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests from frontend

LLM_INF = LLM_Inference()  # Initialize LLM client
LLM_model = LLM()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_palm(image_path):
    """
    TODO: Replace with your actual hand analysis model
    For now, returns mock data
    """
    import random

    ### still hard coded, replace with model output
    scores = {
        "strength": round(random.uniform(0.1, 0.9), 2),
        "romantic": round(random.uniform(0.1, 0.9), 2),
        "luck": round(random.uniform(0.1, 0.9), 2),
        "potential": round(random.uniform(0.1, 0.9), 2),
    }

    user_prompt = LLM_INF.get_user_prompt(scores=scores)
    text = LLM_model.get_LLM_output(
        user_prompt=user_prompt,
    )

    # text = (
    #     "Your palm suggests balanced inner strength, a warm romantic nature, "
    #     "favorable luck in key moments, and strong potential for growth across "
    #     "multiple areas of life."
    # )
    
    return {
        "score": scores,
        "text": text
    }

@app.route('/api/analyze-palm', methods=['POST'])
def analyze():
    """Main API endpoint for palm analysis"""
    # Check if file is present
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': 'No file selected'
        }), 400
    
    # Validate file
    if not allowed_file(file.filename):
        return jsonify({
            'status': 'error',
            'message': 'Invalid file type. Allowed: png, jpg, jpeg, gif'
        }), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the palm image
        analysis_result = analyze_palm(filepath)
        
        # Return results
        return jsonify({
            'status': 'success',
            'data': analysis_result
        }), 200
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Palm.AI API is running'
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
