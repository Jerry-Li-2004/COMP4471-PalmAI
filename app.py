from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests from frontend

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
    
    # This is where you'll integrate your ML model
    # Example: model.predict(image_path)
    
    
    return {
        'personality': {
            'title': '👤 Personality',
            'description': 'You are naturally intuitive and creative. Your balanced palm lines suggest good emotional stability and adaptability.',
            'confidence': random.randint(75, 95)
        },
        'career': {
            'title': '💼 Career Potential',
            'description': 'Well-suited for leadership roles and independent ventures.',
            'confidence': random.randint(75, 95)
        },
        'relationships': {
            'title': '❤️ Relationships',
            'description': 'Prominent heart line shows strong emotional capacity. You value deep connections and loyalty in relationships.',
            'confidence': random.randint(75, 95)
        },
        'health': {
            'title': '🏥 Wellness',
            'description': 'Overall indicators suggest good vitality and energy. Regular exercise and balance will support long-term wellness.',
            'confidence': random.randint(75, 95)
        },
        'destiny': {
            'title': '✨ Life Path',
            'description': 'Fate line suggests you are destined for meaningful achievements. Trust your instincts and pursue your passions.',
            'confidence': random.randint(75, 95)
        },
        'strengths': {
            'title': '⭐ Key Strengths',
            'description': 'Intuition, creativity, resilience, emotional intelligence, and natural leadership abilities.',
            'confidence': random.randint(75, 95)
        }
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
