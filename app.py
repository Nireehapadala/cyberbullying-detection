from flask import Flask, request, jsonify, render_template
import joblib
import re
import string
import sqlite3
import os
from datetime import datetime
import nltk
nltk.download('punkt')
nltk.download('stopwords')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "detection_history.db")
app = Flask(__name__)

# Load enhanced model and vectorizer
try:
    model = joblib.load('models/toxic_classifier.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("Enhanced model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: Enhanced model files not found. Please run train_enhanced_model.py first.")
    # Fallback to old model
    try:
        model = joblib.load('models/model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        print("Fallback: Using original model and vectorizer!")
    except FileNotFoundError:
        print("Error: No model files found. Please train a model first.")
        model = None
        vectorizer = None

# Database setup
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table if not exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            toxicity_probability REAL NOT NULL,
            is_toxic BOOLEAN NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ip_address TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    # Check if database has existing records
    try:
        check_conn = sqlite3.connect(DB_PATH)
        check_cursor = check_conn.cursor()
        check_cursor.execute("SELECT COUNT(*) FROM detection_history")
        count = check_cursor.fetchone()[0]
        check_conn.close()
        print(f"Database initialized successfully! Existing records: {count}")
    except Exception as e:
        print(f"Error checking existing records: {e}")

# Initialize database on startup
init_db()
print("DEBUG: Database initialized at startup")

# Function to save detection to database
def save_detection(original_text, processed_text, prediction, label, confidence, toxicity_probability, is_toxic, ip_address):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO detection_history 
            (original_text, processed_text, prediction, label, confidence, toxicity_probability, is_toxic, ip_address)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (original_text, processed_text, prediction, label, confidence, toxicity_probability, is_toxic, ip_address))
        
        conn.commit()
        conn.close()
        print(f"DEBUG: Saved detection - '{original_text[:30]}...' as {label}")
        return True
    except Exception as e:
        print(f"DEBUG: Error saving to database: {e}")
        return False

# Function to get detection history
def get_detection_history(limit=50):
    try:
        conn = sqlite3.connect(DB_PATH)
        print("DEBUG: Using database file:", os.path.abspath('detection_history.db'))
        cursor = conn.cursor()
        
        print(f"DEBUG: Connected to database, fetching limit={limit}")
        cursor.execute('''
            SELECT 
                id,
                original_text,
                processed_text,
                prediction,
                label,
                confidence,
                toxicity_probability,
                is_toxic,
                strftime('%Y-%m-%d %H:%M:%S', timestamp) as formatted_timestamp,
                ip_address
            FROM detection_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        history = cursor.fetchall()
        print(f"DEBUG: Database query returned {len(history)} records")
        
        conn.close()
        
        # Helper to safely convert BLOB or int to Python int
        def to_int(val):
            if val is None:
                return None
            if isinstance(val, (bytes, bytearray)):
                return int.from_bytes(val, byteorder='little')
            return int(val)
        
        # Convert to JSON-serializable format
        serializable_history = []
        for record in history:
            serializable_history.append({
                'id': to_int(record[0]),
                'original_text': str(record[1]) if record[1] is not None else None,
                'processed_text': str(record[2]) if record[2] is not None else None,
                'prediction': to_int(record[3]),
                'label': str(record[4]) if record[4] is not None else None,
                'confidence': round(float(record[5]) * 100, 2) if record[5] is not None else None,
                'toxicity_probability': round(float(record[6]) * 100, 2) if record[6] is not None else None,
                'is_toxic': bool(to_int(record[7])) if record[7] is not None else None,
                'timestamp': str(record[8]) if record[8] is not None else None,
                'ip_address': str(record[9]) if record[9] is not None else None
            })
        
        print(f"DEBUG: Returning {len(serializable_history)} serializable records")
        return serializable_history
    except Exception as e:
        print(f"DEBUG: Error fetching history: {e}")
        return []
# Enhanced text preprocessing function (matching the training pipeline)
def preprocess_text(text):
    if not text:
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions and hashtags
    text = re.sub(r'@\w+|\#\w+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train model first.'}), 500
    
    try:
        # Get text from request
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Vectorize text
        text_vector = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = model.predict(text_vector)[0]
        prediction_proba = model.predict_proba(text_vector)[0]
        
        # Get confidence and toxicity probability
        confidence = max(prediction_proba)
        toxicity_probability = prediction_proba[1] if len(prediction_proba) > 1 else 0.0
        
        # Get client IP address
        ip_address = request.remote_addr
        
        # Prepare response
        result = {
            'text': text,
            'processed_text': processed_text,
            'prediction': int(prediction),
            'label': 'Toxic Content Detected' if prediction == 1 else 'Safe Message',
            'confidence': round(confidence * 100, 2),
            'toxicity_probability': round(toxicity_probability * 100, 2),
            'is_toxic': bool(prediction == 1),
            'is_cyberbullying': bool(prediction == 1),  # Add this for JavaScript compatibility
            'model_type': 'Enhanced Logistic Regression'
        }
        
        # Save to database
        save_detection(
            text, 
            processed_text, 
            prediction, 
            result['label'], 
            confidence, 
            toxicity_probability, 
            bool(prediction == 1), 
            ip_address
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        limit = request.args.get('limit', 50, type=int)
        print(f"DEBUG: History requested with limit={limit}")
        history = get_detection_history(limit)
        print(f"DEBUG: Retrieved {len(history)} records from database")
        
        response_data = {
            'success': True,
            'history': history,
            'total': len(history)
        }
        print(f"DEBUG: Response data: {response_data}")
        
        return jsonify(response_data), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        print(f"DEBUG: Error in history endpoint: {e}")
        return jsonify({'error': str(e)}), 500, {'Content-Type': 'application/json'}

@app.route('/clear-history', methods=['DELETE'])
def clear_history():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM detection_history')
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'History cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:history_id>', methods=['PUT'])
def update_history(history_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get fields to update
        original_text = data.get('original_text')
        label = data.get('label')
        confidence = data.get('confidence')
        toxicity_probability = data.get('toxicity_probability')
        is_toxic = data.get('is_toxic')
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Build update query dynamically
        update_fields = []
        update_values = []
        
        if original_text is not None:
            update_fields.append('original_text = ?')
            update_values.append(original_text)
        if label is not None:
            update_fields.append('label = ?')
            update_values.append(label)
        if confidence is not None:
            update_fields.append('confidence = ?')
            update_values.append(confidence)
        if toxicity_probability is not None:
            update_fields.append('toxicity_probability = ?')
            update_values.append(toxicity_probability)
        if is_toxic is not None:
            update_fields.append('is_toxic = ?')
            update_values.append(is_toxic)
        
        if update_fields:
            update_query = f"UPDATE detection_history SET {', '.join(update_fields)} WHERE id = ?"
            update_values.append(history_id)
            
            cursor.execute(update_query, update_values)
            conn.commit()
            
            # Get updated record
            cursor.execute('SELECT * FROM detection_history WHERE id = ?', (history_id,))
            updated_record = cursor.fetchone()
            conn.close()
            
            return jsonify({
                'success': True,
                'message': 'History item updated successfully',
                'updated_record': {
                    'id': updated_record[0],
                    'original_text': updated_record[1],
                    'label': updated_record[4],
                    'confidence': updated_record[5],
                    'is_toxic': bool(updated_record[7])
                }
            })
        else:
            return jsonify({'error': 'No valid fields to update'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history/<int:history_id>', methods=['DELETE'])
def delete_history_item(history_id):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get the item before deleting
        cursor.execute('SELECT * FROM detection_history WHERE id = ?', (history_id,))
        item = cursor.fetchone()
        
        if not item:
            conn.close()
            return jsonify({'error': 'History item not found'}), 404
        
        # Delete the item
        cursor.execute('DELETE FROM detection_history WHERE id = ?', (history_id,))
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': 'History item deleted successfully',
            'deleted_item': {
                'id': item[0],
                'original_text': item[1],
                'label': item[4]
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == "__main__":
    app.run()
