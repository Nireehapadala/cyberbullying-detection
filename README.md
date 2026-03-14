# Toxic Comment Detection System

A comprehensive Machine Learning project for detecting toxic, abusive, and harmful text messages using Natural Language Processing and Logistic Regression classification.

## 🚀 Features

- **Machine Learning Powered**: Uses Logistic Regression with 99.16% accuracy
- **Real-time Analysis**: Instant detection with confidence scores
- **Modern Web Interface**: Clean, responsive design with animations
- **REST API**: Flask backend with POST /predict endpoint
- **Comprehensive Text Preprocessing**: Enhanced NLP pipeline with TF-IDF vectorization
- **Multi-Category Detection**: Detects various types of abusive language
- **Confidence Scoring**: Shows prediction confidence percentage
- **Warning System**: Displays alerts for toxic content detection

## 📊 Model Performance

The final comprehensive model achieved outstanding performance metrics:
- **Accuracy**: 99.16%
- **Precision**: 99% (both classes)
- **Recall**: 99% (both classes)
- **F1-Score**: 99% (both classes)
- **Dataset Size**: 4,182 balanced samples

## 🎯 Detection Categories

The model is trained to detect comprehensive abusive language including:

### ✅ Sexual Insults & Prostitution
- slut, whore, prostitute, hooker, escort, bitch, cunt, etc.

### ✅ Body Shaming
- fat, obese, ugly, disgusting, pig, whale, cow, etc.

### ✅ Mental Health Insults
- mental, psycho, crazy, insane, schizo, lunatic, etc.

### ✅ Racial Slurs
- nigger, paki, chink, gook, kike, etc.

### ✅ Homophobic Slurs
- faggot, fag, queer, dyke, tranny, etc.

### ✅ General Insults
- fuck, shit, bastard, asshole, motherfucker, etc.

### ✅ Threats
- kill yourself, go die, rot in hell, etc.
## 🛠️ Technology Stack

### Machine Learning
- **Python 3.x**
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **joblib**: Model serialization

### Backend
- **Flask**: REST API framework

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with animations
- **JavaScript**: Dynamic functionality
- **Font Awesome**: Icons

## 📁 Project Structure

```
mlneu/
├── app.py                    # Flask REST API
├── train_final_model.py      # Final model training script
├── prepare_datasets.py       # Dataset preparation script
├── requirements.txt         # Python dependencies
├── README.md                 # Project documentation
├── data/
│   ├── dataset.csv          # Final comprehensive dataset
│   ├── train.csv            # Training split
│   ├── val.csv              # Validation split
│   └── test.csv             # Test split
├── models/
│   ├── toxic_classifier.pkl # Final trained model
│   └── vectorizer.pkl       # TF-IDF vectorizer
├── static/
│   ├── style.css            # Frontend styling
│   └── script.js            # Frontend JavaScript
└── templates/
    └── index.html           # Frontend HTML template
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset and Train Model

```bash
python prepare_datasets.py
python train_final_model.py
```

This will:
- Create comprehensive abusive language dataset
- Train optimized Logistic Regression model
- Achieve 99.16% accuracy
- Save model and vectorizer

### 3. Run the Flask Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Open in Browser

Navigate to `http://localhost:5000` to access the web interface.

## 📡 API Usage

### POST /predict

**Endpoint**: `POST /predict`

**Content-Type**: `application/json`

**Request Body**:
```json
{
    "text": "Your message here"
}
```

**Response**:
```json
{
    "text": "Your message here",
    "prediction": 1,
    "label": "Toxic Content Detected",
    "confidence": 89.90,
    "is_toxic": true,
    "is_cyberbullying": true,
    "model_type": "Enhanced Logistic Regression"
}
```

**Response Fields**:
- `text`: Original message
- `prediction`: 0 (safe) or 1 (toxic)
- `label`: Human-readable classification
- `confidence`: Prediction confidence percentage
- `is_toxic`: Boolean result
- `is_cyberbullying`: Boolean result (for compatibility)

### GET /health

**Endpoint**: `GET /health`

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

## 🧠 Model Details

### Text Preprocessing Pipeline

1. **Lowercasing**: Convert text to lowercase
2. **URL Removal**: Remove URLs and links
3. **Mention/Hashtag Removal**: Remove @mentions and #hashtags
4. **Punctuation Removal**: Remove special characters (keep basic punctuation)
5. **Whitespace Normalization**: Remove extra spaces

### Machine Learning Model

- **Algorithm**: Logistic Regression (optimized)
- **Features**: 30,000 TF-IDF features
- **N-grams**: 1-5 word sequences
- **Regularization**: L2 penalty with C=3.0
- **Class Weight**: Balanced for equal treatment

### Feature Engineering

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **N-grams**: 1-5 grams for context understanding
- **Max Features**: 30,000 most important features
- **Sublinear TF**: Logarithmic TF scaling

## 🎨 Frontend Features

- **Responsive Design**: Works on desktop and mobile devices
- **Real-time Character Count**: Shows remaining characters (max 500)
- **Loading States**: Visual feedback during analysis
- **Color-coded Results**: Green for safe, red for toxic content
- **Confidence Display**: Shows prediction confidence percentage
- **Warning Messages**: Alerts for detected toxic content
- **Smooth Animations**: Modern UI transitions and effects
- **Error Handling**: User-friendly error messages

## 🔧 Configuration

### Model Parameters

- **Algorithm**: Logistic Regression
- **TF-IDF Max Features**: 30,000
- **N-gram Range**: (1, 5) for 1-5 word sequences
- **Regularization**: C=3.0
- **Max Iterations**: 3000
- **Class Weight**: Balanced

### API Configuration

- **Host**: 0.0.0.0 (accessible from any IP)
- **Port**: 5000
- **Debug Mode**: Enabled for development

## 📈 Dataset

The final dataset contains 4,182 balanced labeled text messages:
- **2,091 Toxic Messages**: Various categories of abusive content
- **2,091 Safe Messages**: Positive or neutral content

**Categories Covered**:
- Sexual insults & prostitution terms
- Body shaming & appearance insults
- Mental health shaming
- Racial slurs
- Homophobic slurs
- General insults
- Threats

**Sample Labels**:
- `0`: Safe Message
- `1`: Toxic Content Detected

## 🔄 Model Training Process

1. **Dataset Creation**: Generate comprehensive abusive language dataset
2. **Text Preprocessing**: Clean and normalize text
3. **Feature Extraction**: Convert text to TF-IDF features
4. **Train-Test Split**: 80% training, 20% testing
5. **Model Training**: Train optimized Logistic Regression
6. **Performance Evaluation**: 99.16% accuracy achieved
7. **Model Serialization**: Save with joblib

## 🚀 Deployment

### Production Deployment

1. **Install Production Server**:
```bash
pip install gunicorn
```

2. **Run with Gunicorn**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t toxic-comment-detector .
docker run -p 5000:5000 toxic-comment-detector
```

## 🧪 Testing

### Test the API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are a stupid prostitute"}'
```

### Test with Python

```python
import requests

response = requests.post('http://localhost:5000/predict', 
                        json={'text': 'You are amazing!'})
print(response.json())
```

## 🔒 Security Considerations

- **Input Validation**: Text length limited to 500 characters
- **Error Handling**: Proper error responses for invalid requests
- **Model Security**: No sensitive data in model files
- **Content Filtering**: Comprehensive abusive language detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🎯 Future Enhancements

- **Larger Dataset**: Train on more diverse data
- **Deep Learning**: Implement BERT or transformer models
- **Multi-language Support**: Support other languages
- **Real-time Monitoring**: Live dashboard for analysis
- **User Authentication**: User accounts and history
- **Batch Processing**: Analyze multiple messages
- **Export Features**: Download analysis reports

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Built with ❤️ for safer online communities**
