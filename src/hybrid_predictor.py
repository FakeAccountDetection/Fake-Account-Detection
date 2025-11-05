import numpy as np
import joblib
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

class HybridPredictor:
    def __init__(self, model_dir='models'):
        """Initialize hybrid predictor with trained models"""
        self.model_dir = model_dir
        
        # Load SVM model and scaler
        print("Loading SVM model...")
        self.svm_model = joblib.load(f'{model_dir}/svm_model.pkl')
        self.svm_scaler = joblib.load(f'{model_dir}/svm_scaler.pkl')
        
        # Load NN model and tokenizer
        print("Loading Neural Network model...")
        self.nn_model = load_model(f'{model_dir}/nn_model.h5')
        with open(f'{model_dir}/nn_tokenizer.pkl', 'rb') as f:
            self.nn_tokenizer = pickle.load(f)
        
        # Initialize text processing
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
        
        self.max_len = 100  # Same as training
        
        print("Hybrid Predictor initialized successfully!")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text or text == '':
            return ''
        
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    
    def predict_svm(self, structured_features):
        """
        Predict using SVM model
        
        Args:
            structured_features: List/array of 12 features matching CSV structure
            [profile_pic, nums/length_username, fullname_words, nums/length_fullname,
             name==username, description_length, external_URL, private, 
             #posts, #followers, #follows, fake]
        
        Returns:
            probability of being fake (0-1)
        """
        # Ensure correct shape
        features = np.array(structured_features).reshape(1, -1)
        
        # Scale features
        features_scaled = self.svm_scaler.transform(features)
        
        # Get probability
        prob = self.svm_model.predict_proba(features_scaled)[0]
        
        return prob[1]  # Probability of being fake
    
    def predict_nn(self, text_data):
        """
        Predict using Neural Network
        
        Args:
            text_data: Dictionary with keys 'bio', 'captions', 'comments'
                      Each can be a string or list of strings
        
        Returns:
            probability of being fake (0-1)
        """
        # Combine all text
        combined_text = []
        
        if text_data.get('bio'):
            combined_text.append(self.clean_text(text_data['bio']))
        
        if text_data.get('captions'):
            if isinstance(text_data['captions'], list):
                for caption in text_data['captions']:
                    combined_text.append(self.clean_text(caption))
            else:
                combined_text.append(self.clean_text(text_data['captions']))
        
        if text_data.get('comments'):
            if isinstance(text_data['comments'], list):
                for comment in text_data['comments']:
                    combined_text.append(self.clean_text(comment))
            else:
                combined_text.append(self.clean_text(text_data['comments']))
        
        full_text = ' '.join(combined_text)
        
        # Convert to sequence
        sequence = self.nn_tokenizer.texts_to_sequences([full_text])
        padded = pad_sequences(sequence, maxlen=self.max_len, padding='post', truncating='post')
        
        # Predict
        prob = self.nn_model.predict(padded, verbose=0)[0][0]
        
        return float(prob)
    
    def predict_hybrid(self, structured_features, text_data, svm_weight=0.6, nn_weight=0.4):
        """
        Hybrid prediction combining SVM and NN
        
        Args:
            structured_features: List of 12 numeric features for SVM
            text_data: Dictionary with text fields for NN
            svm_weight: Weight for SVM prediction (default 0.6)
            nn_weight: Weight for NN prediction (default 0.4)
        
        Returns:
            Dictionary with detailed results
        """
        # Get individual predictions
        svm_prob = self.predict_svm(structured_features)
        nn_prob = self.predict_nn(text_data)
        
        # Weighted combination
        hybrid_prob = (svm_weight * svm_prob) + (nn_weight * nn_prob)
        
        # Determine classification
        is_fake = hybrid_prob > 0.5
        confidence = hybrid_prob if is_fake else (1 - hybrid_prob)
        
        return {
            'is_fake': bool(is_fake),
            'confidence': float(confidence),
            'hybrid_probability': float(hybrid_prob),
            'svm_probability': float(svm_prob),
            'nn_probability': float(nn_prob),
            'classification': 'FAKE' if is_fake else 'REAL'
        }

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = HybridPredictor()
    
    # Example: Test with sample data
    # Structured features (12 values from CSV)
    sample_features = [1, 0.27, 0, 0.0, 0, 53, 0, 0, 32, 1000, 955, 0]
    
    # Text data
    sample_text = {
        'bio': 'Be your own kind of beautiful;)',
        'captions': ['Good morning', 'They are soo cute'],
        'comments': ['']
    }
    
    # Make prediction
    result = predictor.predict_hybrid(sample_features[:-1], sample_text)  # Exclude 'fake' column
    
    print("\nPrediction Results:")
    print(f"Classification: {result['classification']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Hybrid Probability: {result['hybrid_probability']:.2%}")
    print(f"SVM Probability: {result['svm_probability']:.2%}")
    print(f"NN Probability: {result['nn_probability']:.2%}")