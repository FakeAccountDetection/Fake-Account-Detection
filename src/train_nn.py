import json
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class NeuralNetworkTrainer:
    def __init__(self, json_path, max_words=5000, max_len=100):
        """Initialize NN trainer with JSON data path"""
        self.json_path = json_path
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        """Clean and preprocess text"""
        if not text or text == '':
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def load_and_process_data(self):
        """Load JSON data and create text features"""
        print("Loading JSON data...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for account in data:
            # Combine all text fields
            combined_text = []
            
            # Add bio
            if account.get('bio'):
                combined_text.append(self.clean_text(account['bio']))
            
            # Add captions
            if account.get('captions'):
                for caption in account['captions']:
                    combined_text.append(self.clean_text(caption))
            
            # Add comments
            if account.get('comments'):
                for comment in account['comments']:
                    combined_text.append(self.clean_text(comment))
            
            # Join all text
            full_text = ' '.join(combined_text)
            
            if full_text.strip():  # Only add non-empty texts
                texts.append(full_text)
                labels.append(account['fake'])
        
        print(f"Processed {len(texts)} accounts")
        print(f"Fake accounts: {sum(labels)}, Real accounts: {len(labels) - sum(labels)}")
        
        return texts, np.array(labels)
    
    def prepare_sequences(self, texts, is_training=True):
        """Convert texts to sequences"""
        if is_training:
            self.tokenizer.fit_on_texts(texts)
        
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=self.max_len, padding='post', truncating='post')
        
        return padded
    
    def build_model(self):
        """Build LSTM neural network"""
        print("\nBuilding Neural Network...")
        
        self.model = Sequential([
            Embedding(self.max_words, 128, input_length=self.max_len),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.5),
            Bidirectional(LSTM(32)),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print(self.model.summary())
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """Train the neural network"""
        print("\nTraining Neural Network...")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ModelCheckpoint('models/nn_model_best.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating Neural Network...")
        
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save model and tokenizer"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, 'nn_model.h5')
        tokenizer_path = os.path.join(model_dir, 'nn_tokenizer.pkl')
        
        self.model.save(model_path)
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        print(f"\nModel saved to {model_path}")
        print(f"Tokenizer saved to {tokenizer_path}")
    
    def run_pipeline(self):
        """Execute complete training pipeline"""
        # Load and process data
        texts, labels = self.load_and_process_data()
        
        # Split data (80-20)
        print("\nSplitting data (80% train, 20% test)...")
        texts_train, texts_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Further split training into train and validation
        texts_train, texts_val, y_train, y_val = train_test_split(
            texts_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        # Prepare sequences
        print("Preparing sequences...")
        X_train = self.prepare_sequences(texts_train, is_training=True)
        X_val = self.prepare_sequences(texts_val, is_training=False)
        X_test = self.prepare_sequences(texts_test, is_training=False)
        
        # Build and train model
        self.build_model()
        history = self.train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate
        accuracy = self.evaluate_model(X_test, y_test)
        
        # Save
        self.save_model()
        
        return accuracy, history

if __name__ == "__main__":
    # Run NN training
    trainer = NeuralNetworkTrainer('data/train_nlp.json')
    accuracy, history = trainer.run_pipeline()
    print(f"\n{'='*50}")
    print(f"Neural Network Training Complete! Final Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")