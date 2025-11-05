import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

class SVMTrainer:
    def __init__(self, csv_path):
        """Initialize SVM trainer with CSV data path"""
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.model = None
        
    def load_data(self):
        """Load and prepare CSV data"""
        print("Loading CSV data...")
        df = pd.read_csv(self.csv_path)
        
        # Features: all columns except 'fake'
        feature_columns = [col for col in df.columns if col != 'fake']
        X = df[feature_columns].values
        y = df['fake'].values
        
        print(f"Dataset shape: {X.shape}")
        print(f"Fake accounts: {sum(y)}, Real accounts: {len(y) - sum(y)}")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """Split and scale the data (80-20 split)"""
        print("\nSplitting data (80% train, 20% test)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train SVM model"""
        print("\nTraining SVM model...")
        self.model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,  # Enable probability estimates
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        print("SVM training completed!")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("\nEvaluating SVM model...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Real', 'Fake']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return accuracy
    
    def save_model(self, model_dir='models'):
        """Save trained model and scaler"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, 'svm_model.pkl')
        scaler_path = os.path.join(model_dir, 'svm_scaler.pkl')
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def run_pipeline(self):
        """Execute complete training pipeline"""
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        self.train_model(X_train, y_train)
        accuracy = self.evaluate_model(X_test, y_test)
        self.save_model()
        
        return accuracy

if __name__ == "__main__":
    # Run SVM training
    trainer = SVMTrainer('data/train_csv.csv')
    accuracy = trainer.run_pipeline()
    print(f"\n{'='*50}")
    print(f"SVM Training Complete! Final Accuracy: {accuracy:.4f}")
    print(f"{'='*50}")