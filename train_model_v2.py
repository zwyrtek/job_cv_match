"""
Train Neural Network for Job-CV Matching - VERSION 2
Updated for new data format with Stage value 0-10
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import json

class JobCVMatcherV2:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.job_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.resume_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.skills_vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
        
    def preprocess_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text) or text == '' or text == 'nan':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def extract_skills(self, text):
        """Extract skills from text"""
        if pd.isna(text) or text == '' or text == 'nan':
            return ''
        skills = str(text).split('\\n')
        skills = [s.strip() for s in skills if s.strip()]
        return ' '.join(skills)
    
    def extract_features(self, df, fit=True):
        """Extract features from dataset"""
        # Preprocess text fields
        df['job_clean'] = df['Job Description'].apply(self.preprocess_text)
        df['resume_clean'] = df['Resume'].fillna('').apply(self.preprocess_text)
        
        # Combine Resume and Work Experience for better context
        df['cv_combined'] = df['resume_clean'] + ' ' + df['Work Experience'].fillna('').apply(self.preprocess_text)
        
        df['skills_clean'] = df['Skills'].fillna('').apply(self.extract_skills)
        
        # TF-IDF vectorization
        if fit:
            job_features = self.job_vectorizer.fit_transform(df['job_clean']).toarray()
            resume_features = self.resume_vectorizer.fit_transform(df['cv_combined']).toarray()
            skills_features = self.skills_vectorizer.fit_transform(df['skills_clean']).toarray()
        else:
            job_features = self.job_vectorizer.transform(df['job_clean']).toarray()
            resume_features = self.resume_vectorizer.transform(df['cv_combined']).toarray()
            skills_features = self.skills_vectorizer.transform(df['skills_clean']).toarray()
        
        # Years of experience
        years_exp = df['Total Years Experience'].values.reshape(-1, 1)
        
        # Combine all features
        features = np.hstack([
            job_features,
            resume_features,
            skills_features,
            years_exp
        ])
        
        return features
    
    def build_model(self, input_dim):
        """Build neural network architecture"""
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='linear')  # Regression output (0-10 scale)
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, csv_path, epochs=100, batch_size=32):
        """Train the model on historical data"""
        print("Loading data...")
        df = pd.read_csv(csv_path, skiprows=1)
        
        print(f"Total samples: {len(df)}")
        print(f"Stage value distribution:\n{df['Stage value'].value_counts().sort_index()}")
        
        # Handle class imbalance by adjusting sampling
        print("\nBalancing dataset...")
        
        # Separate by stage values
        df_2 = df[df['Stage value'] == 2]   # Review - 473
        df_4 = df[df['Stage value'] == 4]   # Screen - 45
        df_6 = df[df['Stage value'] == 6]   # Interview - 18
        df_8 = df[df['Stage value'] == 8]   # Declined - 832
        df_10 = df[df['Stage value'] == 10] # Hired - 157
        
        # Balance the dataset
        # Oversample minority classes, undersample majority
        df_2_sampled = df_2.sample(n=min(300, len(df_2)), replace=True, random_state=42)
        df_4_sampled = df_4.sample(n=min(150, len(df_4)), replace=True, random_state=42)
        df_6_sampled = df_6.sample(n=min(150, len(df_6)), replace=True, random_state=42)
        df_8_sampled = df_8.sample(n=400, replace=False, random_state=42)  # Undersample
        df_10_sampled = df_10.sample(n=min(300, len(df_10)), replace=True, random_state=42)
        
        # Combine
        df_balanced = pd.concat([
            df_2_sampled, 
            df_4_sampled, 
            df_6_sampled, 
            df_8_sampled, 
            df_10_sampled
        ])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Balanced dataset size: {len(df_balanced)}")
        print(f"New distribution:\n{df_balanced['Stage value'].value_counts().sort_index()}")
        
        # Extract features
        print("\nExtracting features...")
        X = self.extract_features(df_balanced, fit=True)
        y = df_balanced['Stage value'].values
        
        # Normalize features
        print("Normalizing features...")
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Build model
        print("\nBuilding neural network...")
        self.model = self.build_model(X_train.shape[1])
        print(self.model.summary())
        
        # Compute class weights
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights_array = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = {int(cls): weight for cls, weight in zip(classes, class_weights_array)}
        print(f"\nClass weights: {class_weights}")
        
        # Train model
        print("\nTraining model...")
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating model...")
        test_loss, test_mae = self.model.evaluate(X_test, y_test)
        print(f"Test Loss (MSE): {test_loss:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        
        # Test predictions on different score ranges
        print("\nSample predictions:")
        for stage in sorted(classes):
            mask = y_test == stage
            if mask.any():
                preds = self.model.predict(X_test[mask][:5], verbose=0)
                print(f"  Stage {int(stage)}: predictions = {preds.flatten()}")
        
        return history
    
    def save_model(self, model_dir='models'):
        """Save trained model and preprocessing objects"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        # Save neural network
        self.model.save(f'{model_dir}/model.keras')
        
        # Save preprocessing objects
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(f'{model_dir}/job_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.job_vectorizer, f)
        
        with open(f'{model_dir}/resume_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.resume_vectorizer, f)
        
        with open(f'{model_dir}/skills_vectorizer.pkl', 'wb') as f:
            pickle.dump(self.skills_vectorizer, f)
        
        print(f"\nModel saved to {model_dir}/")
    
    def load_model(self, model_dir='models'):
        """Load trained model and preprocessing objects"""
        self.model = keras.models.load_model(f'{model_dir}/model.keras')
        
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f'{model_dir}/job_vectorizer.pkl', 'rb') as f:
            self.job_vectorizer = pickle.load(f)
        
        with open(f'{model_dir}/resume_vectorizer.pkl', 'rb') as f:
            self.resume_vectorizer = pickle.load(f)
        
        with open(f'{model_dir}/skills_vectorizer.pkl', 'rb') as f:
            self.skills_vectorizer = pickle.load(f)
    
    def predict(self, job_description, cv_text, years_experience=0, skills='', work_experience=''):
        """Predict match score for a single candidate"""
        # Combine cv_text with work_experience like in training
        cv_combined = cv_text + ' ' + work_experience
        
        # Create dataframe for single prediction
        df = pd.DataFrame({
            'Job Description': [job_description],
            'Resume': [cv_text],
            'Work Experience': [work_experience],
            'Total Years Experience': [years_experience],
            'Skills': [skills]
        })
        
        # Extract features
        X = self.extract_features(df, fit=False)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        score = self.model.predict(X_scaled, verbose=0)[0][0]
        
        # Clip to 0-10 range (now includes 0!)
        score = np.clip(score, 0, 10)
        
        return float(score)


if __name__ == "__main__":
    print("=" * 60)
    print("JOB-CV MATCHING NEURAL NETWORK TRAINING - VERSION 2")
    print("=" * 60)
    
    # Initialize matcher
    matcher = JobCVMatcherV2()
    
    # Train model
    history = matcher.train('new_data.csv', epochs=100, batch_size=32)
    
    # Save model
    matcher.save_model('models')
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print("\nModel and preprocessing objects saved to ./models/")
    print("\nYou can now use this model in the Flask API.")
