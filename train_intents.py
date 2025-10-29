import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Switched to TF-IDF
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='train.log'
)
logger = logging.getLogger('train_intents')

# Load intents data
def load_intents(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data["intents"]
    except Exception as e:
        logger.error(f"Error loading intents: {e}")
        raise

# Prepare training data
def prepare_training_data(intents):
    X, y = [], []
    for intent in intents:
        for pattern in intent["patterns"]:
            X.append(pattern.lower())
            y.append(intent["tag"])
    return X, y

# Generate additional training data with variations
def augment_data(X, y):
    X_augmented, y_augmented = X.copy(), y.copy()

    for i, text in enumerate(X):
        text = text.lower()
        tag = y[i]

        # Preserve meaning of 'on' and 'off'
        if "off" in text:
            X_augmented.append(text.replace("off", "switch off"))
            X_augmented.append(f"please {text}")
            X_augmented.append(text + " now")
            y_augmented.extend([tag, tag, tag])

        elif "on" in text:
            X_augmented.append(text.replace("on", "switch on"))
            X_augmented.append(f"please {text}")
            X_augmented.append(text + " now")
            y_augmented.extend([tag, tag, tag])

        # Add polite forms
        fillers = ["can you", "would you", "could you", "i want to"]
        for filler in fillers:
            X_augmented.append(f"{filler} {text}")
            y_augmented.append(tag)

    return X_augmented, y_augmented



# Train and evaluate the model
def train_intent_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Use TF-IDF instead of CountVectorizer
    vectorizer = TfidfVectorizer(
    stop_words=None,             # keep “on” and “off”
    ngram_range=(1, 3),
    max_features=1500,
    sublinear_tf=True
    )

    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    model = MultinomialNB(alpha=0.1)  # Lower alpha for less smoothing, better fit
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_vectorized, y_train, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean CV score: {cv_scores.mean():.4f}")
    
    # Train final model
    model.fit(X_train_vectorized, y_train)
    
    # Evaluate on test set
    y_pred = model.predict(X_test_vectorized)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    
    # Confusion matrix
    classes = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return model, vectorizer, (X_test, y_test, y_pred)

if __name__ == "__main__":
    logger.info("Starting intent training...")
    
    # Check if intents.json exists, create a default one if not
    if not os.path.exists("intents.json"):
        default_intents = {
            "intents": [
                {"tag": "turn_on_light", "patterns": ["turn on the light", "light on", "turn on lights"]},
                {"tag": "turn_off_light", "patterns": ["turn off the light", "light off", "turn off lights"]},
                {"tag": "turn_on_fan", "patterns": ["turn on the fan", "fan on", "start fan"]},
                {"tag": "turn_off_fan", "patterns": ["turn off the fan", "fan off", "stop fan"]},
                {"tag": "play_music", "patterns": ["play music", "start music", "play the music"]},
                {"tag": "stop_music", "patterns": ["stop music", "music off", "stop the music"]},
                {"tag": "lock_front_door", "patterns": ["lock front door", "lock the door"]},
                {"tag": "unlock_front_door", "patterns": ["unlock front door", "unlock the door"]},
                {"tag": "set_ac_temperature", "patterns": ["set temperature to 22", "set ac to 25"]},
                {"tag": "greeting", "patterns": ["hello", "hi", "hey"]},
                {"tag": "farewell", "patterns": ["goodbye", "bye", "see you"]}
            ]
        }
        with open("intents.json", "w") as f:
            json.dump(default_intents, f, indent=4)
        logger.info("Created default intents.json")
    
    print("Loading intents data...")
    intents = load_intents("intents.json")
    X, y = prepare_training_data(intents)
    
    print(f"Original dataset: {len(X)} examples")
    X_augmented, y_augmented = augment_data(X, y)
    print(f"Augmented dataset: {len(X_augmented)} examples")
    
    print("Training model with cross-validation...")
    model, vectorizer, eval_data = train_intent_classifier(X_augmented, y_augmented)
    
    print("Saving model and vectorizer...")
    with open("intent_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    # Test examples
    print("\nTesting model with examples:")
    test_examples = [
        "turn on the lights in the living room",
        "please switch off all lights",
        "can you play some jazz music",
        "i want to lock the front door",
        "set the temperature to 23 degrees",
        "hello there",
        "goodbye now",
        "turn on the music",
        "play the music"
    ]
    
    X_examples = vectorizer.transform(test_examples)
    predictions = model.predict(X_examples)
    probs = model.predict_proba(X_examples)
    
    for i, example in enumerate(test_examples):
        print(f"Input: '{example}'")
        print(f"Predicted intent: '{predictions[i]}' (Confidence: {max(probs[i]):.2f})\n")
    
    print("Model trained and saved successfully! 🎉")
    logger.info("Training completed successfully")