import numpy as np
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Input, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from tensorflow.keras.regularizers import l2

# 1. Load and preprocess your data
def prepare_data(df):
    conversations = df['chat_completion'].tolist()
    # Get all binary columns
    binary_cols = [col for col in df.columns if col.endswith('_binary')]
    # Create multi-label array
    labels = df[binary_cols].values
    return conversations, labels

# 2. Generate embeddings
def generate_embeddings(conversations):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for conversation in conversations:
        # Split conversation into utterances
        utterances = conversation.split('@@@')
        utterance_embeddings = []
        for utterance in utterances:
            if utterance.strip():
                embedding = model.encode(utterance.strip())
                utterance_embeddings.append(embedding)
        embeddings.append(utterance_embeddings)
    return embeddings

# 3. Create the model
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.1, 
                          kernel_regularizer=l2(0.01))),
        Bidirectional(LSTM(64, recurrent_dropout=0.1, kernel_regularizer=l2(0.001))),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Use sigmoid for multi-label
    ])
    return model

# 4. Main training loop
def train_model(df):
    conversations, labels = prepare_data(df)
    embeddings = generate_embeddings(conversations)
    
    # Pad sequences
    max_length = max(len(conv) for conv in embeddings)
    X = pad_sequences(embeddings, maxlen=max_length, padding='post', dtype='float32')
    y = labels

    # Cross-validation setup
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Metrics storage
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, np.argmax(y, axis=1)), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Create and compile model
        model = create_model((X_train.shape[1], X_train.shape[2]), y.shape[1])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=10,
            batch_size=16,
            validation_data=(X_val, y_val)
        )

        # Evaluate
        y_pred = (model.predict(X_val) > 0.5).astype(int)
        
        # Print results for this fold
        print(f'\nFold {fold} Results:')
        for i, col in enumerate([col for col in df.columns if col.endswith('_binary')]):
            print(f'\nMetrics for {col}:')
            print(classification_report(y_val[:, i], y_pred[:, i]))

        # Store metrics
        all_metrics.append({
            'fold': fold,
            'history': history.history,
            'val_predictions': y_pred
        })

    return all_metrics

# 5. Run the training
metrics = train_model(filtered_df)  # Your filtered DataFrame

# 6. Print average results
print("\nAverage Results Across Folds:")
avg_val_acc = np.mean([m['history']['val_accuracy'][-1] for m in metrics])
avg_val_loss = np.mean([m['history']['val_loss'][-1] for m in metrics])
print(f'Average Validation Accuracy: {avg_val_acc:.4f}')
print(f'Average Validation Loss: {avg_val_loss:.4f}')
