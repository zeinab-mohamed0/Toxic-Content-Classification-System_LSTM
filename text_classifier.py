import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

model_classification = pickle.load(open(r"New folder\model.pkl", "rb"))
tokenizer = pickle.load(open(r"New folder\tokenizer.pkl", "rb"))
label_encoder = pickle.load(open(r"label_encoder.pkl", "rb"))

max_len = 60


def predict_label(text):
     # 1. text → sequence
    sequence = tokenizer.texts_to_sequences([text])
    
    # 2. padding
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    
    # 3. prediction
    prediction = model_classification.predict(padded)
    label_index = prediction.argmax(axis=1)
    
    # 4. نجيب الكلاس
    label = label_encoder.inverse_transform(label_index)[0]
    
    return label
