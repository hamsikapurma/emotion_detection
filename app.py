import streamlit as st
import joblib

# Load model and vectorizer using joblib
model = joblib.load(r"C:\Users\HAMSIKA\Desktop\emotion_detection\detection.pkl")
vectorizer = joblib.load(r"C:\Users\HAMSIKA\Desktop\emotion_detection\vectorization.pkl")

# Label map
label_map = {
    'joy': 0,
    'fear': 1,
    'anger': 2,
    'sadness': 3,
    'disgust': 4,
    'shame': 5,
    'surprise': 6,
    'neutral': 7
}
# Reverse for output
reverse_label_map = {v: k for k, v in label_map.items()}

# Streamlit UI
st.title("😊 Emotion Detection from Text")

st.write("Enter a sentence and detect the emotion behind it.")

# Show label mapping
with st.expander("🗺️ Emotion Labels Mapping"):
    st.json(label_map)

# Input area
user_input = st.text_area("Enter your sentence here:")

if st.button("Predict Emotion"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        cleaned_input = user_input.lower()
        vect_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vect_input)[0]
        emotion_label = reverse_label_map[prediction]
        st.success(f"🎯 Predicted Emotion: **{emotion_label.upper()}**")
