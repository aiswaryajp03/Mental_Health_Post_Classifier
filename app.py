

import streamlit as st
import pickle

with open('tfidf_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

st.title("🧠 Mental Health Post Classifier")
st.write("Type a sentence below. The model checks it for signs of distress.")

user_input = st.text_area("Enter text here:", height=120)

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please type something first.")

    else:

        vec = vectorizer.transform([user_input])

        if vec.nnz == 0:
            st.warning(
                "⚠️ None of these words were recognized by the model. "
                "It has no real signal to go on, so any prediction here "
                "would just reflect its default guess, not your actual text. "
                "Try rephrasing with more common words."
            )

        else:
            prediction = model.predict(vec)[0]
            confidence = model.predict_proba(vec)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Distress signals detected ({confidence*100:.1f}% confidence)")
            else:
                st.success(f"✅ No distress signals detected ({(1-confidence)*100:.1f}% confidence)")


            recognized_words = vec.nnz
            st.caption(
                f"ℹ️ {recognized_words} known word(s) out of your input "
                f"were recognized by the model's vocabulary."
            )

