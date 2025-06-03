# ============================
# TRAINING MODEL BINARY SUICIDAL DETECTION
# ============================

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load dataset
df = pd.read_csv("C:/Users/peter/Desktop/Tugas/Tugas kuliah/SisCer/Suicide_Detection.csv")

# Ganti nama kolom agar konsisten
df = df.rename(columns={"text": "text", "class": "label"})

# Ambil hanya label suicide  = 1
df = df[df['label'].isin(['suicide'])]

# Tambahkan label biner: 1 untuk suicidal
df['label'] = 1  # Karena semua yang disimpan dianggap suicidal

# Tambahkan contoh non-suicidal untuk kelas 0
df_full = pd.read_csv("Suicide_Detection.csv")
df_full = df_full[df_full['class'].isin(['non-suicide'])]
df_full = df_full.rename(columns={"text": "text", "class": "label"})
df_full['label'] = 0  # Non-suicidal

# Gabungkan suicidal (1) dan non-suicidal (0)
df_binary = pd.concat([df, df_full], ignore_index=True)

# Pisah data latih dan uji
X_train, X_test, y_train, y_test = train_test_split(
    df_binary['text'], df_binary['label'], test_size=0.3, random_state=42, stratify=df_binary['label']
)

# TF-IDF vektorisasi
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model logistic regression
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train_tfidf, y_train)

# Evaluasi
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

# Simpan model
pickle.dump(model, open("suicidal_binary_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

# ============================
# STREAMLIT DASHBOARD
# ============================

import streamlit as st
import pandas as pd
import pickle

# Load model dan vectorizer
model = pickle.load(open("suicidal_binary_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

st.title("Deteksi Risiko Suicidal (Binary Classifier)")
st.write("Unggah file CSV atau TXT berisi pesan teks. Sistem akan mendeteksi apakah pesan mengandung potensi suicidal atau tidak.")

uploaded_file = st.file_uploader("Unggah file (.csv atau .txt)", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if 'text' not in df.columns:
            st.error("File CSV harus memiliki kolom bernama 'text'")
        else:
            texts = df['text'].astype(str)
    elif uploaded_file.name.endswith(".txt"):
        lines = uploaded_file.read().decode("utf-8").splitlines()
        texts = pd.Series(lines)
    else:
        st.error("Format file tidak dikenali.")

    # Preprocessing dan prediksi
    tfidf_text = vectorizer.transform(texts)
    predictions = model.predict(tfidf_text)
    probs = model.predict_proba(tfidf_text)

    result_df = pd.DataFrame({
        "Pesan": texts,
        "Risiko Suicidal": ["YA" if p == 1 else "TIDAK" for p in predictions],
        "Probabilitas Suicidal (%)": [f"{p[1]*100:.2f}%" for p in probs]
    })

    st.subheader(" Hasil Analisis")
    st.dataframe(result_df)

    st.subheader(" Ringkasan")
    count_series = result_df['Risiko Suicidal'].value_counts()
    for label, count in count_series.items():
        st.write(f"{label}: {count} pesan ({(count/len(result_df))*100:.2f}%)")

    st.bar_chart(count_series)

    # Tampilkan TF-IDF (opsional, bisa berat kalau terlalu banyak teks)
    st.subheader("Nilai TF-IDF per Kata")
    tfidf_array = tfidf_text.toarray()
    tfidf_df = pd.DataFrame(tfidf_array, columns=vectorizer.get_feature_names_out())
    tfidf_df_display = pd.concat([texts.rename("Pesan").reset_index(drop=True), tfidf_df], axis=1)
    st.dataframe(tfidf_df_display)
