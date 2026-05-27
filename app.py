import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from auth import check_login

st.set_page_config(page_title="Smart Product Recommender", layout="wide")

# ---------------- LOGIN SYSTEM ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.title("🔐 Login to Product Recommender")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Successful 🎉")
            st.rerun()
        else:
            st.error("Invalid Credentials ❌")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("products.csv")

# Clean column names (important safety step)
df.columns = df.columns.str.strip()

# Create ML feature space
df["combined"] = df["category"] + " " + df["features"]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df["combined"])
similarity = cosine_similarity(matrix)

# Add lowercase column for flexible search
df["name_lower"] = df["name"].str.lower()

# ---------------- RECOMMENDATION ENGINE ----------------
def recommend(product):
    product = product.lower().strip()

    match = df[df["name_lower"].str.contains(product)]

    if match.empty:
        return None

    idx = match.index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []
    for i in scores[1:6]:
        recommendations.append(df.iloc[i[0]])

    return recommendations

# ---------------- MAIN APP ----------------
def recommender_app():
    st.title("🛍️ Smart Product Recommendation System")
    st.sidebar.success(f"Welcome {st.session_state.username} 👋")

    product_name = st.text_input("Enter Product Name (e.g., iPhone 13, macbook, samsung)")

    if st.button("Recommend"):
        results = recommend(product_name)

        if results:
            st.subheader("Top Recommendations 🔥")

            for r in results:
                st.markdown(f"""
                ---
                ### 📦 {r['name']}
                - 📱 Category: {r['category']}
                - 📝 Features: {r['features']}
                """)
        else:
            st.error("Product not found. Try partial name like 'iphone' or 'macbook'")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- ROUTING ----------------
if not st.session_state.logged_in:
    login_page()
else:
    recommender_app()