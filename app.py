import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from auth import check_login

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Product Recommender",
    page_icon="🛍️",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    background-color: #ff9900;
    color: white;
    font-weight: bold;
    border: none;
    padding: 10px;
}

.stButton>button:hover {
    background-color: #e68a00;
}

.product-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.title-text {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
    color: #232f3e;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN PAGE ----------------
def login_page():

    st.markdown('<p class="title-text">🛍️ Smart Product Recommender</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">AI-Powered Recommendation System</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.subheader("🔐 Login")

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login Successful 🎉")
                st.rerun()
            else:
                st.error("Invalid Username or Password ❌")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("products.csv")
df.columns = df.columns.str.strip()

# ---------------- SIDEBAR FILTERS ----------------
category_filter = st.sidebar.selectbox(
    "📂 Filter by Category",
    ["All"] + list(df["category"].unique())
)

max_price = int(df["price"].max())
price_filter = st.sidebar.slider(
    "💰 Maximum Price",
    0,
    max_price,
    max_price
)

# ---------------- ML MODEL ----------------
df["combined"] = df["category"] + " " + df["features"] + " " + df["name"]

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df["combined"])
similarity = cosine_similarity(matrix)

df["name_lower"] = df["name"].str.lower()

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(product):

    product = product.lower().strip()

    match = df[df["name_lower"].str.contains(product)]

    if match.empty:
        return None

    idx = match.index[0]

    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    recommendations = []

    for i in scores[1:10]:

        item = df.iloc[i[0]]

        if category_filter != "All":
            if item["category"] != category_filter:
                continue

        if item["price"] > price_filter:
            continue

        recommendations.append(item)

    return recommendations

# ---------------- MAIN APP ----------------
def recommender_app():

    st.markdown('<p class="title-text">🛍️ Smart Product Recommender</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">Welcome {st.session_state.username} 👋</p>', unsafe_allow_html=True)

    search = st.text_input(
        "🔎 Search Product",
        placeholder="Try: iphone, samsung, macbook"
    )

    if st.button("Recommend Products"):

        results = recommend(search)

        if results:

            st.subheader("🔥 Recommended Products")

            cols = st.columns(2)

            for index, r in enumerate(results):

                with cols[index % 2]:

                    st.markdown('<div class="product-card">', unsafe_allow_html=True)

                    st.image(r["image"], use_container_width=True)

                    st.markdown(f"## {r['name']}")
                    st.markdown(f"⭐ Rating: **{r['rating']}**")
                    st.markdown(f"💰 Price: **₹{r['price']}**")
                    st.markdown(f"📦 Category: **{r['category']}**")
                    st.markdown(f"📝 {r['features']}")

                    st.button("🛒 Buy Now", key=r['name'])

                    st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error("Product not found ❌")

    st.sidebar.markdown("---")

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# ---------------- ROUTING ----------------
if not st.session_state.logged_in:
    login_page()
else:
    recommender_app()
