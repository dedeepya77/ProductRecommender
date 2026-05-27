import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from auth import check_login

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Startup Ecommerce MVP", layout="wide")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("products.csv").dropna()
df.columns = df.columns.str.strip()

df["combined"] = df["name"] + " " + df["category"] + " " + df["features"]
df["name_lower"] = df["name"].str.lower()

vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(df["combined"])
similarity = cosine_similarity(matrix)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = ""

if "cart" not in st.session_state:
    st.session_state.cart = []

if "orders" not in st.session_state:
    st.session_state.orders = []

# ---------------- CART SAFE ADD ----------------
def add_to_cart(item):
    cart = st.session_state.get("cart", [])
    st.session_state.cart = cart + [item]

# ---------------- UTIL ----------------
def get_suggestions(query):
    if not query:
        return []
    return df[df["name_lower"].str.contains(query.lower())]["name"].head(5).tolist()

def recommend(product, category, price_limit):

    product = product.lower().strip()

    match = df[df["name_lower"].str.contains(product)]

    if match.empty:
        return []

    idx = match.index[0]

    scores = sorted(list(enumerate(similarity[idx])), key=lambda x: x[1], reverse=True)

    results = []

    for i, _ in scores[1:50]:

        item = df.iloc[i]

        if category != "All" and item["category"] != category:
            continue

        if item["price"] > price_limit:
            continue

        results.append(item)

        if len(results) == 12:
            break

    return results

# ---------------- LOGIN ----------------
def login_page():

    st.title("🛍️ Startup Ecommerce MVP")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")

        if st.button("Login"):
            if check_login(u, p):
                st.session_state.logged_in = True
                st.session_state.user = u
                st.rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        nu = st.text_input("New Username")
        np = st.text_input("New Password", type="password")

        if st.button("Create Account"):
            users = pd.read_csv("users.csv")

            if nu in users["username"].values:
                st.warning("User already exists")
            else:
                users.loc[len(users)] = [nu, np]
                users.to_csv("users.csv", index=False)
                st.success("Account created")

# ---------------- HOME ----------------
def home_page():

    st.markdown(f"## 👋 Welcome {st.session_state.user}")

    category = st.sidebar.selectbox("Category", ["All"] + list(df["category"].unique()))
    price_limit = st.sidebar.slider("Max Price", 0, int(df["price"].max()), int(df["price"].max()))

    search = st.text_input("🔎 Search products")

    if search:
        st.caption(f"Showing results for: {search}")

    suggestions = get_suggestions(search)

    if suggestions:
        st.write("### Suggestions")

        for s in suggestions:
            if st.button(s, key="sugg_" + s):
                search = s

    if st.button("Get Recommendations"):

        results = recommend(search, category, price_limit)

        if not results:
            st.warning("No products found")
            return

        st.subheader("🔥 Recommended Products")

        cols = st.columns(4)

        for i, r in enumerate(results):

            with cols[i % 4]:

                st.image(r["image"], use_container_width=True)

                st.markdown(f"### {r['name']}")
                st.markdown(f"⭐ {r['rating']} / 5")
                st.markdown(f"💰 ₹{r['price']}")
                st.caption(r["features"])

                if st.button("🛒 Add to Cart", key="cart_" + str(r["name"]) + str(i)):
                    add_to_cart(r)
                    st.toast("Added to cart 🛒")
                    st.rerun()

# ---------------- CART ----------------
def cart_page():

    st.subheader("🛒 Your Cart")

    cart = st.session_state.get("cart", [])

    if len(cart) == 0:
        st.info("Cart is empty")
        return

    total = 0

    for i, item in enumerate(cart):

        col1, col2 = st.columns([3, 1])

        with col1:
            st.write(f"**{item['name']}** - ₹{item['price']}")

        with col2:
            if st.button("Remove", key="rm_" + str(i)):
                st.session_state.cart.pop(i)
                st.rerun()

        total += item["price"]

    st.markdown(f"### Total: ₹{total}")

    if st.button("Proceed to Checkout"):

        st.session_state.orders.append({
            "items": cart.copy(),
            "total": total
        })

        st.session_state.cart = []
        st.success("Order placed 🎉")
        st.rerun()

# ---------------- ORDERS ----------------
def orders_page():

    st.subheader("📦 Orders")

    if not st.session_state.orders:
        st.info("No orders yet")
        return

    for i, order in enumerate(st.session_state.orders):

        st.markdown(f"### Order {i+1}")
        st.write(f"Total: ₹{order['total']}")

        for item in order["items"]:
            st.write(f"- {item['name']}")

        st.markdown("---")

# ---------------- ROUTER ----------------
def app():

    menu = st.sidebar.radio("Navigation", ["Home", "Cart", "Orders"])

    if menu == "Home":
        home_page()

    elif menu == "Cart":
        cart_page()

    elif menu == "Orders":
        orders_page()

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.cart = []
        st.rerun()

# ---------------- MAIN ----------------
if not st.session_state.logged_in:
    login_page()
else:
    app()