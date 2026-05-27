import pandas as pd

def check_login(username, password):
    users = pd.read_csv("users.csv")

    user = users[
        (users["username"] == username) &
        (users["password"] == password)
    ]

    return not user.empty