# hash_password.py
from streamlit_authenticator.utilities.hasher import Hasher
import streamlit

passwords = ['123']
hashed_passwords = Hasher.hash_list(passwords)
print(hashed_passwords)