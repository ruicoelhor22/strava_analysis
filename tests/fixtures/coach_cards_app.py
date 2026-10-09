import streamlit as st
from dashboard.coach_pages import _prescription_card

for prescription in st.session_state.get('prescriptions', []):
    _prescription_card(prescription)
