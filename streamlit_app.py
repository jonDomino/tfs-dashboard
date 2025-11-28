"""Streamlit app entry point - thin wrapper"""
from app.main import render, setup_refresh_timer

# Main execution
render()
setup_refresh_timer(30)

