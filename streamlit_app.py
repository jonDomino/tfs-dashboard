"""Streamlit app entry point - thin wrapper"""
from app.main import render, setup_refresh_timer
from app.config import config

# Main execution
render()
setup_refresh_timer(config.REFRESH_INTERVAL)

