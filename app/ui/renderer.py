"""UI rendering components"""
import streamlit as st
from typing import Optional
import matplotlib.pyplot as plt


def render_chart(fig: plt.Figure):
    """Render a matplotlib figure.
    
    Args:
        fig: Matplotlib figure
    """
    st.pyplot(fig)


def render_error(message: str):
    """Render an error message.
    
    Args:
        message: Error message
    """
    st.error(message)


def render_warning(message: str):
    """Render a warning message.
    
    Args:
        message: Warning message
    """
    st.warning(message)


def render_info(message: str):
    """Render an info message.
    
    Args:
        message: Info message
    """
    st.info(message)


def render_badge(text: str, color: str = "blue"):
    """Render a status badge.
    
    Args:
        text: Badge text
        color: Badge color
    """
    st.markdown(f'<span style="background-color: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.8em;">{text}</span>', 
                unsafe_allow_html=True)

