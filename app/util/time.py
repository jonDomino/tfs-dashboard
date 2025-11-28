"""Time and refresh utilities"""
import streamlit as st
from datetime import datetime
import time


def setup_refresh_timer(interval_seconds: int = 30):
    """Setup flicker-free refresh timer.
    
    This should be called at the end of the main render function.
    Adds a manual refresh button and optional auto-refresh using JavaScript
    to avoid browser crashes from st.rerun() loops.
    
    Args:
        interval_seconds: Refresh interval in seconds
    """
    # Add manual refresh button in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Dashboard", use_container_width=True):
        st.session_state.last_refresh_time = datetime.now()
        st.rerun()
    
    # Show last refresh time
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = datetime.now()
    
    st.sidebar.caption(f"Last refresh: {st.session_state.last_refresh_time.strftime('%H:%M:%S')}")
    
    # Optional: Add auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        # Use JavaScript-based refresh to avoid st.rerun() loops that crash the browser
        # This is safer than calling st.rerun() during render
        st.markdown(
            f"""
            <script>
                // Only set up refresh if not already set
                if (!window.refreshTimerSet) {{
                    window.refreshTimerSet = true;
                    setTimeout(function(){{
                        window.location.reload();
                    }}, {interval_seconds * 1000});
                }}
            </script>
            """,
            unsafe_allow_html=True
        )

