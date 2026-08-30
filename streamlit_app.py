"""
streamlit_app.py
────────────────
Root entry point for Streamlit Community Cloud deployment.
Forwards execution to app.main.
"""

from app.main import main

if __name__ == "__main__":
    main()
