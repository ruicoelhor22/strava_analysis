"""Compatibility entrypoint for the new Streamlit dashboard.

Prefer:
    streamlit run dashboard/app.py
"""

from dashboard.app import main


if __name__ == "__main__":
    main()
