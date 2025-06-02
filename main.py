import subprocess
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_streamlit_app():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "ui", "app.py")

    if not os.path.exists(app_path):
        logger.error(f"Streamlit app not found at {app_path}")
        sys.exit(1)

    logger.info(f"Attempting to launch Streamlit app: {app_path}")
    
    try:
        command = [sys.executable, "-m", "streamlit", "run", app_path]
        process = subprocess.Popen(command)
        process.wait()
    except FileNotFoundError:
        logger.error("`streamlit` command not found. Make sure Streamlit is installed and in your PATH.")
        logger.info("You can typically install it with: pip install streamlit")
    except Exception as e:
        logger.error(f"An error occurred while trying to run the Streamlit app: {e}")

if __name__ == "__main__":
    run_streamlit_app()
