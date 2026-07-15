# mlatelier/__main__.py
import os
import sys
import subprocess


def main():
    """The entry point for the MLatelier CLI."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(dir_path, "app.py")

    print("Starting MLatelier Zero-Code Dashboard on localhost...")
    sys.exit(subprocess.run(["streamlit", "run", app_path]).returncode)


if __name__ == "__main__":
    main()
