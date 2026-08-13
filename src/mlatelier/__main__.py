# mlatelier/__main__.py
import os
import sys
import subprocess


def main():
    """The entry point for the MLatelier CLI."""
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_path = os.path.join(dir_path, "app.py")

    print("Starting MLatelier Zero-Code Dashboard on localhost...")
    # Invoke Streamlit through the interpreter that is running MLatelier rather
    # than a bare "streamlit" name. The bare name resolves only when the
    # environment's Scripts/bin directory happens to be on PATH, so it fails
    # with FileNotFoundError for anyone who installs into a virtualenv and
    # launches by path without activating it first.
    sys.exit(
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_path]
        ).returncode
    )


if __name__ == "__main__":
    main()
