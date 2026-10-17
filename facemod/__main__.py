import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from .server import main

if __name__ == "__main__":
    main(open_browser="--no-browser" not in sys.argv)
