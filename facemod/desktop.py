"""Desktop entry point: local server + native window (Edge WebView2 via pywebview).

Falls back to the default browser if a native window can't be created.
"""
import os
import socket
import sys
import threading
import webbrowser

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run():
    from http.server import ThreadingHTTPServer
    from . import server

    headless = os.environ.get("FACEMOD_HEADLESS")  # testing: serve on this port, no window
    port = int(headless) if headless else _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
    if headless:
        srv.serve_forever()
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:%d/" % port

    try:
        import webview
    except ImportError:
        webview = None

    if webview is None:
        webbrowser.open(url)
        print("MDickie FaceMod running at", url, "- close this window to quit")
        threading.Event().wait()
        return

    win = webview.create_window("MDickie FaceMod", url, width=1480, height=940, min_size=(1000, 700),
                                background_color="#141414")

    def pick_folder():
        r = win.create_file_dialog(webview.FileDialog.FOLDER)
        return list(r) if r else []

    def pick_files():
        r = win.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True,
                                   file_types=("Images (*.jpg;*.jpeg;*.png;*.webp;*.bmp)",))
        return list(r) if r else []

    server.NATIVE["pick_folder"] = pick_folder
    server.NATIVE["pick_files"] = pick_files
    try:
        webview.start()
    except Exception:
        webbrowser.open(url)
        threading.Event().wait()
    srv.shutdown()


if __name__ == "__main__":
    run()
