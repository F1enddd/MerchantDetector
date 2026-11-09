
import time
import threading
import webbrowser

import uvicorn

from main import app


HOST = "127.0.0.1"
PORT = 8765
URL = f"http://{HOST}:{PORT}"


def open_browser_when_ready(server):
    while not server.started and not server.should_exit:
        time.sleep(0.2)

    if server.started:
        webbrowser.open(URL)


if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )

    server = uvicorn.Server(config)

    browser_thread = threading.Thread(
        target=open_browser_when_ready,
        args=(server,),
        daemon=True
    )
    browser_thread.start()

    print("MerchantDetector запускается...")
    print(f"Адрес: {URL}")
    print("Для завершения нажмите Ctrl+C.")

    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
        print("MerchantDetector остановлен.")
