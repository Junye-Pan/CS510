from __future__ import annotations

import argparse
import select
import socket
import threading
from pathlib import Path


class UnixProxyBridge:
    """Bridge a container-local TCP HTTP proxy port to a mounted Unix socket."""

    def __init__(self, *, host: str, port: int, socket_path: Path) -> None:
        self.host = host
        self.port = port
        self.socket_path = socket_path

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.port))
            listener.listen()
            while True:
                client, _ = listener.accept()
                thread = threading.Thread(target=self._handle_client, args=(client,), daemon=True)
                thread.start()

    def _handle_client(self, client: socket.socket) -> None:
        with client:
            try:
                upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                upstream.connect(str(self.socket_path))
            except OSError:
                return
            with upstream:
                _copy_bidirectional(client, upstream)


def _copy_bidirectional(left: socket.socket, right: socket.socket) -> None:
    sockets = [left, right]
    while True:
        try:
            readable, _, _ = select.select(sockets, [], [], 30)
        except OSError:
            return
        if not readable:
            return
        for source in readable:
            try:
                data = source.recv(65536)
            except OSError:
                return
            if not data:
                return
            target = right if source is left else left
            try:
                target.sendall(data)
            except OSError:
                return


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bridge a local TCP HTTP proxy port to a Unix socket proxy.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    UnixProxyBridge(host=args.host, port=args.port, socket_path=args.socket).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
