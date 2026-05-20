"""Local TCP forwarder for tunneling Postgres connections through an HTTP CONNECT proxy.

Postgres is a binary TCP protocol; DB drivers (asyncpg, psycopg) do not honor
HTTP_PROXY / HTTPS_PROXY env vars natively. Instead of monkey-patching the
drivers, this module accepts inbound connections on 127.0.0.1:<random_port>
and forwards each one through `proxy_url` via HTTP CONNECT (handled by
`python-socks`). The SQLAlchemy URL is then rewritten to point at the local
listener — both async (asyncpg) and sync (psycopg, Alembic) drivers go through
the same tunnel transparently.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from urllib.parse import urlsplit, urlunsplit

logger = logging.getLogger(__name__)

_PIPE_BUFFER = 65536


def _redact_proxy_url(url: str) -> str:
    parts = urlsplit(url)
    if not (parts.username or parts.password):
        return url
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    user = parts.username or ""
    netloc = f"{user}:***@{host}" if user else f"***@{host}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class DBProxyTunnel:
    """Forwards each accepted connection on 127.0.0.1:<port> through an HTTP CONNECT proxy.

    Lifecycle:
        - `start_async()` — bind a listener on the current event loop (agent CLI).
        - `start_sync()` — bind a listener in a background thread with its own
          event loop (Streamlit dashboard, which is sync-only).
        - `stop()` — close the listener and any in-flight connections. The sync
          variant also stops the background loop.

    The tunnel is intentionally driver-agnostic: it doesn't inspect Postgres
    frames, doesn't terminate TLS, and doesn't fork a per-connection task tree.
    Each accepted connection gets one CONNECT handshake and two pipe coroutines.
    """

    def __init__(self, proxy_url: str, target_host: str, target_port: int) -> None:
        self.proxy_url = proxy_url
        self.target_host = target_host
        self.target_port = target_port

        self.local_host: str | None = None
        self.local_port: int | None = None

        self._server: asyncio.AbstractServer | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        from python_socks import ProxyError
        from python_socks.async_.asyncio import Proxy

        try:
            proxy = Proxy.from_url(self.proxy_url)
            sock = await proxy.connect(
                dest_host=self.target_host, dest_port=self.target_port
            )
        except (ProxyError, OSError):
            logger.exception(
                "db proxy tunnel: CONNECT to %s:%d failed",
                self.target_host, self.target_port,
            )
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            return

        target_reader, target_writer = await asyncio.open_connection(sock=sock)

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await src.read(_PIPE_BUFFER)
                    if not chunk:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except (asyncio.CancelledError, ConnectionError, OSError):
                pass
            finally:
                if not dst.is_closing():
                    dst.close()

        try:
            await asyncio.gather(
                pipe(reader, target_writer),
                pipe(target_reader, writer),
            )
        finally:
            for w in (writer, target_writer):
                if not w.is_closing():
                    w.close()

    async def start_async(self) -> tuple[str, int]:
        """Bind listener on the current event loop. Returns (host, port)."""
        self._loop = asyncio.get_running_loop()
        self._server = await asyncio.start_server(
            self._handle_client, "127.0.0.1", 0
        )
        sockname = self._server.sockets[0].getsockname()
        self.local_host, self.local_port = sockname[0], sockname[1]
        logger.info(
            "db proxy tunnel ready: 127.0.0.1:%d -> %s:%d via %s",
            self.local_port, self.target_host, self.target_port,
            _redact_proxy_url(self.proxy_url),
        )
        return self.local_host, self.local_port

    def start_sync(self) -> tuple[str, int]:
        """Bind listener in a background thread with its own event loop.

        Used by the Streamlit dashboard. The thread is a daemon — the listener
        is torn down when the process exits, and the cached engine handle keeps
        the tunnel alive across reruns.
        """
        ready = threading.Event()
        error: dict[str, BaseException] = {}

        def _run() -> None:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                loop.run_until_complete(self.start_async())
                ready.set()
                loop.run_forever()
            except BaseException as e:
                error["e"] = e
                ready.set()

        self._thread = threading.Thread(
            target=_run, name="db-proxy-tunnel", daemon=True
        )
        self._thread.start()
        ready.wait()
        if "e" in error:
            raise error["e"]
        assert self.local_host is not None and self.local_port is not None
        return self.local_host, self.local_port

    async def stop_async(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except OSError:
                pass
            self._server = None

    def stop(self) -> None:
        """Stop the listener. Safe to call from either sync or async contexts."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if self._thread is not None:
            future = asyncio.run_coroutine_threadsafe(self.stop_async(), loop)
            try:
                future.result(timeout=5)
            except (TimeoutError, RuntimeError):
                pass
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=5)
            self._thread = None
        else:
            # Same-loop stop is the caller's responsibility (await stop_async).
            pass
