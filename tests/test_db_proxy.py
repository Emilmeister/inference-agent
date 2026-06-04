"""Tests for the DB-connection HTTP CONNECT tunnel (`inference_api.db_proxy.DBProxyTunnel`).

Stands up two in-process servers — a tiny HTTP CONNECT proxy and an echo
target — and verifies that bytes routed through the tunnel land on the target
and come back. Keeps the dependency surface narrow (asyncio + python-socks
client only), no real Postgres involved.
"""

from __future__ import annotations

import asyncio

import pytest

from inference_api.db_proxy import DBProxyTunnel, _redact_proxy_url


async def _start_echo_server() -> tuple[asyncio.AbstractServer, int]:
    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            if not writer.is_closing():
                writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


async def _start_http_connect_proxy() -> tuple[asyncio.AbstractServer, int]:
    """Minimal HTTP CONNECT proxy: parses `CONNECT host:port HTTP/1.1`, opens a
    TCP socket to that target, and pipes bytes in both directions."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            if not request_line.startswith(b"CONNECT "):
                writer.close()
                return

            # Drain remaining headers until empty line.
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break

            try:
                _, target, _ = request_line.decode().split(" ", 2)
                host, port_str = target.split(":")
                port = int(port_str)
            except ValueError:
                writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            try:
                target_reader, target_writer = await asyncio.open_connection(host, port)
            except OSError:
                writer.write(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                await writer.drain()
                writer.close()
                return

            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()

            async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
                try:
                    while True:
                        data = await src.read(4096)
                        if not data:
                            break
                        dst.write(data)
                        await dst.drain()
                except (ConnectionError, OSError):
                    pass
                finally:
                    if not dst.is_closing():
                        dst.close()

            await asyncio.gather(
                pipe(reader, target_writer),
                pipe(target_reader, writer),
            )
        except (ConnectionError, OSError):
            pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port


@pytest.mark.asyncio
async def test_tunnel_pipes_bytes_through_http_connect_proxy() -> None:
    echo_server, echo_port = await _start_echo_server()
    proxy_server, proxy_port = await _start_http_connect_proxy()
    tunnel = DBProxyTunnel(
        proxy_url=f"http://127.0.0.1:{proxy_port}",
        target_host="127.0.0.1",
        target_port=echo_port,
    )

    try:
        local_host, local_port = await tunnel.start_async()
        reader, writer = await asyncio.open_connection(local_host, local_port)
        try:
            payload = b"hello postgres frame\x00\x01\x02"
            writer.write(payload)
            await writer.drain()

            received = b""
            while len(received) < len(payload):
                chunk = await asyncio.wait_for(reader.read(4096), timeout=2.0)
                if not chunk:
                    break
                received += chunk
            assert received == payload
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
    finally:
        await tunnel.stop_async()
        proxy_server.close()
        echo_server.close()
        await proxy_server.wait_closed()
        await echo_server.wait_closed()


def test_redact_proxy_url_masks_password() -> None:
    assert (
        _redact_proxy_url("http://alice:secret@proxy.example:3128")
        == "http://alice:***@proxy.example:3128"
    )
    assert _redact_proxy_url("http://proxy.example:3128") == "http://proxy.example:3128"
