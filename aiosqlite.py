"""Lightweight local fallback for aiosqlite-compatible APIs.

Implements the subset of APIs used in this project, including the dual pattern:
- await db.execute(...)
- async with db.execute(...) as cursor
"""

import asyncio
import sqlite3
from typing import Any, Iterable, Optional

IntegrityError = sqlite3.IntegrityError


class Cursor:
    def __init__(self, cursor: sqlite3.Cursor):
        self._cursor = cursor

    async def fetchone(self):
        return await asyncio.to_thread(self._cursor.fetchone)

    async def fetchall(self):
        return await asyncio.to_thread(self._cursor.fetchall)

    async def close(self):
        await asyncio.to_thread(self._cursor.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    def __aiter__(self):
        return self

    async def __anext__(self):
        row = await self.fetchone()
        if row is None:
            raise StopAsyncIteration
        return row


class _ExecuteResult:
    def __init__(self, conn: sqlite3.Connection, sql: str, params: Iterable[Any]):
        self._conn = conn
        self._sql = sql
        self._params = tuple(params)
        self._cursor: Optional[Cursor] = None

    async def _ensure_cursor(self) -> Cursor:
        if self._cursor is None:
            def _execute():
                cur = self._conn.cursor()
                cur.execute(self._sql, self._params)
                return cur

            cur = await asyncio.to_thread(_execute)
            self._cursor = Cursor(cur)
        return self._cursor

    def __await__(self):
        return self._ensure_cursor().__await__()

    async def __aenter__(self):
        return await self._ensure_cursor()

    async def __aexit__(self, exc_type, exc, tb):
        if self._cursor is not None:
            await self._cursor.close()


class Connection:
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def execute(self, sql: str, params: Iterable[Any] = ()):  # awaitable + async context manager
        return _ExecuteResult(self._conn, sql, params)

    async def commit(self):
        await asyncio.to_thread(self._conn.commit)

    async def close(self):
        await asyncio.to_thread(self._conn.close)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()


class _ConnectCtx:
    def __init__(self, database: str, timeout: float = 5.0):
        self._database = database
        self._timeout = timeout
        self._conn: Optional[Connection] = None

    async def __aenter__(self):
        conn = await asyncio.to_thread(
            sqlite3.connect,
            self._database,
            timeout=self._timeout,
            check_same_thread=False,
        )
        self._conn = Connection(conn)
        return self._conn

    async def __aexit__(self, exc_type, exc, tb):
        if self._conn is not None:
            await self._conn.close()


def connect(database: str, timeout: float = 5.0):
    return _ConnectCtx(database, timeout=timeout)
