import aiosqlite
import hashlib
import json
import re
import time
import os
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Any, Dict

class RequestRepository:
    def __init__(self, db_path: str = "requests.db"):
        self.db_path = db_path
        self.templates = [] # Cache templates in memory for faster matching

    async def initialize(self):
        async with aiosqlite.connect(self.db_path) as db:
            # Enable WAL mode for better concurrency
            await db.execute("PRAGMA journal_mode=WAL;")
            await db.execute("PRAGMA synchronous=NORMAL;")
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    input_hash TEXT UNIQUE,
                    input_text TEXT,
                    output_text TEXT,
                    model_name TEXT,
                    created_at REAL,
                    image_hash TEXT
                )
            """)
            
            # Check if image_hash column exists (migration for existing DB)
            try:
                await db.execute("ALTER TABLE requests ADD COLUMN image_hash TEXT")
            except Exception:
                pass # Column likely exists

            await db.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern TEXT,
                    minimized_prompt_format TEXT,
                    description TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS model_call_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at REAL,
                    requested_model TEXT,
                    actual_model TEXT,
                    config_file TEXT,
                    provider TEXT,
                    response_format TEXT,
                    has_images INTEGER,
                    input_hash TEXT,
                    status TEXT,
                    error_code INTEGER,
                    error_text TEXT,
                    tokens_used INTEGER,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms REAL,
                    cost_usd REAL
                )
            """)
            # Best-effort migration for existing DBs.
            try:
                await db.execute("ALTER TABLE model_call_logs ADD COLUMN input_tokens INTEGER")
            except Exception:
                pass
            try:
                await db.execute("ALTER TABLE model_call_logs ADD COLUMN output_tokens INTEGER")
            except Exception:
                pass
            try:
                await db.execute("ALTER TABLE model_call_logs ADD COLUMN latency_ms REAL")
            except Exception:
                pass
            try:
                await db.execute("ALTER TABLE model_call_logs ADD COLUMN cost_usd REAL")
            except Exception:
                pass
            await db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON requests(input_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_img_hash ON requests(image_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_call_logs_created_at ON model_call_logs(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_call_logs_actual_model ON model_call_logs(actual_model)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_call_logs_status ON model_call_logs(status)")
            await db.commit()
            
            # Load templates
            async with db.execute("SELECT id, pattern, minimized_prompt_format, description FROM templates") as cursor:
                async for row in cursor:
                    try:
                        self.templates.append({
                            "id": row[0],
                            "pattern": re.compile(row[1], re.DOTALL | re.IGNORECASE),
                            "format": row[2],
                            "description": row[3]
                        })
                    except Exception as e:
                        print(f"Error compiling template {row[0]}: {e}")

    def compute_hash(self, text: str, images: list = None, model_name: Optional[str] = None) -> str:
        scope = model_name or "no-model"
        content = f"{scope}\n{text}"
        if images:
            content += json.dumps(images, sort_keys=True)
            
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def compute_image_hash(self, images: list) -> Optional[str]:
        if not images:
            return None
        content = json.dumps(images, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def get_cached_response(self, input_hash: str) -> Optional[str]:
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;") # Optimization for readers
                async with db.execute("SELECT output_text FROM requests WHERE input_hash = ?", (input_hash,)) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return row[0]
        except Exception as e:
            print(f"DB Read Error: {e}")
        return None

    async def get_cached_response_by_image_hash(self, image_hash: str, model_name: Optional[str] = None) -> Optional[Tuple[str, str]]:
        """Returns (output_text, original_input_text) for the most recent request with these images."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;")
                if model_name:
                    query = (
                        "SELECT output_text, input_text FROM requests "
                        "WHERE image_hash = ? AND model_name = ? "
                        "ORDER BY created_at DESC LIMIT 1"
                    )
                    params = (image_hash, model_name)
                else:
                    query = (
                        "SELECT output_text, input_text FROM requests "
                        "WHERE image_hash = ? ORDER BY created_at DESC LIMIT 1"
                    )
                    params = (image_hash,)
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        return row[0], row[1]
        except Exception as e:
            print(f"DB Read Error (Image Hash): {e}")
        return None

    async def get_recent_requests(self, limit: int = 5) -> List[dict]:
        """Get recent requests for browsing cache."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;")
                async with db.execute(
                    "SELECT input_text, output_text, model_name, created_at FROM requests ORDER BY created_at DESC LIMIT ?", 
                    (limit,)
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [
                        {
                            "input": row[0],
                            "output": row[1],
                            "model": row[2],
                            "timestamp": row[3]
                        }
                        for row in rows
                    ]
        except Exception as e:
            print(f"DB Read Error: {e}")
            return []


    async def save_request(self, input_hash: str, input_text: str, output_text: str, model_name: str, image_hash: str = None):
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                try:
                    await db.execute(
                        "INSERT INTO requests (input_hash, input_text, output_text, model_name, created_at, image_hash) VALUES (?, ?, ?, ?, ?, ?)",
                        (input_hash, input_text, output_text, model_name, time.time(), image_hash)
                    )
                    await db.commit()
                except aiosqlite.IntegrityError:
                    # Already exists, maybe update timestamp?
                    pass
        except Exception as e:
            print(f"DB Write Error: {e}")

    async def save_model_call(
        self,
        requested_model: Optional[str],
        actual_model: Optional[str],
        config_file: Optional[str],
        provider: Optional[str],
        response_format: str,
        has_images: bool,
        input_hash: Optional[str],
        status: str,
        error_code: Optional[int] = None,
        error_text: Optional[str] = None,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: Optional[float] = None,
        cost_usd: Optional[float] = None,
    ):
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute(
                    """
                    INSERT INTO model_call_logs (
                        created_at, requested_model, actual_model, config_file, provider,
                        response_format, has_images, input_hash, status, error_code, error_text,
                        tokens_used, input_tokens, output_tokens, latency_ms, cost_usd
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        time.time(),
                        requested_model,
                        actual_model,
                        config_file,
                        provider,
                        response_format,
                        1 if has_images else 0,
                        input_hash,
                        status,
                        error_code,
                        error_text,
                        int(tokens_used or 0),
                        int(input_tokens or 0),
                        int(output_tokens or 0),
                        float(latency_ms) if latency_ms is not None else None,
                        float(cost_usd) if cost_usd is not None else None,
                    ),
                )
                await db.commit()
        except Exception as e:
            print(f"DB Call Log Write Error: {e}")

    async def get_model_call_logs(
        self,
        limit: int = 200,
        model: Optional[str] = None,
        status: Optional[str] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> dict:
        limit = max(1, min(int(limit or 200), 2000))
        conditions = []
        params: List[Any] = []

        if model:
            conditions.append("(actual_model = ? OR requested_model = ?)")
            params.extend([model, model])
        if status and status != "all":
            conditions.append("status = ?")
            params.append(status)
        if start_ts is not None:
            conditions.append("created_at >= ?")
            params.append(float(start_ts))
        if end_ts is not None:
            conditions.append("created_at <= ?")
            params.append(float(end_ts))

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows_query = f"""
            SELECT created_at, requested_model, actual_model, config_file, provider,
                   response_format, has_images, input_hash, status, error_code, error_text,
                   tokens_used, input_tokens, output_tokens, latency_ms, cost_usd
            FROM model_call_logs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        total_query = f"SELECT COUNT(*) FROM model_call_logs {where_clause}"
        status_query = f"""
            SELECT status, COUNT(*) as cnt
            FROM model_call_logs
            {where_clause}
            GROUP BY status
        """

        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;")

                async with db.execute(rows_query, tuple(params + [limit])) as cursor:
                    raw_rows = await cursor.fetchall()

                async with db.execute(total_query, tuple(params)) as cursor:
                    total_row = await cursor.fetchone()
                    total = int(total_row[0] if total_row else 0)

                async with db.execute(status_query, tuple(params)) as cursor:
                    status_rows = await cursor.fetchall()

            records = []
            for row in raw_rows:
                ts = float(row[0] or 0)
                iso_ts = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z") if ts else None
                records.append({
                    "ts": iso_ts,
                    "timestamp": ts,
                    "requested_model": row[1],
                    "actual_model": row[2],
                    "config_file": row[3],
                    "provider": row[4],
                    "response_format": row[5],
                    "has_images": bool(row[6]),
                    "input_hash": row[7],
                    "status": row[8],
                    "error_code": row[9],
                    "error_text": row[10],
                    "tokens_used": int(row[11] or 0),
                    "input_tokens": int(row[12] or 0),
                    "output_tokens": int(row[13] or 0),
                    "latency_ms": float(row[14]) if row[14] is not None else None,
                    "cost_usd": float(row[15]) if row[15] is not None else None,
                })

            by_status = {str(s or "unknown"): int(c or 0) for s, c in status_rows}
            return {
                "records": records,
                "summary": {
                    "total": total,
                    "returned": len(records),
                    "by_status": by_status,
                },
            }
        except Exception as e:
            print(f"DB Read Error (Model Call Logs): {e}")
            return {"records": [], "summary": {"total": 0, "returned": 0, "by_status": {}}}

    async def get_model_call_metrics(
        self,
        days: int = 7,
        status: Optional[str] = "success",
    ) -> Dict[str, Any]:
        days = max(1, min(int(days or 7), 365))
        start_ts = time.time() - (days * 86400)
        where_parts = ["created_at >= ?"]
        params: List[Any] = [float(start_ts)]
        if status and status != "all":
            where_parts.append("status = ?")
            params.append(status)
        where_clause = " AND ".join(where_parts)
        query = f"""
            SELECT
                actual_model,
                COUNT(*) as calls,
                AVG(COALESCE(latency_ms, 0)) as avg_latency_ms,
                SUM(COALESCE(tokens_used, 0)) as total_tokens,
                AVG(COALESCE(tokens_used, 0)) as avg_tokens_per_call,
                SUM(COALESCE(cost_usd, 0)) as total_cost_usd,
                AVG(COALESCE(cost_usd, 0)) as avg_cost_per_call_usd
            FROM model_call_logs
            WHERE {where_clause} AND actual_model IS NOT NULL AND actual_model != ''
            GROUP BY actual_model
            ORDER BY calls DESC
        """
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;")
                async with db.execute(query, tuple(params)) as cursor:
                    rows = await cursor.fetchall()

            by_model: Dict[str, Any] = {}
            for row in rows:
                model = str(row[0] or "").strip()
                if not model:
                    continue
                by_model[model] = {
                    "calls": int(row[1] or 0),
                    "avg_latency_ms": round(float(row[2] or 0.0), 2),
                    "total_tokens": int(row[3] or 0),
                    "avg_tokens_per_call": round(float(row[4] or 0.0), 2),
                    "total_cost_usd": round(float(row[5] or 0.0), 6),
                    "avg_cost_per_call_usd": round(float(row[6] or 0.0), 6),
                }
            return {"days": days, "status_filter": status or "all", "models": by_model}
        except Exception as e:
            print(f"DB Read Error (Model Metrics): {e}")
            return {"days": days, "status_filter": status or "all", "models": {}}

    def find_matching_template(self, text: str) -> Tuple[Optional[str], Optional[dict]]:
        """
        Returns (optimized_prompt, extracted_fields) if a template matches.
        Otherwise (None, None).
        """
        for tmpl in self.templates:
            match = tmpl["pattern"].match(text)
            if match:
                # Extract groups
                groups = match.groups()
                # If template has named groups, we could use groupdict(), but for now assuming positional or basic
                # If the regex has groups, we can use them to format the minimized string
                
                # Check if minimized_prompt_format uses {0}, {1} etc or just appends
                # Simple implementation: Use the groups to format the string
                # Note: Python's format expects {0}, {1} etc.
                # But re groups are 1-based.
                
                try:
                    # Create a context for formatting
                    # If regex was (.*) and (.*), groups is (val1, val2)
                    # format string should use {0}, {1}
                    optimized = tmpl["format"].format(*groups)
                    return optimized, {"groups": groups, "template_id": tmpl["id"]}
                except Exception as e:
                    print(f"Error formatting template {tmpl['id']}: {e}")
                    
        return None, None

    async def add_template(self, pattern: str, format_str: str, description: str = ""):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO templates (pattern, minimized_prompt_format, description) VALUES (?, ?, ?)",
                (pattern, format_str, description)
            )
            await db.commit()
            
            # Update local cache
            try:
                self.templates.append({
                    "id": None, # Reload or ignore ID for now
                    "pattern": re.compile(pattern, re.DOTALL | re.IGNORECASE),
                    "format": format_str,
                    "description": description
                })
            except Exception as e:
                print(f"Error compiling new template: {e}")
