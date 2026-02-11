import aiosqlite
import hashlib
import json
import re
import time
import os
from typing import Optional, Tuple, List, Any

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
            await db.execute("CREATE INDEX IF NOT EXISTS idx_hash ON requests(input_hash)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_img_hash ON requests(image_hash)")
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

    def compute_hash(self, text: str, images: list = None) -> str:
        content = text
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

    async def get_cached_response_by_image_hash(self, image_hash: str) -> Optional[Tuple[str, str]]:
        """Returns (output_text, original_input_text) for the most recent request with these images."""
        try:
            async with aiosqlite.connect(self.db_path, timeout=5.0) as db:
                await db.execute("PRAGMA query_only = ON;")
                async with db.execute(
                    "SELECT output_text, input_text FROM requests WHERE image_hash = ? ORDER BY created_at DESC LIMIT 1", 
                    (image_hash,)
                ) as cursor:
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
