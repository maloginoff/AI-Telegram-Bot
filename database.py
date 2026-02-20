import aiosqlite
import logging
from datetime import datetime, date
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._create_tables()
        logger.info("Database connected: %s", self._path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None
            logger.info("Database closed")

    @property
    def db(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Database not connected")
        return self._db

    async def _create_tables(self) -> None:
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_active TEXT NOT NULL DEFAULT (datetime('now')),
                is_banned INTEGER NOT NULL DEFAULT 0,
                total_messages INTEGER NOT NULL DEFAULT 0,
                selected_model TEXT,
                selected_provider TEXT
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                model_used TEXT,
                response_time_ms INTEGER,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                provider TEXT NOT NULL,
                key_hash TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'exhausted', 'error')),
                total_requests INTEGER NOT NULL DEFAULT 0,
                exhausted_count INTEGER NOT NULL DEFAULT 0,
                last_used TEXT,
                last_exhausted TEXT
            );

            CREATE TABLE IF NOT EXISTS bot_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                total_requests INTEGER NOT NULL DEFAULT 0,
                unique_users INTEGER NOT NULL DEFAULT 0,
                avg_response_time REAL NOT NULL DEFAULT 0.0
            );
        """)
        await self.db.commit()

    # ---- Users ----

    async def upsert_user(
        self, user_id: int, username: str | None, first_name: str | None
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO users (user_id, username, first_name)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                username = excluded.username,
                first_name = excluded.first_name,
                last_active = datetime('now')
            """,
            (user_id, username, first_name),
        )
        await self.db.commit()

    async def get_user(self, user_id: int) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def is_banned(self, user_id: int) -> bool:
        cursor = await self.db.execute(
            "SELECT is_banned FROM users WHERE user_id = ?", (user_id,)
        )
        row = await cursor.fetchone()
        return bool(row["is_banned"]) if row else False

    async def set_ban(self, user_id: int, banned: bool) -> bool:
        cursor = await self.db.execute(
            "UPDATE users SET is_banned = ? WHERE user_id = ?",
            (int(banned), user_id),
        )
        await self.db.commit()
        return cursor.rowcount > 0

    async def increment_user_messages(self, user_id: int) -> None:
        await self.db.execute(
            """
            UPDATE users SET total_messages = total_messages + 1,
                             last_active = datetime('now')
            WHERE user_id = ?
            """,
            (user_id,),
        )
        await self.db.commit()

    async def set_user_model(
        self, user_id: int, provider: str, model: str
    ) -> None:
        await self.db.execute(
            "UPDATE users SET selected_provider = ?, selected_model = ? WHERE user_id = ?",
            (provider, model, user_id),
        )
        await self.db.commit()

    async def get_user_model(self, user_id: int) -> tuple[str | None, str | None]:
        cursor = await self.db.execute(
            "SELECT selected_provider, selected_model FROM users WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return row["selected_provider"], row["selected_model"]
        return None, None

    async def get_all_user_ids(self) -> list[int]:
        cursor = await self.db.execute(
            "SELECT user_id FROM users WHERE is_banned = 0"
        )
        rows = await cursor.fetchall()
        return [row["user_id"] for row in rows]

    async def get_total_users(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) as cnt FROM users")
        row = await cursor.fetchone()
        return row["cnt"]

    async def get_top_users(self, limit: int = 10) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT user_id, username, first_name, total_messages, last_active "
            "FROM users ORDER BY total_messages DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ---- Messages / Context ----

    async def save_message(
        self,
        user_id: int,
        role: str,
        content: str,
        model_used: str | None = None,
        response_time_ms: int | None = None,
    ) -> None:
        await self.db.execute(
            """
            INSERT INTO messages (user_id, role, content, model_used, response_time_ms)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, role, content, model_used, response_time_ms),
        )
        await self.db.commit()

    async def get_context(
        self, user_id: int, limit: int = 15
    ) -> list[dict[str, str]]:
        cursor = await self.db.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, created_at FROM messages
                WHERE user_id = ? AND role IN ('user', 'assistant')
                ORDER BY created_at DESC LIMIT ?
            ) sub ORDER BY created_at ASC
            """,
            (user_id, limit),
        )
        rows = await cursor.fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    async def clear_context(self, user_id: int) -> int:
        cursor = await self.db.execute(
            "DELETE FROM messages WHERE user_id = ?", (user_id,)
        )
        await self.db.commit()
        return cursor.rowcount

    async def get_messages_today(self) -> int:
        today = date.today().isoformat()
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE date(created_at) = ?",
            (today,),
        )
        row = await cursor.fetchone()
        return row["cnt"]

    async def get_total_messages(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) as cnt FROM messages")
        row = await cursor.fetchone()
        return row["cnt"]

    async def get_avg_response_time(self) -> float:
        cursor = await self.db.execute(
            """
            SELECT AVG(response_time_ms) as avg_ms FROM messages
            WHERE response_time_ms IS NOT NULL AND date(created_at) = date('now')
            """
        )
        row = await cursor.fetchone()
        return round(row["avg_ms"] or 0.0, 1)

    # ---- API Keys ----

    async def upsert_api_key(self, provider: str, key_hash: str) -> None:
        await self.db.execute(
            """
            INSERT INTO api_keys (provider, key_hash)
            VALUES (?, ?)
            ON CONFLICT(key_hash) DO NOTHING
            """,
            (provider, key_hash),
        )
        await self.db.commit()

    async def get_api_keys(self, provider: str | None = None) -> list[dict]:
        if provider:
            cursor = await self.db.execute(
                "SELECT * FROM api_keys WHERE provider = ? ORDER BY id",
                (provider,),
            )
        else:
            cursor = await self.db.execute(
                "SELECT * FROM api_keys ORDER BY provider, id"
            )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_key_status(self, key_hash: str, status: str) -> None:
        extra = ""
        if status == "exhausted":
            extra = ", last_exhausted = datetime('now'), exhausted_count = exhausted_count + 1"
        await self.db.execute(
            f"UPDATE api_keys SET status = ?{extra} WHERE key_hash = ?",
            (status, key_hash),
        )
        await self.db.commit()

    async def increment_key_requests(self, key_hash: str) -> None:
        await self.db.execute(
            """
            UPDATE api_keys SET total_requests = total_requests + 1,
                                last_used = datetime('now')
            WHERE key_hash = ?
            """,
            (key_hash,),
        )
        await self.db.commit()

    async def reset_exhausted_keys(self, provider: str, cooldown_minutes: int) -> int:
        cursor = await self.db.execute(
            """
            UPDATE api_keys SET status = 'active'
            WHERE provider = ? AND status = 'exhausted'
            AND last_exhausted IS NOT NULL
            AND (julianday('now') - julianday(last_exhausted)) * 1440 >= ?
            """,
            (provider, cooldown_minutes),
        )
        await self.db.commit()
        return cursor.rowcount

    async def get_active_key_count(self, provider: str) -> int:
        cursor = await self.db.execute(
            "SELECT COUNT(*) as cnt FROM api_keys WHERE provider = ? AND status = 'active'",
            (provider,),
        )
        row = await cursor.fetchone()
        return row["cnt"]

    async def get_earliest_exhausted_recovery(self, provider: str, cooldown_minutes: int) -> str | None:
        cursor = await self.db.execute(
            """
            SELECT MIN(datetime(last_exhausted, '+' || ? || ' minutes')) as recovery
            FROM api_keys
            WHERE provider = ? AND status = 'exhausted' AND last_exhausted IS NOT NULL
            """,
            (cooldown_minutes, provider),
        )
        row = await cursor.fetchone()
        return row["recovery"] if row else None

    # ---- Bot Stats ----

    async def update_daily_stats(self) -> None:
        today = date.today().isoformat()
        messages_today = await self.get_messages_today()
        cursor = await self.db.execute(
            "SELECT COUNT(DISTINCT user_id) as cnt FROM messages WHERE date(created_at) = ?",
            (today,),
        )
        row = await cursor.fetchone()
        unique_today = row["cnt"]
        avg_time = await self.get_avg_response_time()

        await self.db.execute(
            """
            INSERT INTO bot_stats (date, total_requests, unique_users, avg_response_time)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_requests = excluded.total_requests,
                unique_users = excluded.unique_users,
                avg_response_time = excluded.avg_response_time
            """,
            (today, messages_today, unique_today, avg_time),
        )
        await self.db.commit()