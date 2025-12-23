"""MongoDB Connection Manager for QuestionBank Generator."""

import logging
from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection

from questionbank.config import config

logger = logging.getLogger(__name__)


class MongoDBManager:
    """Singleton MongoDB connection manager."""

    _instance: Optional["MongoDBManager"] = None
    _client: Optional[MongoClient] = None
    _db: Optional[Database] = None

    def __new__(cls) -> "MongoDBManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._client is None:
            self._connect()

    def _connect(self) -> None:
        """Establish MongoDB connection."""
        try:
            mongo_uri = config.mongodb.uri
            if not mongo_uri:
                raise ValueError(
                    "MONGODB_URI not found in environment variables. "
                    "Please create a .env file with MONGODB_URI. "
                    "See .env.example for template."
                )

            db_name = config.mongodb.db_name

            self._client = MongoClient(mongo_uri)
            self._db = self._client[db_name]

            # Test connection
            self._client.admin.command("ping")
            logger.info(f"[MONGODB] Connected to database: {db_name}")

        except Exception as e:
            logger.error(f"[MONGODB] Connection failed: {e}")
            raise

    @property
    def db(self) -> Database:
        """Get database instance."""
        if self._db is None:
            self._connect()
        return self._db

    @property
    def scraped_questions(self) -> Collection:
        """Get scraped_questions collection (source)."""
        return self.db[config.mongodb.source_collection]

    @property
    def generated_questions(self) -> Collection:
        """Get generated_questions collection (output)."""
        return self.db[config.mongodb.output_collection]

    def test_connection(self) -> bool:
        """Test if MongoDB connection is working."""
        try:
            self._client.admin.command("ping")
            collections = self.db.list_collection_names()
            logger.info(f"[MONGODB] Connection OK. Collections: {collections}")
            return True
        except Exception as e:
            logger.error(f"[MONGODB] Connection test failed: {e}")
            return False

    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            logger.info("[MONGODB] Connection closed")


# Global instance
mongo_db = MongoDBManager()
