"""
Database module for IGQK SaaS Platform
Uses SQLite for simplicity and portability
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import contextmanager


class Database:
    """Simple SQLite database for job and model tracking"""

    def __init__(self, db_path: str = "igqk_saas.db"):
        """
        Initialize database

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Jobs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    job_name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_identifier TEXT,
                    model_source TEXT,
                    compression_method TEXT,
                    quality_target REAL,
                    auto_validate BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    error TEXT,
                    results TEXT
                )
            """)

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    job_id TEXT,
                    name TEXT NOT NULL,
                    model_identifier TEXT,
                    model_type TEXT,
                    original_size_mb REAL,
                    compressed_size_mb REAL,
                    compression_ratio REAL,
                    accuracy_original REAL,
                    accuracy_compressed REAL,
                    accuracy_loss REAL,
                    compression_method TEXT,
                    save_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs(job_id)
                )
            """)

            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT NOT NULL,
                    api_key TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    quota_jobs_remaining INTEGER DEFAULT 10,
                    tier TEXT DEFAULT 'free',
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            conn.commit()

    @contextmanager
    def get_connection(self):
        """Get database connection as context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        try:
            yield conn
        finally:
            conn.close()

    # ==================== JOBS ====================

    def create_job(
        self,
        job_id: str,
        job_name: str,
        job_type: str,
        model_identifier: str = None,
        model_source: str = None,
        compression_method: str = None,
        quality_target: float = None,
        auto_validate: bool = False
    ) -> bool:
        """Create a new job"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO jobs (
                        job_id, job_name, job_type, status,
                        model_identifier, model_source, compression_method,
                        quality_target, auto_validate
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job_id, job_name, job_type, "pending",
                    model_identifier, model_source, compression_method,
                    quality_target, auto_validate
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating job: {str(e)}")
            return False

    def update_job_status(
        self,
        job_id: str,
        status: str,
        error: str = None,
        results: Dict[str, Any] = None
    ) -> bool:
        """Update job status"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Serialize results to JSON
                results_json = json.dumps(results) if results else None

                # Set completed_at if status is completed or failed
                if status in ["completed", "failed"]:
                    cursor.execute("""
                        UPDATE jobs
                        SET status = ?,
                            error = ?,
                            results = ?,
                            updated_at = CURRENT_TIMESTAMP,
                            completed_at = CURRENT_TIMESTAMP
                        WHERE job_id = ?
                    """, (status, error, results_json, job_id))
                else:
                    cursor.execute("""
                        UPDATE jobs
                        SET status = ?,
                            error = ?,
                            results = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE job_id = ?
                    """, (status, error, results_json, job_id))

                conn.commit()
                return True
        except Exception as e:
            print(f"Error updating job: {str(e)}")
            return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
                row = cursor.fetchone()

                if row:
                    job = dict(row)
                    # Deserialize results from JSON
                    if job.get("results"):
                        job["results"] = json.loads(job["results"])
                    return job
                return None
        except Exception as e:
            print(f"Error getting job: {str(e)}")
            return None

    def list_jobs(self, limit: int = 50, status: str = None) -> List[Dict[str, Any]]:
        """List all jobs"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                if status:
                    cursor.execute("""
                        SELECT * FROM jobs
                        WHERE status = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (status, limit))
                else:
                    cursor.execute("""
                        SELECT * FROM jobs
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (limit,))

                rows = cursor.fetchall()
                jobs = []
                for row in rows:
                    job = dict(row)
                    if job.get("results"):
                        job["results"] = json.loads(job["results"])
                    jobs.append(job)

                return jobs
        except Exception as e:
            print(f"Error listing jobs: {str(e)}")
            return []

    # ==================== MODELS ====================

    def create_model(
        self,
        model_id: str,
        job_id: str,
        name: str,
        model_identifier: str,
        model_type: str,
        original_size_mb: float,
        compressed_size_mb: float,
        compression_ratio: float,
        compression_method: str,
        save_path: str,
        accuracy_original: float = None,
        accuracy_compressed: float = None,
        accuracy_loss: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a new model record"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                metadata_json = json.dumps(metadata) if metadata else None

                cursor.execute("""
                    INSERT INTO models (
                        model_id, job_id, name, model_identifier, model_type,
                        original_size_mb, compressed_size_mb, compression_ratio,
                        accuracy_original, accuracy_compressed, accuracy_loss,
                        compression_method, save_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model_id, job_id, name, model_identifier, model_type,
                    original_size_mb, compressed_size_mb, compression_ratio,
                    accuracy_original, accuracy_compressed, accuracy_loss,
                    compression_method, save_path, metadata_json
                ))

                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating model: {str(e)}")
            return False

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
                row = cursor.fetchone()

                if row:
                    model = dict(row)
                    if model.get("metadata"):
                        model["metadata"] = json.loads(model["metadata"])
                    return model
                return None
        except Exception as e:
            print(f"Error getting model: {str(e)}")
            return None

    def list_models(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List all models"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM models
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,))

                rows = cursor.fetchall()
                models = []
                for row in rows:
                    model = dict(row)
                    if model.get("metadata"):
                        model["metadata"] = json.loads(model["metadata"])
                    models.append(model)

                return models
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Get platform statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Total jobs
                cursor.execute("SELECT COUNT(*) as count FROM jobs")
                total_jobs = cursor.fetchone()["count"]

                # Completed jobs
                cursor.execute("SELECT COUNT(*) as count FROM jobs WHERE status = 'completed'")
                completed_jobs = cursor.fetchone()["count"]

                # Total models
                cursor.execute("SELECT COUNT(*) as count FROM models")
                total_models = cursor.fetchone()["count"]

                # Total storage saved
                cursor.execute("""
                    SELECT SUM(original_size_mb - compressed_size_mb) as saved
                    FROM models
                """)
                storage_saved_mb = cursor.fetchone()["saved"] or 0.0

                return {
                    "total_jobs": total_jobs,
                    "completed_jobs": completed_jobs,
                    "total_models": total_models,
                    "storage_saved_mb": round(storage_saved_mb, 2),
                    "storage_saved_gb": round(storage_saved_mb / 1024, 2)
                }
        except Exception as e:
            print(f"Error getting stats: {str(e)}")
            return {
                "total_jobs": 0,
                "completed_jobs": 0,
                "total_models": 0,
                "storage_saved_mb": 0.0,
                "storage_saved_gb": 0.0
            }

    # ==================== USERS ====================

    def create_user(
        self,
        user_id: str,
        email: str,
        password_hash: str,
        full_name: str,
        api_key: str = None,
        tier: str = "free"
    ) -> bool:
        """Create a new user"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (
                        user_id, email, password_hash, full_name, api_key, tier
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, email, password_hash, full_name, api_key, tier))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error creating user: {str(e)}")
            return False

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
        except Exception as e:
            print(f"Error getting user: {str(e)}")
            return None

    def update_user_quota(self, user_id: str, quota_remaining: int) -> bool:
        """Update user quota"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users
                    SET quota_jobs_remaining = ?
                    WHERE user_id = ?
                """, (quota_remaining, user_id))
                conn.commit()
                return True
        except Exception as e:
            print(f"Error updating user quota: {str(e)}")
            return False


# Global database instance
db = Database()
