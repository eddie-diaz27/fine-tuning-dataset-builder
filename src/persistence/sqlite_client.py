"""
SQLite client for dataset persistence.
Handles database operations, CSV export, and backups.
"""

import sqlite3
import json
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import uuid

from persistence.models import CREATE_TABLES_SQL, CREATE_METADATA_TABLE_SQL, CREATE_CONFIGS_TABLE_SQL


class SQLiteClient:
    """SQLite database client for persistence"""

    def __init__(self, db_path: str = "memory/dataset_builder.db"):
        """
        Initialize SQLite client.

        Args:
            db_path: Path to database file
        """
        self.db_path = db_path
        self.conn = None
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create database and tables if they don't exist"""
        # Ensure directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        # Connect to database
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # Enable column access by name

            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")

            # Check database integrity
            result = self.conn.execute("PRAGMA integrity_check").fetchone()
            if result[0] != "ok":
                raise sqlite3.DatabaseError("Database integrity check failed")

            # Create tables
            self.conn.executescript(CREATE_TABLES_SQL)
            self.conn.executescript(CREATE_METADATA_TABLE_SQL)
            self.conn.executescript(CREATE_CONFIGS_TABLE_SQL)
            self.conn.commit()

        except sqlite3.DatabaseError as e:
            print(f"Warning: Database error: {e}")
            self._attempt_recovery()

    def _attempt_recovery(self):
        """Attempt to recover from database corruption"""
        print("Attempting database recovery...")

        # Try to restore from backup
        backup_dir = Path("memory/backups")
        if backup_dir.exists():
            backups = sorted(backup_dir.glob("backup_*.db"), reverse=True)
            if backups:
                latest_backup = backups[0]
                print(f"Restoring from backup: {latest_backup.name}")

                try:
                    shutil.copy2(latest_backup, self.db_path)
                    self.conn = sqlite3.connect(self.db_path)
                    self.conn.row_factory = sqlite3.Row
                    self.conn.execute("PRAGMA foreign_keys = ON")
                    print("Recovery successful!")
                    return
                except Exception as e:
                    print(f"Recovery from backup failed: {e}")

        # If no backup or recovery failed, create fresh database
        print("Creating fresh database...")
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(CREATE_TABLES_SQL)
        self.conn.executescript(CREATE_METADATA_TABLE_SQL)
        self.conn.commit()
        print("Fresh database created!")

    def save_dataset(
        self,
        dataset_id: str,
        name: str,
        provider: str,
        model: str,
        mode: str,
        samples: List,
        config: Dict
    ) -> str:
        """
        Save dataset and samples to database.

        Args:
            dataset_id: Unique dataset ID
            name: Dataset name
            provider: Provider name
            model: Model name
            mode: Generation mode
            samples: List of Sample objects
            config: Configuration dict

        Returns:
            dataset_id
        """
        # Calculate stats
        total_tokens = sum(s.token_length for s in samples)

        # Insert dataset record
        self.conn.execute("""
            INSERT INTO datasets
            (id, name, provider, model, mode, total_samples, total_tokens, config)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            name,
            provider,
            model,
            mode,
            len(samples),
            total_tokens,
            json.dumps(config)
        ))

        # Insert samples
        for sample in samples:
            sample_id = str(uuid.uuid4())

            # Get split - check both direct attribute and metadata
            split = "train"  # Default
            if hasattr(sample, 'split') and sample.split:
                split = sample.split
            elif sample.metadata and 'split' in sample.metadata:
                split = sample.metadata['split']

            self.conn.execute("""
                INSERT INTO samples
                (id, dataset_id, content, token_length, topic_keywords, source_file, split)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                sample_id,
                dataset_id,
                json.dumps(sample.content),
                sample.token_length,
                ",".join(sample.topic_keywords) if sample.topic_keywords else "",
                sample.source_file,
                split
            ))

        self.conn.commit()
        return dataset_id

    def get_dataset(self, dataset_id: str) -> Optional[Dict]:
        """
        Retrieve dataset with all samples.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dict with dataset info and samples, or None if not found
        """
        # Get dataset info
        dataset_row = self.conn.execute("""
            SELECT * FROM datasets WHERE id = ?
        """, (dataset_id,)).fetchone()

        if not dataset_row:
            return None

        # Get samples
        sample_rows = self.conn.execute("""
            SELECT * FROM samples WHERE dataset_id = ?
            ORDER BY created_at
        """, (dataset_id,)).fetchall()

        # Convert to dict
        dataset = dict(dataset_row)
        dataset['config'] = json.loads(dataset['config']) if dataset['config'] else {}

        samples = []
        for row in sample_rows:
            sample = dict(row)
            sample['content'] = json.loads(sample['content'])
            sample['topic_keywords'] = sample['topic_keywords'].split(',') if sample['topic_keywords'] else []
            samples.append(sample)

        dataset['samples'] = samples

        return dataset

    def get_samples(self, dataset_id: str) -> List[Dict]:
        """
        Get all samples for a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            List of sample dicts
        """
        sample_rows = self.conn.execute("""
            SELECT * FROM samples WHERE dataset_id = ?
            ORDER BY created_at
        """, (dataset_id,)).fetchall()

        samples = []
        for row in sample_rows:
            sample = dict(row)
            # Keep content as JSON string for compatibility
            # Don't parse it here - let callers decide
            samples.append(sample)

        return samples

    def list_datasets(self, limit: int = 50) -> List[Dict]:
        """
        List all datasets.

        Args:
            limit: Maximum number of datasets to return

        Returns:
            List of dataset info dicts
        """
        rows = self.conn.execute("""
            SELECT * FROM datasets
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()

        datasets = []
        for row in rows:
            dataset = dict(row)
            dataset['config'] = json.loads(dataset['config']) if dataset['config'] else {}
            datasets.append(dataset)

        return datasets

    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Delete dataset and all its samples.

        Args:
            dataset_id: Dataset ID

        Returns:
            True if deleted, False if not found
        """
        # Check if exists
        exists = self.conn.execute("""
            SELECT COUNT(*) FROM datasets WHERE id = ?
        """, (dataset_id,)).fetchone()[0]

        if not exists:
            return False

        # Delete dataset (samples will be deleted automatically due to CASCADE)
        self.conn.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
        self.conn.commit()

        return True

    def export_csv(self, dataset_id: str, output_path: str):
        """
        Export dataset metadata to CSV.

        Args:
            dataset_id: Dataset ID
            output_path: Path to output CSV file
        """
        import csv

        # Get samples
        samples = self.conn.execute("""
            SELECT id, content, token_length, topic_keywords, source_file, split
            FROM samples
            WHERE dataset_id = ?
            ORDER BY created_at
        """, (dataset_id,)).fetchall()

        # Write CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'sample_id',
                'content_preview',
                'token_length',
                'topic_keywords',
                'source_file',
                'split'
            ])

            # Rows
            for sample in samples:
                content = json.loads(sample['content'])
                user_text = content.get('user', '')

                # Create preview (first 100 chars)
                preview = user_text[:100].replace('\n', ' ')
                if len(user_text) > 100:
                    preview += "..."

                writer.writerow([
                    sample['id'],
                    preview,
                    sample['token_length'],
                    sample['topic_keywords'],
                    sample['source_file'],
                    sample['split']
                ])

    def backup_database(self, backup_dir: str = "memory/backups", max_backups: int = 10):
        """
        Create timestamped backup of database.

        Args:
            backup_dir: Directory to store backups
            max_backups: Maximum number of backups to keep
        """
        # Create backup directory
        os.makedirs(backup_dir, exist_ok=True)

        # Create backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(backup_dir, f"backup_{timestamp}.db")

        # Copy database
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to: {backup_path}")

        # Clean old backups
        self._cleanup_old_backups(backup_dir, max_backups)

    def _cleanup_old_backups(self, backup_dir: str, max_backups: int):
        """
        Remove old backups, keeping only the most recent.

        Args:
            backup_dir: Backup directory
            max_backups: Maximum backups to keep
        """
        backup_files = sorted(
            Path(backup_dir).glob("backup_*.db"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old backups
        for old_backup in backup_files[max_backups:]:
            old_backup.unlink()
            print(f"Removed old backup: {old_backup.name}")

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dict with statistics
        """
        stats = {}

        # Total datasets
        stats['total_datasets'] = self.conn.execute(
            "SELECT COUNT(*) FROM datasets"
        ).fetchone()[0]

        # Total samples
        stats['total_samples'] = self.conn.execute(
            "SELECT COUNT(*) FROM samples"
        ).fetchone()[0]

        # Total tokens
        stats['total_tokens'] = self.conn.execute(
            "SELECT SUM(total_tokens) FROM datasets"
        ).fetchone()[0] or 0

        # Provider distribution
        provider_dist = self.conn.execute("""
            SELECT provider, COUNT(*) as count
            FROM datasets
            GROUP BY provider
        """).fetchall()
        stats['provider_distribution'] = {row['provider']: row['count'] for row in provider_dist}

        # Mode distribution
        mode_dist = self.conn.execute("""
            SELECT mode, COUNT(*) as count
            FROM datasets
            GROUP BY mode
        """).fetchall()
        stats['mode_distribution'] = {row['mode']: row['count'] for row in mode_dist}

        # Database size
        if os.path.exists(self.db_path):
            stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

        return stats

    # ===== Configuration Management =====

    def save_config(self, name: str, config: Dict[str, Any]) -> str:
        """
        Save a configuration for reuse.

        Args:
            name: Unique name for the configuration
            config: Configuration dictionary

        Returns:
            Configuration ID

        Raises:
            ValueError: If config name already exists
        """
        config_id = str(uuid.uuid4())
        config_json = json.dumps(config)

        try:
            self.conn.execute(
                """
                INSERT INTO saved_configs (id, name, config, created_at, last_used)
                VALUES (?, ?, ?, ?, ?)
                """,
                (config_id, name, config_json, datetime.now().isoformat(), None)
            )
            self.conn.commit()
            return config_id
        except sqlite3.IntegrityError:
            raise ValueError(f"Configuration with name '{name}' already exists")

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a saved configuration by name.

        Args:
            name: Configuration name

        Returns:
            Configuration dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT id, config FROM saved_configs WHERE name = ?",
            (name,)
        )
        row = cursor.fetchone()

        if row:
            # Update last_used timestamp
            self.conn.execute(
                "UPDATE saved_configs SET last_used = ? WHERE id = ?",
                (datetime.now().isoformat(), row['id'])
            )
            self.conn.commit()

            return {
                'id': row['id'],
                'config': json.loads(row['config'])
            }

        return None

    def list_saved_configs(self) -> List[Dict[str, Any]]:
        """
        List all saved configurations.

        Returns:
            List of configuration dictionaries with metadata
        """
        cursor = self.conn.execute(
            """
            SELECT id, name, config, created_at, last_used
            FROM saved_configs
            ORDER BY last_used DESC, created_at DESC
            """
        )
        rows = cursor.fetchall()

        configs = []
        for row in rows:
            config_data = json.loads(row['config'])
            configs.append({
                'id': row['id'],
                'name': row['name'],
                'provider': config_data.get('dataset', {}).get('provider', 'Unknown'),
                'model': config_data.get('dataset', {}).get('model', 'Unknown'),
                'mode': config_data.get('agent', {}).get('mode', 'Unknown'),
                'created_at': row['created_at'],
                'last_used': row['last_used']
            })

        return configs

    def update_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        Update an existing configuration.

        Args:
            name: Configuration name
            config: Updated configuration dictionary

        Returns:
            True if updated, False if config not found
        """
        config_json = json.dumps(config)

        cursor = self.conn.execute(
            "UPDATE saved_configs SET config = ? WHERE name = ?",
            (config_json, name)
        )
        self.conn.commit()

        return cursor.rowcount > 0

    def delete_config(self, name: str):
        """
        Delete a saved configuration.

        Args:
            name: Configuration name
        """
        self.conn.execute("DELETE FROM saved_configs WHERE name = ?", (name,))
        self.conn.commit()

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
