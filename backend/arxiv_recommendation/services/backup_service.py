#!/usr/bin/env python3
"""
Backup Service for ArXiv Recommendation System.

This service provides comprehensive backup and restore functionality
with automatic cleanup, verification, and enhanced security features.
"""

import asyncio
import json
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..database import DatabaseManager

logger = logging.getLogger(__name__)


class BackupService:
    """Service for database backup and restore operations with enhanced security."""
    
    def __init__(self, backup_dir: str = "data"):
        """Initialize the backup service."""
        self.backup_dir = Path(backup_dir) / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.db_manager = DatabaseManager()
        
        # Enhanced backup settings
        self.max_backups = 10
        self.max_age_days = 30
    
    async def create_backup(self, backup_type: str = "pre_clean", metadata: Optional[Dict] = None) -> Dict:
        """
        Create a comprehensive database backup with metadata and verification.
        
        Args:
            backup_type: Type of backup ('pre_clean', 'manual', 'ratings', 'papers', 'full')
            metadata: Additional metadata to include
            
        Returns:
            Dictionary with backup information
        """
        try:
            await self.db_manager.initialize()
            
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{backup_type}_{timestamp}.db"
            backup_path = self.backup_dir / filename
            
            # Get database statistics before backup
            stats = await self._get_database_stats()
            
            # Create the backup
            start_time = datetime.now()
            logger.info(f"Creating {backup_type} backup: {filename}")
            
            # Use SQLite backup API for consistent backup
            await self._create_sqlite_backup(backup_path)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Get backup file size
            backup_size = backup_path.stat().st_size
            
            # Create comprehensive metadata
            backup_metadata = {
                "filename": filename,
                "created_at": end_time.isoformat(),
                "backup_type": backup_type,
                "duration_seconds": duration,
                "size_bytes": backup_size,
                "size_mb": round(backup_size / (1024 * 1024), 2),
                "database_stats": stats,
                "original_db_path": str(self.db_manager.db_path),
                "backup_path": str(backup_path),
                "verified": False
            }
            
            # Add custom metadata if provided
            if metadata:
                backup_metadata.update(metadata)
            
            # Verify backup integrity
            if await self._verify_backup_integrity(backup_path):
                backup_metadata["verified"] = True
            else:
                logger.warning(f"Backup verification failed for {filename}")
            
            # Save metadata file
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(backup_metadata, f, indent=2)
            
            logger.info(f"Backup created successfully: {filename} ({backup_metadata['size_mb']} MB)")
            
            # Clean up old backups
            await self._cleanup_old_backups()
            
            return {
                "success": True,
                "backup_path": str(backup_path),
                "metadata": backup_metadata
            }
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _create_sqlite_backup(self, backup_path: Path) -> None:
        """Create SQLite backup using backup API for consistency."""
        import aiosqlite
        
        # Open source database in read-only mode
        async with aiosqlite.connect(f"file:{self.db_manager.db_path}?mode=ro", uri=True) as source:
            # Create backup database
            async with aiosqlite.connect(backup_path) as backup:
                # Use SQLite backup API for atomic backup
                await source.backup(backup)
    
    async def _get_database_stats(self) -> Dict:
        """Get comprehensive database statistics."""
        import aiosqlite
        
        stats = {}
        
        try:
            async with aiosqlite.connect(self.db_manager.db_path) as db:
                # Count records in each table
                tables = [
                    "papers", "user_ratings", "paper_embeddings", 
                    "search_history", "recommendations_history"
                ]
                
                for table in tables:
                    try:
                        cursor = await db.execute(f"SELECT COUNT(*) FROM {table}")
                        count = await cursor.fetchone()
                        stats[f"{table}_count"] = count[0] if count else 0
                    except Exception as e:
                        logger.warning(f"Could not count {table}: {e}")
                        stats[f"{table}_count"] = 0
                
                # Get database size information
                cursor = await db.execute("PRAGMA page_count")
                page_count = await cursor.fetchone()
                cursor = await db.execute("PRAGMA page_size")
                page_size = await cursor.fetchone()
                
                if page_count and page_size:
                    db_size_bytes = page_count[0] * page_size[0]
                    stats["database_size_bytes"] = db_size_bytes
                    stats["database_size_mb"] = round(db_size_bytes / (1024 * 1024), 2)
                
                # Get last modified time of critical tables
                try:
                    cursor = await db.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type='table' AND name IN ('papers', 'user_ratings')
                    """)
                    tables_exist = await cursor.fetchall()
                    stats["critical_tables_exist"] = len(tables_exist)
                except Exception:
                    stats["critical_tables_exist"] = 0
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}
    
    async def _verify_backup_integrity(self, backup_path: Path) -> bool:
        """Verify backup file integrity using SQLite pragma."""
        try:
            # Use synchronous connection for integrity check
            conn = sqlite3.connect(backup_path)
            cursor = conn.cursor()
            
            # Run comprehensive integrity check
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            # Also check that critical tables exist
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name IN ('papers', 'user_ratings')
            """)
            table_count = cursor.fetchone()
            
            conn.close()
            
            integrity_ok = result and result[0] == "ok"
            tables_ok = table_count and table_count[0] >= 2
            
            return integrity_ok and tables_ok
            
        except Exception as e:
            logger.error(f"Backup integrity check failed: {e}")
            return False
    
    async def _backup_ratings(self, timestamp: str) -> Dict:
        """Backup user ratings to JSON file."""
        try:
            papers = await self.db_manager.get_papers_with_ratings()
            
            # Extract only papers with ratings
            rated_papers = []
            for paper in papers:
                if paper.get("rating") and paper.get("rating") > 0:
                    rated_papers.append({
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "rating": paper["rating"],
                        "notes": paper.get("notes", ""),
                        "rated_at": paper.get("rated_at", "")
                    })
            
            backup_filename = f"user_ratings_backup_{timestamp}.json"
            backup_path = self.backup_dir / backup_filename
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "backup_type": "ratings",
                    "timestamp": timestamp,
                    "total_ratings": len(rated_papers),
                    "ratings": rated_papers
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Ratings backup created: {backup_path}")
            
            return {
                "success": True,
                "backup_type": "ratings",
                "filename": backup_filename,
                "path": str(backup_path),
                "total_ratings": len(rated_papers),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Ratings backup failed: {e}")
            raise
    
    async def _backup_papers(self, timestamp: str) -> Dict:
        """Backup all papers to JSON file."""
        try:
            papers = await self.db_manager.get_papers_with_ratings()
            
            backup_filename = f"papers_backup_{timestamp}.json"
            backup_path = self.backup_dir / backup_filename
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "backup_type": "papers",
                    "timestamp": timestamp,
                    "total_papers": len(papers),
                    "papers": papers
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Papers backup created: {backup_path}")
            
            return {
                "success": True,
                "backup_type": "papers",
                "filename": backup_filename,
                "path": str(backup_path),
                "total_papers": len(papers),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Papers backup failed: {e}")
            raise
    
    async def _backup_full_database(self, timestamp: str) -> Dict:
        """Create a full database backup."""
        try:
            backup_filename = f"database_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # Copy the entire database file
            shutil.copy2(self.db_manager.db_path, backup_path)
            
            logger.info(f"Full database backup created: {backup_path}")
            
            return {
                "success": True,
                "backup_type": "full",
                "filename": backup_filename,
                "path": str(backup_path),
                "timestamp": timestamp
            }
            
        except Exception as e:
            logger.error(f"Full database backup failed: {e}")
            raise
    
    def list_backups(self) -> List[Dict]:
        """List all available backups."""
        backups = []
        
        for backup_file in self.backup_dir.glob("*_backup_*.json"):
            try:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                backups.append({
                    "filename": backup_file.name,
                    "path": str(backup_file),
                    "backup_type": backup_data.get("backup_type", "unknown"),
                    "timestamp": backup_data.get("timestamp", "unknown"),
                    "size_mb": backup_file.stat().st_size / (1024 * 1024),
                    "total_items": backup_data.get("total_ratings", backup_data.get("total_papers", 0))
                })
                
            except Exception as e:
                logger.warning(f"Could not read backup file {backup_file}: {e}")
                continue
        
        # Also check for database backups
        for backup_file in self.backup_dir.glob("database_backup_*.db"):
            backups.append({
                "filename": backup_file.name,
                "path": str(backup_file),
                "backup_type": "full",
                "timestamp": backup_file.name.split("_")[-1].replace(".db", ""),
                "size_mb": backup_file.stat().st_size / (1024 * 1024),
                "total_items": "N/A"
            })
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x["timestamp"], reverse=True)
        
        return backups
    
    async def restore_ratings(self, backup_filename: str) -> Dict:
        """
        Restore user ratings from a backup file.
        
        Args:
            backup_filename: Name of the backup file to restore from
            
        Returns:
            Dictionary with restore results
        """
        try:
            backup_path = self.backup_dir / backup_filename
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            if backup_data.get("backup_type") != "ratings":
                raise ValueError("Backup file is not a ratings backup")
            
            await self.db_manager.initialize()
            
            # Restore each rating
            restored_count = 0
            failed_count = 0
            
            for rating_data in backup_data.get("ratings", []):
                try:
                    success = await self.db_manager.store_user_rating(
                        paper_id=rating_data["paper_id"],
                        rating=rating_data["rating"],
                        notes=rating_data.get("notes", "")
                    )
                    
                    if success:
                        restored_count += 1
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to restore rating for paper {rating_data.get('paper_id')}: {e}")
                    failed_count += 1
            
            logger.info(f"Ratings restore completed: {restored_count} restored, {failed_count} failed")
            
            return {
                "success": True,
                "restored_count": restored_count,
                "failed_count": failed_count,
                "total_in_backup": len(backup_data.get("ratings", [])),
                "backup_timestamp": backup_data.get("timestamp")
            }
            
        except Exception as e:
            logger.error(f"Ratings restore failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def restore_full_database(self, backup_filename: str) -> Dict:
        """
        Restore the full database from a backup file.
        
        Args:
            backup_filename: Name of the database backup file
            
        Returns:
            Dictionary with restore results
        """
        try:
            backup_path = self.backup_dir / backup_filename
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            if not backup_filename.endswith(".db"):
                raise ValueError("Backup file is not a database backup")
            
            # Create a backup of current database first
            current_backup = await self.create_backup("full")
            logger.info(f"Created safety backup before restore: {current_backup.get('filename')}")
            
            # Copy backup over current database
            shutil.copy2(backup_path, self.db_manager.db_path)
            
            logger.info(f"Database restored from: {backup_filename}")
            
            return {
                "success": True,
                "restored_from": backup_filename,
                "safety_backup": current_backup.get("filename")
            }
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_backup(self, backup_filename: str) -> Dict:
        """
        Delete a backup file.
        
        Args:
            backup_filename: Name of the backup file to delete
            
        Returns:
            Dictionary with deletion results
        """
        try:
            backup_path = self.backup_dir / backup_filename
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_filename}")
            
            backup_path.unlink()
            logger.info(f"Backup deleted: {backup_filename}")
            
            return {"success": True, "deleted_file": backup_filename}
            
        except Exception as e:
            logger.error(f"Backup deletion failed: {e}")
            return {"success": False, "error": str(e)}