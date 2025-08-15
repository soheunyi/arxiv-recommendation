#!/usr/bin/env python3
"""
Scheduler module for automated tasks in ArXiv Recommendation System.

This module provides scheduled tasks for:
- Daily paper collection
- Daily score updates
- Preference embedding updates
- Cache maintenance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor

from config import config
from database import DatabaseManager

logger = logging.getLogger(__name__)


class TaskScheduler:
    """
    Manages scheduled tasks for the ArXiv recommendation system.
    
    Uses APScheduler with SQLite job store for persistence across restarts.
    """

    def __init__(self):
        """Initialize the task scheduler."""
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.db_manager = DatabaseManager()
        
        # Task execution statistics
        self.task_stats = {
            "daily_collection": {"last_run": None, "last_status": None, "run_count": 0},
            "daily_scoring": {"last_run": None, "last_status": None, "run_count": 0},
            "preference_update": {"last_run": None, "last_status": None, "run_count": 0},
            "cache_maintenance": {"last_run": None, "last_status": None, "run_count": 0},
            "openalex_enrichment": {"last_run": None, "last_status": None, "run_count": 0}
        }
        
        self._setup_scheduler()

    def _setup_scheduler(self):
        """Configure the APScheduler instance."""
        # Use SQLite job store for persistence
        jobstores = {
            'default': SQLAlchemyJobStore(url=f'sqlite:///{config.database_path.parent}/scheduler.db')
        }
        
        executors = {
            'default': AsyncIOExecutor(),
        }
        
        job_defaults = {
            'coalesce': False,  # Don't combine multiple pending instances
            'max_instances': 1,  # Only one instance of each job at a time
            'misfire_grace_time': 300  # 5 minutes grace period for missed jobs
        }

        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )

    async def start(self):
        """Start the scheduler and add scheduled jobs."""
        if self.scheduler.running:
            logger.warning("Scheduler is already running")
            return

        try:
            # Initialize database
            await self.db_manager.initialize()
            
            # Add scheduled jobs
            await self._add_scheduled_jobs()
            
            # Start the scheduler
            self.scheduler.start()
            logger.info("Task scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            raise

    async def stop(self):
        """Stop the scheduler gracefully."""
        if self.scheduler and self.scheduler.running:
            self.scheduler.shutdown(wait=True)
            logger.info("Task scheduler stopped")

    async def _add_scheduled_jobs(self):
        """Add all scheduled jobs to the scheduler."""
        try:
            # Daily paper collection at 6:00 AM UTC
            self.scheduler.add_job(
                func=self._daily_collection_task,
                trigger=CronTrigger(hour=6, minute=0),
                id='daily_collection',
                name='Daily Paper Collection',
                replace_existing=True
            )
            
            # Daily scoring update at 7:00 AM UTC (after collection)
            self.scheduler.add_job(
                func=self._daily_scoring_task,
                trigger=CronTrigger(hour=7, minute=0),
                id='daily_scoring',
                name='Daily Score Updates',
                replace_existing=True
            )
            
            # Preference embedding update at 8:00 AM UTC (after scoring)
            self.scheduler.add_job(
                func=self._preference_update_task,
                trigger=CronTrigger(hour=8, minute=0),
                id='preference_update',
                name='Preference Embedding Update',
                replace_existing=True
            )
            
            # Cache maintenance every 6 hours
            self.scheduler.add_job(
                func=self._cache_maintenance_task,
                trigger=IntervalTrigger(hours=6),
                id='cache_maintenance',
                name='Cache Maintenance',
                replace_existing=True
            )
            
            # OpenAlex enrichment at 3:00 AM UTC (after cache maintenance)
            self.scheduler.add_job(
                func=self._openalex_enrichment_task,
                trigger=CronTrigger(hour=3, minute=0),
                id='openalex_enrichment',
                name='OpenAlex Data Enrichment',
                replace_existing=True
            )
            
            logger.info("Scheduled jobs added successfully")
            
        except Exception as e:
            logger.error(f"Failed to add scheduled jobs: {e}")
            raise

    async def _daily_collection_task(self):
        """Execute daily paper collection task."""
        task_name = "daily_collection"
        logger.info(f"Starting {task_name} task")
        
        start_time = datetime.utcnow()
        self.task_stats[task_name]["run_count"] += 1
        
        try:
            # Import here to avoid circular imports
            from services.collection_service import CollectionService
            
            # Get collection topics from configuration
            topics = config.collection_topics if hasattr(config, 'collection_topics') else ['machine learning']
            
            total_collected = 0
            for topic in topics:
                try:
                    collection_service = CollectionService(topic)
                    
                    # Generate queries for the topic
                    await collection_service.generate_queries(max_queries=10)
                    
                    # Execute collection with recent papers focus
                    result = await collection_service.execute_collection(
                        max_papers_per_query=20,
                        filter_recent_days=2,  # Only collect papers from last 2 days
                        clean_db=False
                    )
                    
                    collected_count = result.get('papers_collected', 0)
                    total_collected += collected_count
                    
                    logger.info(f"Collected {collected_count} papers for topic: {topic}")
                    
                except Exception as e:
                    logger.error(f"Failed to collect papers for topic {topic}: {e}")
                    continue
            
            # Update task statistics
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"success - {total_collected} papers collected"
            
            # Store execution record in database
            await self._store_task_execution(task_name, True, f"Collected {total_collected} papers", start_time)
            
            logger.info(f"Daily collection task completed - {total_collected} total papers collected")
            
        except Exception as e:
            error_msg = f"Daily collection task failed: {e}"
            logger.error(error_msg)
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"error - {str(e)}"
            
            await self._store_task_execution(task_name, False, str(e), start_time)
            raise

    async def _daily_scoring_task(self):
        """Execute daily scoring update task."""
        task_name = "daily_scoring"
        logger.info(f"Starting {task_name} task")
        
        start_time = datetime.utcnow()
        self.task_stats[task_name]["run_count"] += 1
        
        try:
            # Import here to avoid circular imports
            from .scoring_service import ScoringService
            
            scoring_service = ScoringService()
            
            # Update scores for all unscored and recently added papers
            result = await scoring_service.update_daily_scores()
            
            scored_count = result.get('papers_scored', 0)
            
            # Update task statistics
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"success - {scored_count} papers scored"
            
            await self._store_task_execution(task_name, True, f"Scored {scored_count} papers", start_time)
            
            logger.info(f"Daily scoring task completed - {scored_count} papers scored")
            
        except Exception as e:
            error_msg = f"Daily scoring task failed: {e}"
            logger.error(error_msg)
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"error - {str(e)}"
            
            await self._store_task_execution(task_name, False, str(e), start_time)
            raise

    async def _preference_update_task(self):
        """Execute preference embedding update task."""
        task_name = "preference_update"
        logger.info(f"Starting {task_name} task")
        
        start_time = datetime.utcnow()
        self.task_stats[task_name]["run_count"] += 1
        
        try:
            # Import here to avoid circular imports
            from .preferences import PreferenceManager
            
            preference_manager = PreferenceManager()
            
            # Update both recent and all-time preference embeddings
            result = await preference_manager.update_preference_embeddings()
            
            updated_embeddings = result.get('embeddings_updated', 0)
            
            # Update task statistics
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"success - {updated_embeddings} embeddings updated"
            
            await self._store_task_execution(task_name, True, f"Updated {updated_embeddings} embeddings", start_time)
            
            logger.info(f"Preference update task completed - {updated_embeddings} embeddings updated")
            
        except Exception as e:
            error_msg = f"Preference update task failed: {e}"
            logger.error(error_msg)
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"error - {str(e)}"
            
            await self._store_task_execution(task_name, False, str(e), start_time)
            raise

    async def _cache_maintenance_task(self):
        """Execute cache maintenance task."""
        task_name = "cache_maintenance"
        logger.info(f"Starting {task_name} task")
        
        start_time = datetime.utcnow()
        self.task_stats[task_name]["run_count"] += 1
        
        try:
            from embeddings import EmbeddingManager
            
            embedding_manager = EmbeddingManager()
            
            # Perform cache cleanup and optimization
            result = await embedding_manager.maintain_cache()
            
            cleaned_entries = result.get('cleaned_entries', 0)
            
            # Update task statistics
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"success - {cleaned_entries} entries cleaned"
            
            await self._store_task_execution(task_name, True, f"Cleaned {cleaned_entries} cache entries", start_time)
            
            logger.info(f"Cache maintenance task completed - {cleaned_entries} entries cleaned")
            
        except Exception as e:
            error_msg = f"Cache maintenance task failed: {e}"
            logger.error(error_msg)
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"error - {str(e)}"
            
            await self._store_task_execution(task_name, False, str(e), start_time)
            # Don't raise for maintenance tasks - they're not critical

    async def _store_task_execution(self, task_name: str, success: bool, details: str, start_time: datetime):
        """Store task execution record in database."""
        try:
            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()
            
            # Store in a task_executions table (will need to add this to database schema)
            await self.db_manager.store_task_execution({
                'task_name': task_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration,
                'success': success,
                'details': details
            })
            
        except Exception as e:
            logger.error(f"Failed to store task execution record: {e}")

    async def trigger_task(self, task_name: str) -> Dict[str, Any]:
        """Manually trigger a specific task."""
        if not self.scheduler or not self.scheduler.running:
            raise RuntimeError("Scheduler is not running")
        
        try:
            job = self.scheduler.get_job(task_name)
            if not job:
                raise ValueError(f"Task '{task_name}' not found")
            
            # Execute the job now
            await job.func()
            
            return {
                "task_name": task_name,
                "triggered_at": datetime.utcnow().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Failed to trigger task {task_name}: {e}")
            return {
                "task_name": task_name,
                "triggered_at": datetime.utcnow().isoformat(),
                "status": "error",
                "error": str(e)
            }

    def get_task_status(self) -> Dict[str, Any]:
        """Get status of all scheduled tasks."""
        if not self.scheduler:
            return {"scheduler_status": "not_initialized"}
        
        jobs_info = []
        for job in self.scheduler.get_jobs():
            next_run = job.next_run_time.isoformat() if job.next_run_time else None
            jobs_info.append({
                "id": job.id,
                "name": job.name,
                "next_run": next_run,
                "trigger": str(job.trigger)
            })
        
        return {
            "scheduler_status": "running" if self.scheduler.running else "stopped",
            "jobs": jobs_info,
            "task_statistics": self.task_stats
        }

    async def _openalex_enrichment_task(self):
        """Execute OpenAlex data enrichment task."""
        task_name = "openalex_enrichment"
        logger.info(f"Starting {task_name} task")
        
        start_time = datetime.utcnow()
        self.task_stats[task_name]["run_count"] += 1
        
        try:
            from .services.hybrid_reference_service import HybridReferenceService
            from .config import config
            
            # Initialize hybrid service with email from config
            email = getattr(config, 'openalex_email', None)
            hybrid_service = HybridReferenceService(email=email)
            
            # Get papers that need OpenAlex enrichment
            papers_to_enrich = await self.db_manager.get_papers_without_openalex()
            
            if not papers_to_enrich:
                logger.info("No papers found that need OpenAlex enrichment")
                self.task_stats[task_name]["last_run"] = start_time
                self.task_stats[task_name]["last_status"] = "success - no papers to enrich"
                await self._store_task_execution(task_name, True, "No papers needed enrichment", start_time)
                return
            
            logger.info(f"Found {len(papers_to_enrich)} papers to enrich with OpenAlex data")
            
            # Batch enrich papers
            batch_size = getattr(config, 'openalex_batch_size', 10)
            arxiv_ids = [paper['id'] for paper in papers_to_enrich]
            
            results = await hybrid_service.batch_enrich_papers(arxiv_ids, batch_size=batch_size)
            
            # Update task statistics
            enriched_count = results['enriched_count']
            total_count = results['total_papers']
            error_count = results['error_count']
            
            status_msg = f"success - {enriched_count}/{total_count} papers enriched"
            if error_count > 0:
                status_msg += f", {error_count} errors"
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = status_msg
            
            details = f"Enriched {enriched_count} papers, {error_count} errors, {results['not_found_count']} not found"
            await self._store_task_execution(task_name, True, details, start_time)
            
            logger.info(f"OpenAlex enrichment task completed - {enriched_count}/{total_count} papers enriched")
            
        except Exception as e:
            error_msg = f"OpenAlex enrichment task failed: {e}"
            logger.error(error_msg)
            
            self.task_stats[task_name]["last_run"] = start_time
            self.task_stats[task_name]["last_status"] = f"error - {str(e)}"
            
            await self._store_task_execution(task_name, False, str(e), start_time)
            raise


# Global scheduler instance
_scheduler_instance: Optional[TaskScheduler] = None


def get_scheduler() -> TaskScheduler:
    """Get the global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = TaskScheduler()
    return _scheduler_instance


async def start_scheduler():
    """Start the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Stop the global scheduler."""
    scheduler = get_scheduler()
    await scheduler.stop()