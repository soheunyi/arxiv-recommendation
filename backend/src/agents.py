"""Multi-agent system for ArXiv recommendation."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from rich.console import Console
from rich.logging import RichHandler

from config import config
from arxiv_client import ArXivClient
from embeddings import EmbeddingManager
from recommendations import RecommendationEngine
from database import DatabaseManager

# Set up logging with Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


@dataclass
class AgentMessage:
    """Message structure for agent communication."""

    sender: str
    recipient: str
    content: Any
    message_type: str
    timestamp: datetime


class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        self.name = name
        self.messages: List[AgentMessage] = []
        self.logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process an incoming message and return a response."""
        pass

    def send_message(
        self, recipient: str, content: Any, message_type: str = "data"
    ) -> AgentMessage:
        """Create and send a message to another agent."""
        message = AgentMessage(
            sender=self.name,
            recipient=recipient,
            content=content,
            message_type=message_type,
            timestamp=datetime.now(),
        )
        return message

    def log_activity(self, activity: str, details: Optional[str] = None):
        """Log agent activity with rich formatting."""
        message = f"[bold cyan]{self.name}[/bold cyan]: {activity}"
        if details:
            message += f" - {details}"
        console.print(message)


class DataAgent(BaseAgent):
    """Agent responsible for collecting and preprocessing arXiv papers."""

    def __init__(self):
        super().__init__("DataAgent")
        self.arxiv_client = ArXivClient()
        self.db_manager = DatabaseManager()

    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process data collection requests."""
        if message.message_type == "collect_papers":
            return await self._collect_daily_papers()
        elif message.message_type == "get_new_papers":
            return await self._get_new_papers()

        return None

    async def _collect_daily_papers(self) -> AgentMessage:
        """Collect new papers from arXiv for configured categories."""
        self.log_activity(
            "Starting daily paper collection",
            f"Categories: {', '.join(config.arxiv_categories)}",
        )

        all_papers = []

        for category in config.arxiv_categories:
            try:
                papers = await self.arxiv_client.fetch_recent_papers(
                    category=category,
                    max_results=config.max_daily_papers // len(config.arxiv_categories),
                )
                all_papers.extend(papers)
                self.log_activity(f"Collected {len(papers)} papers from {category}")

            except Exception as e:
                self.logger.error(f"Failed to collect papers from {category}: {e}")

        # Store papers in database
        stored_count = await self.db_manager.store_papers(all_papers)
        self.log_activity("Papers stored in database", f"Stored: {stored_count}")

        return self.send_message(
            recipient="RecommendationAgent",
            content={"papers": all_papers, "stored_count": stored_count},
            message_type="papers_collected",
        )

    async def _get_new_papers(self) -> AgentMessage:
        """Get new papers that haven't been processed for recommendations yet."""
        unprocessed_papers = await self.db_manager.get_unprocessed_papers()

        return self.send_message(
            recipient="RecommendationAgent",
            content={"papers": unprocessed_papers},
            message_type="new_papers_available",
        )


class RecommendationAgent(BaseAgent):
    """Agent responsible for generating personalized recommendations."""

    def __init__(self):
        super().__init__("RecommendationAgent")
        self.embedding_manager = EmbeddingManager()
        self.recommendation_engine = RecommendationEngine(
            embedding_manager=self.embedding_manager
        )
        self.db_manager = DatabaseManager()

    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process recommendation generation requests."""
        if message.message_type == "papers_collected":
            return await self._generate_recommendations(message.content["papers"])
        elif message.message_type == "update_user_preferences":
            return await self._update_user_preferences(message.content)

        return None

    async def _generate_recommendations(self, papers: List[Dict]) -> AgentMessage:
        """Generate recommendations for new papers."""
        self.log_activity(
            "Generating embeddings for new papers", f"Processing {len(papers)} papers"
        )

        # Generate embeddings for new papers
        paper_embeddings = []
        for paper in papers:
            try:
                # Handle both PaperMetadata objects and dictionaries
                if hasattr(paper, "abstract"):
                    # It's a PaperMetadata object
                    abstract = paper.abstract
                    paper_id = paper.id
                else:
                    # It's a dictionary
                    abstract = paper["abstract"]
                    paper_id = paper["id"]

                embedding = await self.embedding_manager.get_embedding(
                    text=abstract, paper_id=paper_id
                )
                paper_embeddings.append({"paper_id": paper_id, "embedding": embedding})
            except Exception as e:
                paper_id = (
                    paper.id if hasattr(paper, "id") else paper.get("id", "unknown")
                )
                self.logger.error(f"Failed to generate embedding for {paper_id}: {e}")

        self.log_activity("Embeddings generated", f"Count: {len(paper_embeddings)}")

        # Generate recommendations based on user preferences
        user_preferences = await self.db_manager.get_user_preferences()
        recommendations = await self.recommendation_engine.generate_recommendations(
            paper_embeddings=paper_embeddings,
            user_preferences=user_preferences,
            top_k=10,
        )

        self.log_activity("Recommendations generated", f"Count: {len(recommendations)}")

        # Routine: bulk score all cached papers (updates current_score)
        try:
            scored_count = await self.recommendation_engine.score_all_cached_papers(
                user_preferences=user_preferences
            )
            self.log_activity("Bulk scoring complete", f"Scored: {scored_count}")
        except Exception as e:
            self.logger.error(f"Bulk scoring failed: {e}")

        return self.send_message(
            recipient="Coordinator",
            content={
                "recommendations": recommendations,
                "paper_count": len(papers),
                "embedding_count": len(paper_embeddings),
            },
            message_type="recommendations_ready",
        )

    async def _update_user_preferences(self, preference_data: Dict) -> AgentMessage:
        """Update user preferences based on ratings and feedback."""
        self.log_activity(
            "Updating user preferences",
            f"Ratings: {len(preference_data.get('ratings', []))}",
        )

        # Update user preferences in database
        await self.db_manager.update_user_preferences(preference_data)

        return self.send_message(
            recipient="Coordinator",
            content={"status": "preferences_updated"},
            message_type="preferences_updated",
        )


class Coordinator(BaseAgent):
    """Coordinator agent that manages workflow between other agents."""

    def __init__(self):
        super().__init__("Coordinator")
        self.agents = {
            "DataAgent": DataAgent(),
            "RecommendationAgent": RecommendationAgent(),
        }

    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Process coordination requests."""
        if message.message_type == "start_daily_workflow":
            return await self._run_daily_workflow()
        elif message.message_type == "process_user_feedback":
            return await self._process_user_feedback(message.content)

        return None

    async def _run_daily_workflow(self) -> AgentMessage:
        """Orchestrate the daily paper collection and recommendation workflow."""
        self.log_activity("Starting daily recommendation workflow")

        try:
            # Step 1: Collect new papers
            collect_message = AgentMessage(
                sender="Coordinator",
                recipient="DataAgent",
                content={},
                message_type="collect_papers",
                timestamp=datetime.now(),
            )

            papers_result = await self.agents["DataAgent"].process(collect_message)

            if papers_result and papers_result.message_type == "papers_collected":
                # Step 2: Generate recommendations
                recommendations_result = await self.agents[
                    "RecommendationAgent"
                ].process(papers_result)

                if (
                    recommendations_result
                    and recommendations_result.message_type == "recommendations_ready"
                ):
                    self.log_activity(
                        "Daily workflow completed successfully",
                        f"Recommendations: {len(recommendations_result.content['recommendations'])}",
                    )

                    return self.send_message(
                        recipient="user",
                        content=recommendations_result.content,
                        message_type="daily_workflow_complete",
                    )

            raise Exception("Workflow failed at recommendation generation stage")

        except Exception as e:
            self.logger.error(f"Daily workflow failed: {e}")
            return self.send_message(
                recipient="user",
                content={"error": str(e)},
                message_type="workflow_error",
            )

    async def _process_user_feedback(self, feedback_data: Dict) -> AgentMessage:
        """Process user feedback and update preferences."""
        self.log_activity("Processing user feedback")

        update_message = AgentMessage(
            sender="Coordinator",
            recipient="RecommendationAgent",
            content=feedback_data,
            message_type="update_user_preferences",
            timestamp=datetime.now(),
        )

        result = await self.agents["RecommendationAgent"].process(update_message)

        return self.send_message(
            recipient="user",
            content={"status": "feedback_processed"},
            message_type="feedback_processed",
        )


class MultiAgentSystem:
    """Main system that orchestrates all agents."""

    def __init__(self):
        self.coordinator = Coordinator()
        self.logger = logging.getLogger("MultiAgentSystem")

    async def run_daily_workflow(self) -> Dict[str, Any]:
        """Run the complete daily recommendation workflow."""
        console.print("[bold green]Starting ArXiv Recommendation System[/bold green]")

        workflow_message = AgentMessage(
            sender="system",
            recipient="Coordinator",
            content={},
            message_type="start_daily_workflow",
            timestamp=datetime.now(),
        )

        result = await self.coordinator.process(workflow_message)

        if result and result.message_type == "daily_workflow_complete":
            console.print(
                "[bold green]Daily workflow completed successfully![/bold green]"
            )
            return result.content
        else:
            console.print("[bold red]Daily workflow failed![/bold red]")
            return {"error": "Workflow failed"}

    async def process_user_feedback(self, ratings: List[Dict]) -> Dict[str, Any]:
        """Process user ratings and feedback."""
        feedback_message = AgentMessage(
            sender="system",
            recipient="Coordinator",
            content={"ratings": ratings},
            message_type="process_user_feedback",
            timestamp=datetime.now(),
        )

        result = await self.coordinator.process(feedback_message)
        return result.content if result else {"error": "Failed to process feedback"}


# Convenience function for external use
async def run_recommendation_system() -> Dict[str, Any]:
    """Run the complete recommendation system workflow."""
    system = MultiAgentSystem()
    return await system.run_daily_workflow()
