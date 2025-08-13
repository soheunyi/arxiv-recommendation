#!/usr/bin/env python3
"""
ArXiv Recommendation System - Server Startup Script

This script manages both frontend and backend servers with:
- Port availability detection
- Existing server checks
- Graceful startup and shutdown
- Health monitoring
"""

import asyncio
import socket
import subprocess
import sys
import time
import signal
import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
import requests
import psutil

console = Console()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger("arxiv-serve")

class ServerManager:
    """Manages frontend and backend server processes"""
    
    def __init__(self):
        self.frontend_process: Optional[subprocess.Popen] = None
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_port = 3000  # Default Vite dev server port
        self.backend_port = 8000   # Default FastAPI port
        self.project_root = Path(__file__).parent
        
        logger.info("Initializing Server Manager...")
        
        # Check if uv is available
        try:
            result = subprocess.run(["uv", "--version"], capture_output=True, check=True, text=True)
            logger.info(f"UV version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[red]✗ UV package manager not found. Please install uv first:[/red]")
            console.print("[blue]curl -LsSf https://astral.sh/uv/install.sh | sh[/blue]")
            sys.exit(1)
        
        # Check Node.js availability
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, check=True, text=True)
            logger.info(f"Node.js version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            console.print("[red]✗ Node.js not found. Please install Node.js first:[/red]")
            console.print("[blue]https://nodejs.org/[/blue]")
            sys.exit(1)
        
    def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port"""
        for port in range(start_port, start_port + max_attempts):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.bind(('localhost', port))
                    return port
            except OSError:
                continue
        raise RuntimeError(f"No available ports found in range {start_port}-{start_port + max_attempts}")
    
    def check_existing_servers(self) -> Tuple[bool, bool]:
        """Check if frontend/backend servers are already running"""
        frontend_running = self.is_port_in_use(self.frontend_port)
        backend_running = self.is_port_in_use(self.backend_port)
        return frontend_running, backend_running
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is currently in use"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def get_process_using_port(self, port: int) -> Optional[dict]:
        """Get information about process using a port"""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    for conn in proc.connections():
                        if conn.laddr.port == port:
                            return {
                                'pid': proc.info['pid'],
                                'name': proc.info['name'],
                                'cmdline': ' '.join(proc.info['cmdline'][:3])
                            }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception:
            pass
        return None
    
    def kill_existing_servers(self) -> None:
        """Kill existing servers if requested by user"""
        frontend_running, backend_running = self.check_existing_servers()
        
        if frontend_running or backend_running:
            table = Table(title="🚨 Existing Servers Detected")
            table.add_column("Service", style="cyan")
            table.add_column("Port", style="yellow")
            table.add_column("Status", style="red")
            table.add_column("Process", style="white")
            
            if frontend_running:
                proc_info = self.get_process_using_port(self.frontend_port)
                proc_desc = f"PID {proc_info['pid']} - {proc_info['name']}" if proc_info else "Unknown"
                table.add_row("Frontend", str(self.frontend_port), "Running", proc_desc)
            
            if backend_running:
                proc_info = self.get_process_using_port(self.backend_port)
                proc_desc = f"PID {proc_info['pid']} - {proc_info['name']}" if proc_info else "Unknown"
                table.add_row("Backend", str(self.backend_port), "Running", proc_desc)
            
            console.print(table)
            
            if console.input("\n[yellow]Kill existing servers and continue? [y/N]: [/yellow]").lower() == 'y':
                if frontend_running:
                    self._kill_process_on_port(self.frontend_port)
                if backend_running:
                    self._kill_process_on_port(self.backend_port)
                
                # Wait for ports to be freed
                console.print("[blue]Waiting for ports to be freed...[/blue]")
                time.sleep(2)
            else:
                console.print("[red]Startup cancelled.[/red]")
                sys.exit(0)
    
    def _kill_process_on_port(self, port: int) -> None:
        """Kill process using specific port"""
        proc_info = self.get_process_using_port(port)
        if proc_info:
            try:
                proc = psutil.Process(proc_info['pid'])
                proc.terminate()
                proc.wait(timeout=5)
                console.print(f"[green]✓ Killed process on port {port}[/green]")
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                try:
                    proc.kill()
                    console.print(f"[yellow]⚠ Force killed process on port {port}[/yellow]")
                except:
                    console.print(f"[red]✗ Failed to kill process on port {port}[/red]")
    
    def start_backend_server(self) -> bool:
        """Start the backend server"""
        backend_dir = self.project_root / "backend"
        
        if not backend_dir.exists():
            console.print("[red]✗ Backend directory not found[/red]")
            return False
        
        # Find available port for backend
        try:
            self.backend_port = self.find_available_port(self.backend_port)
        except RuntimeError as e:
            console.print(f"[red]✗ {e}[/red]")
            return False
        
        # Start backend with uvicorn (assuming FastAPI)
        try:
            # Check if we have a proper backend API server
            api_file = backend_dir / "api.py"
            if not api_file.exists():
                # Create a minimal FastAPI server
                self._create_minimal_api_server()
            
            cmd = [
                "uv", "run", "uvicorn", 
                "backend.api:app", 
                "--host", "0.0.0.0", 
                "--port", str(self.backend_port),
                "--reload"
            ]
            
            self.backend_process = subprocess.Popen(
                cmd,
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to start
            if self._wait_for_server("backend", "localhost", self.backend_port, "/health"):
                console.print(f"[green]✓ Backend server started on port {self.backend_port}[/green]")
                return True
            else:
                console.print("[red]✗ Backend server failed to start[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Backend startup error: {e}[/red]")
            return False
    
    def start_frontend_server(self) -> bool:
        """Start the frontend server"""
        frontend_dir = self.project_root / "frontend"
        
        if not frontend_dir.exists():
            console.print("[red]✗ Frontend directory not found[/red]")
            return False
        
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            logger.info("Frontend dependencies not found, installing...")
            try:
                result = subprocess.run(
                    ["npm", "install"], 
                    cwd=frontend_dir, 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                logger.info("Frontend dependencies installed successfully")
                if result.stdout:
                    logger.debug(f"npm install output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install frontend dependencies: {e}")
                if e.stderr:
                    logger.error(f"npm error: {e.stderr}")
                if e.stdout:
                    logger.error(f"npm output: {e.stdout}")
                return False
        
        # Find available port for frontend
        try:
            self.frontend_port = self.find_available_port(self.frontend_port)
        except RuntimeError as e:
            console.print(f"[red]✗ {e}[/red]")
            return False
        
        # Start frontend dev server (Vite)
        try:
            env = os.environ.copy()
            env["PORT"] = str(self.frontend_port)
            env["VITE_API_URL"] = f"http://localhost:{self.backend_port}"
            
            logger.info(f"Starting Vite dev server on port {self.frontend_port}")
            logger.info(f"Backend API URL: {env['VITE_API_URL']}")
            
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "--port", str(self.frontend_port), "--host", "0.0.0.0"],
                cwd=frontend_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                env=env
            )
            
            # Wait for server to start
            if self._wait_for_server("frontend", "localhost", self.frontend_port):
                console.print(f"[green]✓ Frontend server started on port {self.frontend_port}[/green]")
                return True
            else:
                console.print("[red]✗ Frontend server failed to start[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]✗ Frontend startup error: {e}[/red]")
            return False
    
    def _wait_for_server(self, name: str, host: str, port: int, health_path: str = "") -> bool:
        """Wait for server to become available"""
        max_attempts = 30
        logger.info(f"Waiting for {name} server on {host}:{port}")
        
        for attempt in range(max_attempts):
            try:
                if health_path:
                    response = requests.get(f"http://{host}:{port}{health_path}", timeout=1)
                    if response.status_code == 200:
                        logger.info(f"{name} server health check passed")
                        return True
                else:
                    # Just check if port is responding
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(1)
                        if sock.connect_ex((host, port)) == 0:
                            logger.info(f"{name} server is responding on port {port}")
                            return True
            except Exception as e:
                if attempt == 0:  # Log first attempt error for debugging
                    logger.debug(f"{name} server not ready yet: {e}")
            
            # Check if process is still running
            if name == "frontend" and self.frontend_process:
                if self.frontend_process.poll() is not None:
                    # Process has died, capture output
                    stdout, stderr = self.frontend_process.communicate()
                    logger.error(f"Frontend process died unexpectedly!")
                    if stdout:
                        logger.error(f"Frontend stdout: {stdout}")
                    if stderr:
                        logger.error(f"Frontend stderr: {stderr}")
                    return False
            elif name == "backend" and self.backend_process:
                if self.backend_process.poll() is not None:
                    stdout, stderr = self.backend_process.communicate()
                    logger.error(f"Backend process died unexpectedly!")
                    if stdout:
                        logger.error(f"Backend stdout: {stdout}")
                    if stderr:
                        logger.error(f"Backend stderr: {stderr}")
                    return False
            
            time.sleep(1)
            if attempt % 5 == 0 and attempt > 0:
                logger.info(f"Still waiting for {name} server... ({attempt + 1}/{max_attempts})")
        
        logger.error(f"{name} server failed to start within {max_attempts} seconds")
        return False
    
    def _create_minimal_api_server(self) -> None:
        """Create a minimal FastAPI server for the backend"""
        api_content = '''#!/usr/bin/env python3
"""
Minimal FastAPI server for ArXiv Recommendation System
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import sys
from pathlib import Path

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from arxiv_recommendation.database import DatabaseManager
    from arxiv_recommendation.config import config
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure backend dependencies are installed")

app = FastAPI(
    title="ArXiv Recommendation API",
    description="Backend API for ArXiv Recommendation System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "arxiv-recommendation-api"}

@app.get("/api/config")
async def get_config():
    """Get system configuration"""
    try:
        return {
            "categories": config.arxiv_categories,
            "max_daily_papers": config.max_daily_papers,
            "embedding_model": config.embedding_model
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/papers")
async def get_papers(limit: int = 10):
    """Get recent papers"""
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        papers = await db_manager.get_recent_papers(limit=limit)
        return {"papers": papers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        
        api_file = self.project_root / "backend" / "api.py"
        with open(api_file, 'w') as f:
            f.write(api_content)
        
        console.print("[green]✓ Created minimal API server[/green]")
    
    def show_server_status(self) -> None:
        """Display current server status"""
        table = Table(title="🚀 Server Status")
        table.add_column("Service", style="cyan")
        table.add_column("Port", style="yellow")
        table.add_column("Status", style="green")
        table.add_column("URL", style="blue")
        
        frontend_status = "Running" if self.frontend_process and self.frontend_process.poll() is None else "Stopped"
        backend_status = "Running" if self.backend_process and self.backend_process.poll() is None else "Stopped"
        
        table.add_row(
            "Frontend (React)", 
            str(self.frontend_port), 
            frontend_status,
            f"http://localhost:{self.frontend_port}"
        )
        table.add_row(
            "Backend (API)", 
            str(self.backend_port), 
            backend_status,
            f"http://localhost:{self.backend_port}"
        )
        
        console.print(table)
        
        if frontend_status == "Running" and backend_status == "Running":
            console.print(Panel(
                f"[green]🎉 Both servers are running![/green]\n\n"
                f"• Frontend: [link=http://localhost:{self.frontend_port}]http://localhost:{self.frontend_port}[/link]\n"
                f"• Backend API: [link=http://localhost:{self.backend_port}]http://localhost:{self.backend_port}[/link]\n"
                f"• API Health: [link=http://localhost:{self.backend_port}/health]http://localhost:{self.backend_port}/health[/link]",
                title="✨ Success",
                border_style="green"
            ))
    
    def stop_servers(self) -> None:
        """Stop both servers gracefully"""
        console.print("\n[yellow]Stopping servers...[/yellow]")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
                console.print("[green]✓ Frontend server stopped[/green]")
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
                console.print("[yellow]⚠ Frontend server force stopped[/yellow]")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
                console.print("[green]✓ Backend server stopped[/green]")
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
                console.print("[yellow]⚠ Backend server force stopped[/yellow]")
    
    def debug_frontend_startup(self) -> None:
        """Debug frontend startup issues"""
        frontend_dir = self.project_root / "frontend"
        logger.info("=== Frontend Debug Information ===")
        
        # Check directory structure
        if not frontend_dir.exists():
            logger.error("Frontend directory does not exist!")
            return
        
        # Check package.json
        package_json = frontend_dir / "package.json"
        if package_json.exists():
            logger.info("✓ package.json found")
            try:
                import json
                with open(package_json) as f:
                    pkg = json.load(f)
                    if "scripts" in pkg and "dev" in pkg["scripts"]:
                        logger.info(f"✓ dev script: {pkg['scripts']['dev']}")
                    else:
                        logger.warning("⚠ No 'dev' script found in package.json")
            except Exception as e:
                logger.error(f"✗ Error reading package.json: {e}")
        else:
            logger.error("✗ package.json not found!")
        
        # Check node_modules
        node_modules = frontend_dir / "node_modules"
        if node_modules.exists():
            logger.info("✓ node_modules directory exists")
        else:
            logger.warning("⚠ node_modules directory missing")
        
        # Check vite.config.ts
        vite_config = frontend_dir / "vite.config.ts"
        if vite_config.exists():
            logger.info("✓ vite.config.ts found")
        else:
            logger.warning("⚠ vite.config.ts not found")
        
        # Try running npm run dev directly for debugging
        logger.info("Attempting to run 'npm run dev' for debugging...")
        try:
            result = subprocess.run(
                ["npm", "run", "dev", "--", "--version"],
                cwd=frontend_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                logger.info("✓ npm run dev command is available")
            else:
                logger.error(f"✗ npm run dev failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            logger.info("⚠ npm run dev command started (timed out as expected)")
        except Exception as e:
            logger.error(f"✗ Error running npm run dev: {e}")

    def monitor_servers(self) -> None:
        """Monitor running servers and handle shutdown"""
        try:
            console.print("\n[blue]Press Ctrl+C to stop servers[/blue]")
            console.print("[dim]Monitoring server processes...[/dim]")
            
            while True:
                time.sleep(2)
                
                # Check if processes are still running
                if self.frontend_process and self.frontend_process.poll() is not None:
                    logger.error("Frontend server stopped unexpectedly")
                    # Capture any remaining output
                    try:
                        stdout, stderr = self.frontend_process.communicate(timeout=1)
                        if stdout:
                            logger.error(f"Frontend final output: {stdout}")
                    except:
                        pass
                    break
                
                if self.backend_process and self.backend_process.poll() is not None:
                    logger.error("Backend server stopped unexpectedly")
                    try:
                        stdout, stderr = self.backend_process.communicate(timeout=1)
                        if stdout:
                            logger.error(f"Backend final output: {stdout}")
                    except:
                        pass
                    break
                    
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        finally:
            self.stop_servers()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ArXiv Recommendation System Server Manager")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode with verbose logging")
    parser.add_argument("--debug-frontend", action="store_true", help="Debug frontend startup only")
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger("arxiv-serve").setLevel(logging.DEBUG)
        logger.info("Debug mode enabled")
    
    console.print(Panel(
        "[bold blue]ArXiv Recommendation System[/bold blue]\n"
        "[white]Server Startup Manager[/white]",
        title="🚀 Starting Servers",
        border_style="blue"
    ))
    
    manager = ServerManager()
    
    # If debug-frontend only mode, run diagnostics and exit
    if args.debug_frontend:
        manager.debug_frontend_startup()
        return
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        console.print(f"\n[yellow]Received signal {signum}, shutting down...[/yellow]")
        manager.stop_servers()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Check for existing servers
        manager.kill_existing_servers()
        
        # Start servers with progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            # Start backend
            backend_task = progress.add_task("[blue]Starting backend server...", total=None)
            if not manager.start_backend_server():
                console.print("[red]Failed to start backend server[/red]")
                sys.exit(1)
            progress.update(backend_task, description="[green]✓ Backend server started")
            
            # Start frontend
            frontend_task = progress.add_task("[blue]Starting frontend server...", total=None)
            if not manager.start_frontend_server():
                console.print("[red]Failed to start frontend server[/red]")
                manager.stop_servers()
                sys.exit(1)
            progress.update(frontend_task, description="[green]✓ Frontend server started")
        
        # Show status and monitor
        manager.show_server_status()
        manager.monitor_servers()
        
    except Exception as e:
        console.print(f"[red]Startup error: {e}[/red]")
        manager.stop_servers()
        sys.exit(1)


if __name__ == "__main__":
    main()