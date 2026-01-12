"""
Vizora Web API - FastAPI Application Entry Point

This module sets up the FastAPI application with:
- CORS middleware for React development server
- Static file serving for production frontend builds
- API routes for analysis and health checks
- Lifespan management for startup/shutdown tasks
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from vizora.web.routes import health_router, analysis_router, auth_router, billing_router, schedules_router, google_router
from vizora.web.services.file_manager import file_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup: cleanup old temp files
    deleted = file_manager.cleanup_old_files()
    if deleted:
        print(f"Cleaned up {deleted} old temporary files")

    yield

    # Shutdown: cleanup all temp files
    file_manager.cleanup_old_files()


# Create FastAPI application
app = FastAPI(
    title="Vizora API",
    description="AI-powered data analysis agent API",
    version="1.0.0",
    lifespan=lifespan
)


# CORS configuration
# In development: allow React dev server (localhost:5173)
# In production: restrict to your domain
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Add production origins from environment
if os.getenv("FRONTEND_URL"):
    origins.append(os.getenv("FRONTEND_URL"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routers
app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(auth_router)
app.include_router(billing_router)
app.include_router(schedules_router)
app.include_router(google_router)


# Serve static frontend files in production
# Find project root by looking for pyproject.toml
def find_project_root() -> Path:
    """Find project root by looking for pyproject.toml."""
    current = Path(__file__).resolve().parent
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback to 4 parents up from app.py
    return Path(__file__).resolve().parent.parent.parent.parent

PROJECT_ROOT = find_project_root()
FRONTEND_BUILD_DIR = PROJECT_ROOT / "frontend" / "dist"

# Debug: print the path on startup
print(f"Project root: {PROJECT_ROOT}")
print(f"Frontend build dir: {FRONTEND_BUILD_DIR}")
print(f"Frontend exists: {FRONTEND_BUILD_DIR.exists()}")

if FRONTEND_BUILD_DIR.exists():
    # Serve static assets
    assets_dir = FRONTEND_BUILD_DIR / "assets"
    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="static"
        )

    @app.get("/")
    async def serve_frontend():
        """Serve the React app's index.html."""
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        """
        Catch-all route for React Router.

        Serves index.html for all non-API routes to support
        client-side routing.
        """
        # Check if it's an API route (already handled by routers)
        if path.startswith("api/"):
            return {"error": "Not found"}

        # Check if it's a static file
        file_path = FRONTEND_BUILD_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)

        # Otherwise, serve index.html for React Router
        return FileResponse(FRONTEND_BUILD_DIR / "index.html")

else:
    # No frontend build - show development instructions
    @app.get("/")
    async def dev_instructions():
        """Show instructions when frontend is not built."""
        return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Vizora API</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            text-align: center;
        }
        h1 {
            background: linear-gradient(135deg, #00d4ff, #7b2ff7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        p { color: #a0a0c0; line-height: 1.6; }
        code {
            background: #1a1a2e;
            padding: 2px 8px;
            border-radius: 4px;
            color: #00d4ff;
        }
        .box {
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            text-align: left;
        }
        .box h3 { color: #00d4ff; margin-top: 0; }
        pre {
            background: #0a0a0f;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            color: #00ff88;
        }
        a { color: #00d4ff; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Vizora API</h1>
        <p>The API is running! But the frontend hasn't been built yet.</p>

        <div class="box">
            <h3>For Development</h3>
            <p>Run the Vite dev server in a separate terminal:</p>
            <pre>cd frontend
npm run dev</pre>
            <p>Then open <a href="http://localhost:5173">http://localhost:5173</a></p>
        </div>

        <div class="box">
            <h3>For Production</h3>
            <p>Build the frontend first:</p>
            <pre>cd frontend
npm run build</pre>
            <p>Then refresh this page.</p>
        </div>

        <p>API docs available at <a href="/docs">/docs</a></p>
    </div>
</body>
</html>
        """, status_code=200)


# Development entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "vizora.web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
