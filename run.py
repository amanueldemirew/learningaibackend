import uvicorn
from app.core.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.RELOAD,
        debug=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1,
    )
