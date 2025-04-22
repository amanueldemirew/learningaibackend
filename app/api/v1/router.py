from fastapi import APIRouter
from app.api.v1.endpoints import courses, auth, user

api_router = APIRouter()

# Include all routers
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(user.router, prefix="/users", tags=["users"])
api_router.include_router(courses.router, prefix="/courses", tags=["courses"])
api_router.include_router(courses.module_router, tags=["modules"])
api_router.include_router(courses.unit_router, tags=["units"])
api_router.include_router(courses.content_router, tags=["contents"])
api_router.include_router(courses.publish_router, tags=["publishing"])
api_router.include_router(courses.generate_router, tags=["generate"])
api_router.include_router(courses.public_router, prefix="/courses", tags=["public"])
