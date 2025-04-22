# This file makes the endpoints directory a proper Python package
from app.api.v1.endpoints.courses.course import router as course_router
from app.api.v1.endpoints.courses.module import router as module_router
from app.api.v1.endpoints.courses.unit import router as unit_router
from app.api.v1.endpoints.courses.content import router as content_router
from app.api.v1.endpoints.courses.publish import router as publish_router
from app.api.v1.endpoints.courses.generate import router as generate_router
from app.api.v1.endpoints.courses.public import router as public_router

# Export the routers
router = course_router
module = module_router
unit = unit_router
content = content_router
publish = publish_router
generate = generate_router
public = public_router
