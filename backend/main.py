from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text

from backend.config import get_settings
from backend.dependencies import CurrentUserDependency, DatabaseDependency
from backend.models import User
from backend.routers.agents import router as agents_router
from backend.routers.approvals import router as approvals_router
from backend.routers.dashboard import router as dashboard_router
from backend.routers.executions import router as executions_router
from backend.routers.knowledge import router as knowledge_router
from backend.routers.missions import router as missions_router
from backend.routers.shadowbench import router as shadowbench_router
from backend.routers.system import router as system_router
from backend.routers.uploads import router as uploads_router
from backend.schemas import LoginRequest, LoginResponse, MessageResponse, UserResponse
from backend.security import create_access_token, verify_password
from backend.services.workflow_service import get_workflow_runtime


settings = get_settings()
logger = logging.getLogger("cortex.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.upload_path.mkdir(parents=True, exist_ok=True)
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    yield
    if get_workflow_runtime.cache_info().currsize:
        get_workflow_runtime().close()


app = FastAPI(
    title=settings.app_name,
    description="CORTEX Enterprise AI Mission Control API",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    started_at = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure", extra={"request_id": request_id})
        return JSONResponse(
            status_code=500,
            content={
                "detail": "CORTEX could not complete the request.",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )
    elapsed_ms = int((perf_counter() - started_at) * 1000)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = str(elapsed_ms)
    return response

app.include_router(agents_router)
app.include_router(missions_router)
app.include_router(executions_router)
app.include_router(approvals_router)
app.include_router(knowledge_router)
app.include_router(dashboard_router)
app.include_router(shadowbench_router)
app.include_router(system_router)
app.include_router(uploads_router)

@app.get("/", response_model=MessageResponse)
def application_root() -> MessageResponse:
    return MessageResponse(
        message="CORTEX Enterprise AI Mission Control API is running."
    )


@app.get("/api/v1/health/live")
def health_live() -> dict[str, str]:
    return {"status": "healthy", "application": settings.app_name}


@app.get("/api/v1/health/ready")
def health_ready(database: DatabaseDependency) -> dict[str, str]:
    database.execute(text("SELECT 1"))
    return {"status": "ready", "database": "connected"}


@app.post("/api/v1/auth/login", response_model=LoginResponse)
def login(request: LoginRequest, database: DatabaseDependency) -> LoginResponse:
    normalized_email = str(request.email).strip().lower()
    user = database.scalar(
        select(User).where(func.lower(User.email) == normalized_email)
    )
    if user is None or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account is inactive.",
        )

    user.last_login_at = datetime.now(timezone.utc)
    database.commit()
    database.refresh(user)
    access_token, expires_in = create_access_token(user_id=user.id, role=user.role)
    return LoginResponse(
        access_token=access_token,
        expires_in=expires_in,
        user=UserResponse.model_validate(user),
    )


@app.get("/api/v1/auth/me", response_model=UserResponse)
def authenticated_user(current_user: CurrentUserDependency) -> UserResponse:
    return UserResponse.model_validate(current_user)
