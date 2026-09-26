from typing import Annotated
from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import User
from backend.security import decode_access_token


bearer_scheme = HTTPBearer(auto_error=False)

DatabaseDependency = Annotated[Session, Depends(get_db)]
CredentialsDependency = Annotated[
    HTTPAuthorizationCredentials | None,
    Depends(bearer_scheme),
]


def unauthorized_exception() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user(
    credentials: CredentialsDependency,
    database: DatabaseDependency,
) -> User:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise unauthorized_exception()

    try:
        payload = decode_access_token(credentials.credentials)
        subject = payload.get("sub")
        if not subject:
            raise unauthorized_exception()
        user_id = UUID(subject)
    except (InvalidTokenError, ValueError):
        raise unauthorized_exception()

    user = database.get(User, user_id)
    if user is None:
        raise unauthorized_exception()
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This user account is inactive.",
        )
    return user


CurrentUserDependency = Annotated[User, Depends(get_current_user)]


def require_reviewer(current_user: CurrentUserDependency) -> User:
    if current_user.role not in {"reviewer", "administrator"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Reviewer or administrator access is required.",
        )
    return current_user


ReviewerDependency = Annotated[User, Depends(require_reviewer)]
