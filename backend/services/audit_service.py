from __future__ import annotations

from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from backend.models import AuditLog


class AuditService:
    @staticmethod
    def write(
        database: Session,
        *,
        action: str,
        actor_user_id: UUID | None = None,
        mission_id: UUID | None = None,
        task_id: UUID | None = None,
        entity_type: str | None = None,
        entity_id: str | None = None,
        details: dict[str, Any] | None = None,
        ip_address: str | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            actor_user_id=actor_user_id,
            mission_id=mission_id,
            task_id=task_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            details=details or {},
            ip_address=ip_address,
        )
        database.add(entry)
        return entry
