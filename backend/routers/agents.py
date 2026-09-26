from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.dependencies import get_current_user
from backend.models import Agent, User
from backend.schemas import AgentResponse


router = APIRouter(
    prefix="/api/v1/agents",
    tags=["Agents"],
)


@router.get("", response_model=list[AgentResponse])
def list_agents(
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Agent]:
    statement = select(Agent).order_by(Agent.name)

    return list(
        database.scalars(statement).all()
    )


@router.get(
    "/{agent_id}",
    response_model=AgentResponse,
)
def get_agent(
    agent_id: UUID,
    database: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Agent:
    agent = database.get(Agent, agent_id)

    if agent is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found.",
        )

    return agent