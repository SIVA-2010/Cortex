from __future__ import annotations

import argparse
import time

from sqlalchemy import select

from backend.database import SessionLocal
from backend.models import Mission, User, WorkflowApproval, WorkflowRun
from backend.services.workflow_persistence import WorkflowPersistenceService
from backend.services.workflow_service import WorkflowService


OBJECTIVE = (
    "Analyse the synthetic customer complaints, identify recurring issues and "
    "evidence-backed root causes, verify material claims, apply privacy and "
    "governance controls, and produce prioritised executive recommendations."
)


def get_or_create_mission() -> tuple[Mission, User]:
    with SessionLocal() as database:
        user = database.scalar(
            select(User).where(User.email == "reviewer@cortex.com")
        )
        if user is None:
            raise SystemExit("Run `python -m backend.init_db` first.")

        mission = database.scalar(
            select(Mission)
            .where(Mission.title == "CORTEX Full Workflow Test")
            .order_by(Mission.created_at.desc())
        )
        if mission is None:
            mission = Mission(
                title="CORTEX Full Workflow Test",
                objective=OBJECTIVE,
                business_domain="Customer Complaint Intelligence",
                priority="high",
                risk_tolerance="balanced",
                max_budget=25.0,
                output_format="executive_report",
                human_approval_preference=True,
                status="draft",
                plan_source="pending",
                created_by_id=user.id,
            )
            database.add(mission)
            database.commit()
            database.refresh(mission)
        database.expunge(mission)
        database.expunge(user)
        return mission, user


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-approve", action="store_true")
    args = parser.parse_args()

    mission, user = get_or_create_mission()
    service = WorkflowService()

    with SessionLocal() as database:
        mission_db = database.get(Mission, mission.id)
        run = service.create_run(
            database,
            mission=mission_db,
            actor_user_id=user.id,
            use_default_data=True,
        )
        database.commit()
        run_id = run.id

    print("Workflow run:", run_id)
    service.execute_run(run_id)

    with SessionLocal() as database:
        run = database.get(WorkflowRun, run_id)
        print("Status      :", run.status)
        print("Stage       :", run.current_stage)
        snapshot = run.state_snapshot or {}
        print("Progress    :", snapshot.get("progress_percent"))
        print("Tokens      :", snapshot.get("total_tokens"))

        if run.status == "awaiting_approval" and args.auto_approve:
            approval = database.scalar(
                select(WorkflowApproval).where(
                    WorkflowApproval.workflow_run_id == run_id
                )
            )
            if approval is None:
                raise SystemExit("Approval record was not created.")
            WorkflowPersistenceService.resolve_approval(
                database,
                approval_id=approval.id,
                reviewer_id=user.id,
                action="approve",
                comment="Automated integration test approval.",
            )
            database.commit()
            print("Auto-approving:", approval.id)
            service.resume_run(
                workflow_run_id=run_id,
                action="approve",
                comment="Automated integration test approval.",
            )

    with SessionLocal() as database:
        final_run = database.get(WorkflowRun, run_id)
        final_state = final_run.state_snapshot or {}
        print("Final status :", final_run.status)
        print("Final stage  :", final_run.current_stage)
        report = final_state.get("final_report", {}).get("output", {}).get("report")
        if final_run.status == "completed" and report:
            print("Mission score:", report.get("mission_success_score"))
            print("Next action  :", report.get("next_recommended_action"))
        elif final_run.status == "awaiting_approval":
            print("Run again with --auto-approve to complete the approval path.")
        else:
            print("Last error   :", final_state.get("last_error"))


if __name__ == "__main__":
    main()
