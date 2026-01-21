import sys

from agents.orchestrator import run_claim_evolution_agent
from memory.decay import apply_decay
from memory.events import log_event


def _run_decay_job() -> None:
    updated = apply_decay()
    log_event(
        "system",
        "agent_decay_run",
        float(updated),
        f"decay applied to {updated} claims",
        agent_name="claim_evolution",
    )


def main() -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError as exc:
        raise SystemExit("APScheduler is not installed. Run: pip install apscheduler") from exc

    scheduler = BlockingScheduler()
    scheduler.add_job(run_claim_evolution_agent, "interval", minutes=10)
    scheduler.add_job(_run_decay_job, "interval", hours=24)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()


if __name__ == "__main__":
    sys.exit(main())
