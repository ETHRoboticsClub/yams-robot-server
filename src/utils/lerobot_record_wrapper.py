import logging

logger = logging.getLogger(__name__)

_GRACEFUL_STOP_MARKERS = (
    "Failed to read leader action",
    "Leader returned no action",
)


def _is_leader_action_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if any(marker in str(current) for marker in _GRACEFUL_STOP_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def run_with_graceful_stop(run_record) -> int:
    try:
        run_record()
    except RuntimeError as exc:
        if not _is_leader_action_failure(exc):
            raise
        logger.error("Leader action failed. Recording stopped gracefully.")
        return 0
    return 0


def main() -> None:
    from lerobot.scripts.lerobot_record import main as lerobot_record_main

    raise SystemExit(run_with_graceful_stop(lerobot_record_main))


if __name__ == "__main__":
    main()
