from __future__ import annotations

import time
from pathlib import Path


def main() -> None:
    heartbeat = Path("/tmp/openmatcher-worker-heartbeat")
    while True:
        heartbeat.write_text(str(time.time()), encoding="utf-8")
        time.sleep(10)


if __name__ == "__main__":
    main()
