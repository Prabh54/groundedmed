"""Spend protection (ARCHITECTURE §14.7, ADR-021): per-IP hourly limit and a global daily cap.

In-memory, so limits reset when the single API instance restarts; the Anthropic console spend
limit is the hard backstop.
"""

import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime


class RateLimiter:
    def __init__(self, per_ip_per_hour: int, daily_cap: int, clock=time.time):
        self.per_ip_per_hour, self.daily_cap, self.clock = per_ip_per_hour, daily_cap, clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day: str | None = None
        self._day_count = 0
        self._lock = threading.Lock()

    def check(self, ip: str) -> str | None:
        """Record one query for `ip` and return None, or return why it is refused (not recorded)."""
        now = self.clock()
        day = datetime.fromtimestamp(now, UTC).date().isoformat()
        with self._lock:
            if day != self._day:
                self._day, self._day_count = day, 0
                # Forget IPs idle for over an hour, so memory stays bounded.
                for idle in [k for k, v in self._hits.items() if not v or v[-1] <= now - 3600]:
                    del self._hits[idle]
            if self._day_count >= self.daily_cap:
                return "daily_cap"
            hits = self._hits[ip]
            while hits and hits[0] <= now - 3600:
                hits.popleft()
            if len(hits) >= self.per_ip_per_hour:
                return "per_ip"
            hits.append(now)
            self._day_count += 1
            return None
