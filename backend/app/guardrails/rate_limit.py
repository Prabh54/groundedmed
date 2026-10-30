"""Spend protection (ARCHITECTURE §14.7, ADR-021): per-IP hourly limit and a global daily cap.

In-memory, so limits reset when the single API instance restarts; the Anthropic console spend
limit is the hard backstop.

Behind Railway the connecting address is Railway's proxy, so the visitor's address comes from the
`X-Real-IP` header that Railway's edge sets (ADR-039). That header is read only when the direct
peer is inside TRUSTED_PROXY_CIDRS; from anyone else it is ignored, so a visitor cannot pick their
own key.
"""

import ipaddress
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime

CLIENT_IP_HEADER = "x-real-ip"  # Railway: "X-Real-IP for identifying client's remote IP"
IPV6_PREFIX = 64  # one visitor usually controls a whole /64, so it counts as one address

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


def trusted_networks(env_value: str) -> tuple[list[Network], list[str]]:
    """Parse comma-separated CIDRs (TRUSTED_PROXY_CIDRS). Returns (networks, invalid entries);
    invalid entries are skipped, which only means fewer peers are trusted."""
    networks, invalid = [], []
    for entry in (e.strip() for e in env_value.split(",")):
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            invalid.append(entry)
    return networks, invalid


def _parse(value: str | None) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        address = ipaddress.ip_address((value or "").strip())
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        return address.ipv4_mapped
    return address


def _key(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str:
    if isinstance(address, ipaddress.IPv6Address):
        return str(ipaddress.ip_network(f"{address}/{IPV6_PREFIX}", strict=False))
    return str(address)


def client_key(peer: str | None, header_value: str | None, trusted: list[Network]) -> str:
    """The address the per-IP limit counts for one request.

    The header is used only when the direct peer is a trusted proxy and the header holds one valid
    IP; otherwise the peer itself is counted (a bad header from the proxy shares the proxy's count,
    which is stricter, never looser)."""
    peer_address = _parse(peer)
    if peer_address is None:
        return peer or "unknown"
    if any(peer_address in n for n in trusted):
        forwarded = _parse(header_value)
        if forwarded is not None:
            return _key(forwarded)
    return _key(peer_address)


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
