"""The per-IP limit counts the visitor behind the hosting proxy, never a spoofed header.

The API believes the CLIENT_IP_HEADER header only when the connecting peer is in TRUSTED_PROXY_CIDRS
. The tests set the header name to X-Real-IP; any name works the same way (the
production name is measured on the deployed host). Fakes for the pipeline and DB, no network.
"""

import ipaddress
import json

import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.guardrails.rate_limit import (
    RateLimiter,
    client_key,
    pseudonym,
    trusted_networks,
)
from backend.tests.unit.test_api import FakeConn, final_answer

PROXY = ("10.1.2.3", 50000)  # the connecting peer when the hosting proxy forwards a request
OUTSIDER = ("203.0.113.9", 50000)  # someone connecting directly, not through the proxy
TRUSTED = [ipaddress.ip_network("10.0.0.0/8")]
QUESTION = {"question": "Does X help?", "mode": "baseline"}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(main, "connect", lambda: FakeConn())
    monkeypatch.setattr(main, "clients", lambda: (object(), object()))
    monkeypatch.setattr(main, "answer_baseline", lambda q, **kw: final_answer(question=q))
    monkeypatch.setattr(main, "limiter", RateLimiter(per_ip_per_hour=2, daily_cap=100))
    monkeypatch.setattr(main, "TRUSTED_PROXIES", TRUSTED)
    monkeypatch.setattr(main, "CLIENT_IP_HEADER", "x-real-ip")
    monkeypatch.setattr(main, "DEBUG_CLIENT_IP", False)
    return main.app


def codes(client: TestClient, real_ips: list[str | None], path: str = "/query") -> list[int]:
    out = []
    for ip in real_ips:
        headers = {"X-Real-IP": ip} if ip is not None else {}
        out.append(client.post(path, json=QUESTION, headers=headers).status_code)
    return out


def test_header_from_trusted_proxy_limits_each_visitor_separately(app):
    proxy = TestClient(app, client=PROXY)
    assert codes(proxy, ["198.51.100.1"] * 3) == [200, 200, 429]
    # a second visitor through the same proxy has their own allowance
    assert codes(proxy, ["198.51.100.2"] * 3) == [200, 200, 429]


def test_stream_endpoint_uses_the_same_visitor_key(app):
    proxy = TestClient(app, client=PROXY)
    assert codes(proxy, ["198.51.100.1"] * 2, "/query") == [200, 200]
    assert codes(proxy, ["198.51.100.1", "198.51.100.3"], "/query/stream") == [429, 200]


def test_spoofed_header_from_untrusted_source_is_ignored(app):
    outsider = TestClient(app, client=OUTSIDER)
    # a new fake address every time does not buy more queries: the real peer is counted
    assert codes(outsider, ["198.51.100.10", "198.51.100.11", "198.51.100.12"]) == [200, 200, 429]
    # and it cannot use up another visitor's allowance by naming them
    assert codes(TestClient(app, client=PROXY), ["198.51.100.11"]) == [200]


def test_no_trusted_proxies_configured_means_no_header_is_believed(app, monkeypatch):
    monkeypatch.setattr(main, "TRUSTED_PROXIES", [])
    proxy = TestClient(app, client=PROXY)
    assert codes(proxy, ["198.51.100.1", "198.51.100.2", "198.51.100.3"]) == [200, 200, 429]


def test_bad_header_from_trusted_proxy_counts_the_proxy_not_a_free_pass(app):
    proxy = TestClient(app, client=PROXY)
    assert codes(proxy, ["not-an-ip", "1.2.3.4, 5.6.7.8", None]) == [200, 200, 429]


def test_two_real_ip_headers_are_both_ignored(app):
    # if the proxy appended its own copy after a caller's fake one, neither picks the key
    proxy = TestClient(app, client=PROXY)
    real = ("X-Real-IP", "198.51.100.1")
    fakes = [[("X-Real-IP", f"198.51.100.{30 + i}"), real] for i in range(3)]
    out = [proxy.post("/query", json=QUESTION, headers=h).status_code for h in fakes]
    assert out == [200, 200, 429]
    assert codes(proxy, ["198.51.100.1"]) == [200]  # the named visitor's allowance is untouched


@pytest.mark.parametrize(
    "peer,header,expected",
    [
        ("10.1.2.3", "198.51.100.1", "198.51.100.1"),
        ("10.1.2.3", " 198.51.100.1 ", "198.51.100.1"),
        ("203.0.113.9", "198.51.100.1", "203.0.113.9"),  # untrusted peer: header ignored
        ("10.1.2.3", None, "10.1.2.3"),
        ("10.1.2.3", "", "10.1.2.3"),
        ("10.1.2.3", "198.51.100.1, 10.0.0.1", "10.1.2.3"),  # not one address: ignored
        ("10.1.2.3", "::ffff:198.51.100.1", "198.51.100.1"),  # IPv4-mapped = the IPv4 address
        ("10.1.2.3", "2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),  # IPv6 grouped by /64
        ("10.1.2.3", "2001:db8:1:2:bbbb::9", "2001:db8:1:2::/64"),
        ("10.1.2.3", "2001:db8:1:3::1", "2001:db8:1:3::/64"),
        ("2001:db8:9::5", "198.51.100.1", "2001:db8:9::/64"),  # untrusted IPv6 peer
        ("testclient", "198.51.100.1", "testclient"),  # non-IP peer (test client) kept as is
        (None, "198.51.100.1", "unknown"),
    ],
)
def test_client_key(peer, header, expected):
    assert client_key(peer, header, TRUSTED) == expected


def test_trusted_networks_parsing():
    networks, invalid = trusted_networks(" 10.0.0.0/8, ,fd12::/16,100.64.0.1, nonsense,10.0.0.0/33")
    assert [str(n) for n in networks] == ["10.0.0.0/8", "fd12::/16", "100.64.0.1/32"]
    assert invalid == ["nonsense", "10.0.0.0/33"]
    assert trusted_networks("") == ([], [])


def client_lines(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if '"event": "client"' in line]


def test_log_line_shows_proxy_peer_and_header_counts_but_never_an_address(app, capsys):
    proxy = TestClient(app, client=PROXY)
    xff = {"X-Forwarded-For": "192.0.2.50, 192.0.2.51", "True-Client-IP": "192.0.2.52"}
    for ip in ["198.51.100.1", "198.51.100.1", "198.51.100.2"]:
        proxy.post("/query", json=QUESTION, headers={"X-Real-IP": ip, **xff})
    out = capsys.readouterr().out
    assert "198.51.100." not in out and "192.0.2." not in out  # no address is ever written
    lines = [json.loads(line) for line in out.splitlines() if '"event": "client"' in line]
    assert [(x["peer"], x["trusted_peer"], x["ip_headers"], x["xff_entries"]) for x in lines] == [
        ("10.1.2.3", True, {"x-forwarded-for": 1, "x-real-ip": 1, "true-client-ip": 1}, 2)
    ] * 3
    ids = [x["client_id"] for x in lines]
    assert ids[0] == ids[1] != ids[2]  # same visitor -> same pseudonym; another visitor differs
    assert all(len(i) == 10 for i in ids)


def test_log_line_shows_untrusted_peer_and_duplicate_headers(app, capsys):
    outsider = TestClient(app, client=OUTSIDER)
    codes(outsider, [None, "198.51.100.10"])
    dup = [("X-Real-IP", "198.51.100.1"), ("X-Real-IP", "198.51.100.2")]
    TestClient(app, client=PROXY).post("/query", json=QUESTION, headers=dup)
    a, b, c = client_lines(capsys)
    assert (a["trusted_peer"], a["ip_headers"]) == (False, {})
    assert (b["trusted_peer"], b["ip_headers"]) == (False, {"x-real-ip": 1})
    assert a["client_id"] == b["client_id"]  # a forged header did not change the count used
    assert (c["peer"], c["trusted_peer"], c["ip_headers"]) == ("10.1.2.3", True, {"x-real-ip": 2})
    assert c["client_id"] == pseudonym("10.1.2.3")  # duplicates: the proxy's own count


def test_no_header_name_configured_means_no_header_is_believed(app, monkeypatch):
    monkeypatch.setattr(main, "CLIENT_IP_HEADER", "")
    proxy = TestClient(app, client=PROXY)
    assert codes(proxy, ["198.51.100.1", "198.51.100.2", "198.51.100.3"]) == [200, 200, 429]


def test_the_configured_header_name_is_the_one_believed(app, monkeypatch):
    monkeypatch.setattr(main, "CLIENT_IP_HEADER", "true-client-ip")
    proxy = TestClient(app, client=PROXY)
    # X-Real-IP is now just another header: varying it does not buy more queries
    assert codes(proxy, ["198.51.100.1", "198.51.100.2", "198.51.100.3"]) == [200, 200, 429]
    visitor = {"True-Client-IP": "198.51.100.7"}
    out = [proxy.post("/query", json=QUESTION, headers=visitor).status_code for _ in range(3)]
    assert out == [200, 200, 429]


def test_debug_endpoint_is_off_by_default(app):
    assert TestClient(app, client=PROXY).get("/debug/client-ip").status_code == 404


def test_debug_endpoint_shows_the_callers_own_view_and_is_not_counted(app, monkeypatch, capsys):
    monkeypatch.setattr(main, "DEBUG_CLIENT_IP", True)
    proxy = TestClient(app, client=PROXY)
    headers = [("X-Real-IP", "198.51.100.1"), ("X-Forwarded-For", "192.0.2.1, 10.9.9.9")]
    for _ in range(3):
        body = proxy.get("/debug/client-ip", headers=headers).json()
    assert body == {"peer": "10.1.2.3", "trusted_peer": True, "client_ip_header": "x-real-ip",
                    "limit_key": "198.51.100.1",
                    "headers": {"x-forwarded-for": ["192.0.2.1, 10.9.9.9"],
                                "x-real-ip": ["198.51.100.1"]}}  # fmt: skip
    assert client_lines(capsys) == []  # nothing logged
    assert codes(proxy, ["198.51.100.1"] * 3) == [200, 200, 429]  # its allowance is untouched
