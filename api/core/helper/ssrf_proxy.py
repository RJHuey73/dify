"""SSRF-protected HTTP client for generic outbound requests.

Use this module when the URL represents a normal external HTTP interaction that
must go through network/proxy policy exactly as requested, such as HTTP Request
nodes, provider/API integrations, auth discovery, or custom tool calls.

Do not use this directly for "remote file" retrieval. File downloads, probes,
and metadata checks should use `core.file.remote_fetcher` instead so Dify-signed
file URLs can be resolved through DB + storage before falling back to this SSRF
client.

SSRF protection here is two independent, additive layers:

1. An optional external forward proxy (Squid, etc. -- `SSRF_PROXY_ALL_URL` /
   `SSRF_PROXY_HTTP_URL` + `SSRF_PROXY_HTTPS_URL`). When configured, the proxy
   owns resolving and connecting to the real target; this module only detects
   a Squid-style rejection afterwards via the 401/403 + `Server`/`Via` header
   heuristic in `make_request`. This path's behavior is unchanged by layer 2
   below -- it is expected to enforce its own network ACLs.
2. An in-process address guard (`_resolve_and_pin_host`), applied only when no
   proxy is configured per (1). It resolves the target host exactly once via
   `socket.getaddrinfo`, rejects the result if it falls in a private,
   loopback, link-local, reserved, or multicast range (`ipaddress`; see
   `_is_blocked_address`) unless explicitly exempted by `SSRF_ALLOWED_HOSTS`,
   and then pins the actual connection to the exact address it just validated
   by substituting it directly into the request URL (Host header and TLS SNI
   are forced back to the original hostname via the `sni_hostname` extension
   so virtual hosting and certificate validation still work). Because no
   second, independent DNS lookup ever happens for that request, an attacker
   controlling the target's DNS cannot "rebind" a public answer at check-time
   into a private one at connect-time. This layer exists because layer 1 is
   optional and, when unconfigured (the common case outside the reference
   docker-compose), previously left every caller of this module with *no*
   in-process restriction on the target address at all.

   This layer is process-local per request (no global/monkeypatched state) and
   is skipped entirely whenever a forward proxy is configured, since in that
   mode the real TCP connection targets the proxy, not `url`'s host -- pinning
   would be meaningless there and could conflict with the proxy's own
   hostname-based routing/ACLs.
"""

import ipaddress
import logging
import socket
import time
from typing import Any, NamedTuple

import httpx
from pydantic import TypeAdapter, ValidationError

from configs import dify_config
from core.helper.http_client_pooling import get_pooled_http_client
from core.tools.errors import ToolSSRFError
from graphon.http.response import HttpResponse

logger = logging.getLogger(__name__)

SSRF_DEFAULT_MAX_RETRIES = dify_config.SSRF_DEFAULT_MAX_RETRIES

BACKOFF_FACTOR = 0.5
STATUS_FORCELIST = [429, 500, 502, 503, 504]

type Headers = dict[str, str]
_HEADERS_ADAPTER: TypeAdapter[Headers] = TypeAdapter(Headers)

_SSL_VERIFIED_POOL_KEY = "ssrf:verified"
_SSL_UNVERIFIED_POOL_KEY = "ssrf:unverified"
_SSRF_CLIENT_LIMITS = httpx.Limits(
    max_connections=dify_config.SSRF_POOL_MAX_CONNECTIONS,
    max_keepalive_connections=dify_config.SSRF_POOL_MAX_KEEPALIVE_CONNECTIONS,
    keepalive_expiry=dify_config.SSRF_POOL_KEEPALIVE_EXPIRY,
)


class MaxRetriesExceededError(ValueError):
    """Raised when the maximum number of retries is exceeded."""

    pass


request_error = httpx.RequestError
max_retries_exceeded_error = MaxRetriesExceededError


def _create_proxy_mounts(verify: bool) -> dict[str, httpx.HTTPTransport]:
    """Build per-scheme proxy transports with the same TLS policy as the SSRF client."""
    return {
        "http://": httpx.HTTPTransport(
            proxy=dify_config.SSRF_PROXY_HTTP_URL,
            verify=verify,
        ),
        "https://": httpx.HTTPTransport(
            proxy=dify_config.SSRF_PROXY_HTTPS_URL,
            verify=verify,
        ),
    }


def _build_ssrf_client(verify: bool) -> httpx.Client:
    if dify_config.SSRF_PROXY_ALL_URL:
        return httpx.Client(
            proxy=dify_config.SSRF_PROXY_ALL_URL,
            verify=verify,
            limits=_SSRF_CLIENT_LIMITS,
        )

    if dify_config.SSRF_PROXY_HTTP_URL and dify_config.SSRF_PROXY_HTTPS_URL:
        return httpx.Client(
            mounts=_create_proxy_mounts(verify=verify),
            verify=verify,
            limits=_SSRF_CLIENT_LIMITS,
        )

    return httpx.Client(verify=verify, limits=_SSRF_CLIENT_LIMITS)


def _get_ssrf_client(ssl_verify_enabled: bool) -> httpx.Client:
    if not isinstance(ssl_verify_enabled, bool):
        raise ValueError("SSRF client verify flag must be a boolean")

    return get_pooled_http_client(
        _SSL_VERIFIED_POOL_KEY if ssl_verify_enabled else _SSL_UNVERIFIED_POOL_KEY,
        lambda: _build_ssrf_client(verify=ssl_verify_enabled),
    )


def _get_user_provided_host_header(headers: Headers | None) -> str | None:
    """
    Extract the user-provided Host header from the headers dict.

    This is needed because when using a forward proxy, httpx may override the Host header.
    We preserve the user's explicit Host header to support virtual hosting and other use cases.
    """
    if not headers:
        return None
    # Case-insensitive lookup for Host header
    for key, value in headers.items():
        if key.lower() == "host":
            return value
    return None


def _inject_trace_headers(headers: Headers | None) -> Headers:
    """
    Inject W3C traceparent header for distributed tracing.

    When OTEL is enabled, HTTPXClientInstrumentor handles trace propagation automatically.
    When OTEL is disabled, we manually inject the traceparent header.
    """
    if headers is None:
        headers = {}

    # Skip if already present (case-insensitive check)
    for key in headers:
        if key.lower() == "traceparent":
            return headers

    # Skip if OTEL is enabled - HTTPXClientInstrumentor handles this automatically
    if dify_config.ENABLE_OTEL:
        return headers

    # Generate and inject traceparent for non-OTEL scenarios
    try:
        from core.helper.trace_id_helper import generate_traceparent_header

        traceparent = generate_traceparent_header()
        if traceparent:
            headers["traceparent"] = traceparent
    except Exception:
        # Silently ignore errors to avoid breaking requests
        logger.debug("Failed to generate traceparent header", exc_info=True)

    return headers


def _is_ssrf_proxy_configured() -> bool:
    """Return whether an external forward proxy (Squid, etc.) is configured for this client.

    Mirrors the condition `_build_ssrf_client` uses to decide between a proxied and a direct
    client -- keep the two in sync. When this is true, the in-process address guard below is
    skipped entirely (see module docstring, layer 2).
    """
    return bool(
        dify_config.SSRF_PROXY_ALL_URL or (dify_config.SSRF_PROXY_HTTP_URL and dify_config.SSRF_PROXY_HTTPS_URL)
    )


def _parse_ssrf_allowlist(
    raw: str | None,
) -> tuple[frozenset[str], tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]]:
    """Split `SSRF_ALLOWED_HOSTS` into literal hostnames and IP/CIDR networks.

    Each comma-separated entry is tried as an IP network first (a bare address parses as a
    /32 or /128); anything that fails to parse as one is kept as a literal, lowercased
    hostname for exact (case-insensitive) matching instead.
    """
    hostnames: set[str] = set()
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            hostnames.add(entry.lower())
    return frozenset(hostnames), tuple(networks)


def _is_blocked_address(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True when `ip` falls in a private/internal-use range that a workflow-triggered
    outbound request should never be able to reach by default (RFC 1918, loopback,
    link-local -- including the 169.254.0.0/16 cloud-metadata range -- reserved, multicast,
    and unspecified addresses)."""
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified


def _is_ssrf_allowlisted(host: str, ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None) -> bool:
    """True when `host` (or its resolved `ip`) is explicitly exempted via `SSRF_ALLOWED_HOSTS`."""
    hostnames, networks = _parse_ssrf_allowlist(dify_config.SSRF_ALLOWED_HOSTS)
    if host.lower() in hostnames:
        return True
    return ip is not None and any(ip in network for network in networks)


def _validate_ssrf_address(host: str, ip: ipaddress.IPv4Address | ipaddress.IPv6Address, url: str) -> None:
    if _is_blocked_address(ip) and not _is_ssrf_allowlisted(host, ip):
        raise ToolSSRFError(
            f"Access to '{url}' was blocked by SSRF protection: host '{host}' resolves to "
            f"'{ip}', which is a private, loopback, link-local, or otherwise reserved "
            f"address. Set SSRF_ALLOWED_HOSTS to explicitly exempt trusted internal targets."
        )


class _PinnedTarget(NamedTuple):
    """Result of `_resolve_and_pin_host`: the (possibly rewritten) request URL, plus the
    original Host-header/SNI values to force when a rewrite happened."""

    url: httpx.URL
    host_header: str | None
    sni_hostname: str | None


def _resolve_and_pin_host(url: str) -> _PinnedTarget:
    """
    Validate `url`'s target host against the SSRF address-block policy and, for a hostname
    (as opposed to a literal IP), pin the connection to the exact address that was validated
    by substituting it directly into the returned URL -- see module docstring, layer 2.

    Only called when no forward proxy is configured. Raises `ToolSSRFError` (a `ValueError`,
    never retried by `make_request`) when the target is blocked. DNS resolution failures
    (`socket.gaierror`) are left to propagate as-is so the caller can retry them exactly like
    any other transient network error.
    """
    parsed = httpx.URL(url)
    host = parsed.host
    if not host:
        # No host to validate (e.g. a malformed URL) -- let the request proceed and fail
        # naturally further down the stack rather than inventing a new error surface here.
        return _PinnedTarget(parsed, None, None)

    try:
        literal_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        _validate_ssrf_address(host, literal_ip, url)
        return _PinnedTarget(parsed, None, None)

    # Resolve exactly once. `socket.gaierror` propagates to the caller unchanged.
    addrinfo = socket.getaddrinfo(parsed.raw_host.decode("ascii"), None, type=socket.SOCK_STREAM)
    if not addrinfo:
        raise ToolSSRFError(f"Could not resolve an address to connect to for host '{host}' ('{url}')")

    resolved_ip = ipaddress.ip_address(addrinfo[0][4][0])
    _validate_ssrf_address(host, resolved_ip, url)

    # Substitute the validated address directly into the URL so the actual connection can
    # only ever go to the address we just checked -- no second DNS lookup, so no rebinding
    # window. Host header + TLS SNI are restored to the original hostname by the caller.
    pinned_url = parsed.copy_with(host=str(resolved_ip))
    return _PinnedTarget(pinned_url, parsed.netloc.decode("ascii"), host)


def make_request(method: str, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    # Convert requests-style allow_redirects to httpx-style follow_redirects
    if "allow_redirects" in kwargs:
        allow_redirects = kwargs.pop("allow_redirects")
        if "follow_redirects" not in kwargs:
            kwargs["follow_redirects"] = allow_redirects

    if "timeout" not in kwargs:
        kwargs["timeout"] = httpx.Timeout(
            timeout=dify_config.SSRF_DEFAULT_TIME_OUT,
            connect=dify_config.SSRF_DEFAULT_CONNECT_TIME_OUT,
            read=dify_config.SSRF_DEFAULT_READ_TIME_OUT,
            write=dify_config.SSRF_DEFAULT_WRITE_TIME_OUT,
        )

    # prioritize per-call option, which can be switched on and off inside the HTTP node on the web UI
    verify_option = kwargs.pop("ssl_verify", dify_config.HTTP_REQUEST_NODE_SSL_VERIFY)
    if not isinstance(verify_option, bool):
        raise ValueError("ssl_verify must be a boolean")
    client = _get_ssrf_client(verify_option)

    # Inject traceparent header for distributed tracing (when OTEL is not enabled)
    try:
        headers: Headers = _HEADERS_ADAPTER.validate_python(kwargs.get("headers") or {})
    except ValidationError as e:
        raise ValueError("headers must be a mapping of string keys to string values") from e
    headers = _inject_trace_headers(headers)
    kwargs["headers"] = headers

    # Preserve user-provided Host header
    # When using a forward proxy, httpx may override the Host header based on the URL.
    # We extract and preserve any explicitly set Host header to support virtual hosting.
    user_provided_host = _get_user_provided_host_header(headers)

    # In-process SSRF address guard (module docstring, layer 2). Skipped when a forward
    # proxy is configured: the real TCP connection targets the proxy, not `url`'s host, so
    # there is nothing meaningful for us to pin/validate here -- see _resolve_and_pin_host.
    proxy_configured = _is_ssrf_proxy_configured()

    retries = 0
    while retries <= max_retries:
        try:
            request_url: str | httpx.URL = url
            pinned_host_header: str | None = None
            pinned_sni_hostname: str | None = None
            if not proxy_configured:
                request_url, pinned_host_header, pinned_sni_hostname = _resolve_and_pin_host(url)

            # Preserve the user-provided Host header
            # httpx may override the Host header when using a proxy
            headers = {k: v for k, v in headers.items() if k.lower() != "host"}
            if user_provided_host is not None:
                headers["host"] = user_provided_host
            elif pinned_host_header is not None:
                # The request URL's host was rewritten to a raw, validated IP address to
                # pin the connection (see _resolve_and_pin_host); restore the original
                # virtual host so the server still sees the hostname it expects.
                headers["host"] = pinned_host_header
            kwargs["headers"] = headers
            if pinned_sni_hostname is not None:
                # Keep TLS SNI + certificate verification against the original hostname
                # even though we're connecting to its resolved IP directly.
                kwargs["extensions"] = {**kwargs.get("extensions", {}), "sni_hostname": pinned_sni_hostname}
            response = client.request(method=method, url=request_url, **kwargs)

            # Check for SSRF protection by Squid proxy
            if response.status_code in (401, 403):
                # Check if this is a Squid SSRF rejection
                server_header = response.headers.get("server", "").lower()
                via_header = response.headers.get("via", "").lower()

                # Squid typically identifies itself in Server or Via headers
                if "squid" in server_header or "squid" in via_header:
                    raise ToolSSRFError(
                        f"Access to '{url}' was blocked by SSRF protection. "
                        f"The URL may point to a private or local network address. "
                    )

            if response.status_code not in STATUS_FORCELIST:
                return response
            else:
                logger.warning(
                    "Received status code %s for URL %s which is in the force list",
                    response.status_code,
                    url,
                )

        except (httpx.RequestError, socket.gaierror) as e:
            # socket.gaierror surfaces from _resolve_and_pin_host's DNS resolution above; treat
            # it exactly like any other transient network error (retried with backoff), as
            # opposed to ToolSSRFError -- a definitive policy block that must never be retried.
            logger.warning("Request to URL %s failed on attempt %s: %s", url, retries + 1, e)
            if max_retries == 0:
                raise

        retries += 1
        if retries <= max_retries:
            time.sleep(BACKOFF_FACTOR * (2 ** (retries - 1)))
    raise MaxRetriesExceededError(f"Reached maximum retries ({max_retries}) for URL {url}")


def get(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("GET", url, max_retries=max_retries, **kwargs)


def post(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("POST", url, max_retries=max_retries, **kwargs)


def put(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("PUT", url, max_retries=max_retries, **kwargs)


def patch(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("PATCH", url, max_retries=max_retries, **kwargs)


def delete(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("DELETE", url, max_retries=max_retries, **kwargs)


def head(url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
    return make_request("HEAD", url, max_retries=max_retries, **kwargs)


class SSRFProxy:
    """
    Adapter exposing SSRF-protected HTTP helpers behind HttpClientProtocol.

    This is intentionally a thin wrapper over the existing module-level functions so callers can inject it
    where a protocol-typed HTTP client is expected.
    """

    @property
    def max_retries_exceeded_error(self) -> type[Exception]:
        return max_retries_exceeded_error

    @property
    def request_error(self) -> type[Exception]:
        return request_error

    def get(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return get(url=url, max_retries=max_retries, **kwargs)

    def head(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return head(url=url, max_retries=max_retries, **kwargs)

    def post(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return post(url=url, max_retries=max_retries, **kwargs)

    def put(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return put(url=url, max_retries=max_retries, **kwargs)

    def delete(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return delete(url=url, max_retries=max_retries, **kwargs)

    def patch(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> httpx.Response:
        return patch(url=url, max_retries=max_retries, **kwargs)


def _to_graphon_http_response(response: httpx.Response) -> HttpResponse:
    """Convert an ``httpx`` response into Graphon's transport-agnostic wrapper."""
    return HttpResponse(
        status_code=response.status_code,
        headers=dict(response.headers),
        content=response.content,
        url=str(response.url) if response.url else None,
        reason_phrase=response.reason_phrase,
        fallback_text=response.text,
    )


class GraphonSSRFProxy:
    """Adapter exposing SSRF helpers behind Graphon's ``HttpClientProtocol``."""

    @property
    def max_retries_exceeded_error(self) -> type[Exception]:
        return max_retries_exceeded_error

    @property
    def request_error(self) -> type[Exception]:
        return request_error

    def get(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(get(url=url, max_retries=max_retries, **kwargs))

    def head(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(head(url=url, max_retries=max_retries, **kwargs))

    def post(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(post(url=url, max_retries=max_retries, **kwargs))

    def put(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(put(url=url, max_retries=max_retries, **kwargs))

    def delete(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(delete(url=url, max_retries=max_retries, **kwargs))

    def patch(self, url: str, max_retries: int = SSRF_DEFAULT_MAX_RETRIES, **kwargs: Any) -> HttpResponse:
        return _to_graphon_http_response(patch(url=url, max_retries=max_retries, **kwargs))


ssrf_proxy = SSRFProxy()
graphon_ssrf_proxy = GraphonSSRFProxy()
