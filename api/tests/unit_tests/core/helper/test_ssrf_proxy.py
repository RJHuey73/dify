import ipaddress
import socket
from unittest.mock import ANY, MagicMock, call, patch

import httpx
import pytest

from core.helper.ssrf_proxy import (
    SSRF_DEFAULT_MAX_RETRIES,
    SSRFProxy,
    _build_ssrf_client,
    _get_user_provided_host_header,
    _is_blocked_address,
    _is_ssrf_allowlisted,
    _parse_ssrf_allowlist,
    _resolve_and_pin_host,
    _to_graphon_http_response,
    _validate_ssrf_address,
    graphon_ssrf_proxy,
    make_request,
    max_retries_exceeded_error,
    request_error,
)
from core.tools.errors import ToolSSRFError


# `_resolve_and_pin_host` (the in-process SSRF address guard added alongside
# `SSRF_ALLOWED_HOSTS`) performs a real `socket.getaddrinfo` call whenever no
# forward proxy is configured -- which is every test in this module unless it
# patches the proxy config vars. Without this autouse fixture, every existing
# `make_request(..., "http://example.com")` call below would silently start
# depending on live DNS resolution succeeding for "example.com", turning a
# hermetic unit test suite into one that needs network access. Tests that
# actually exercise the address guard itself override this patch locally.
@pytest.fixture(autouse=True)
def _mock_dns_resolves_to_safe_public_address():
    with patch(
        "core.helper.ssrf_proxy.socket.getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))],
    ) as mocked:
        yield mocked


@patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
def test_successful_request(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.request.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = make_request("GET", "http://example.com")
    assert response.status_code == 200
    mock_client.request.assert_called_once()


@patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
def test_retry_exceed_max_retries(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_client.request.return_value = mock_response
    mock_get_client.return_value = mock_client

    with pytest.raises(Exception) as e:
        make_request("GET", "http://example.com", max_retries=SSRF_DEFAULT_MAX_RETRIES - 1)
    assert str(e.value) == f"Reached maximum retries ({SSRF_DEFAULT_MAX_RETRIES - 1}) for URL http://example.com"


def test_build_ssrf_client_passes_ssl_verify_to_proxy_mount_transports():
    mock_client = MagicMock()
    http_transport = MagicMock()
    https_transport = MagicMock()

    with (
        patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", None),
        patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTP_URL", "http://proxy.example.com:8080"),
        patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTPS_URL", "http://proxy.example.com:8443"),
        patch("core.helper.ssrf_proxy.httpx.HTTPTransport", side_effect=[http_transport, https_transport]) as transport,
        patch("core.helper.ssrf_proxy.httpx.Client", return_value=mock_client) as client,
    ):
        ssrf_client = _build_ssrf_client(verify=False)

    assert ssrf_client is mock_client
    transport.assert_has_calls(
        [
            call(proxy="http://proxy.example.com:8080", verify=False),
            call(proxy="http://proxy.example.com:8443", verify=False),
        ],
    )
    client.assert_called_once_with(
        mounts={"http://": http_transport, "https://": https_transport},
        verify=False,
        limits=ANY,
    )


class TestGetUserProvidedHostHeader:
    """Tests for _get_user_provided_host_header function."""

    def test_returns_none_when_headers_is_none(self):
        assert _get_user_provided_host_header(None) is None

    def test_returns_none_when_headers_is_empty(self):
        assert _get_user_provided_host_header({}) is None

    def test_returns_none_when_host_header_not_present(self):
        headers = {"Content-Type": "application/json", "Authorization": "Bearer token"}
        assert _get_user_provided_host_header(headers) is None

    def test_returns_host_header_lowercase(self):
        headers = {"host": "example.com"}
        assert _get_user_provided_host_header(headers) == "example.com"

    def test_returns_host_header_uppercase(self):
        headers = {"HOST": "example.com"}
        assert _get_user_provided_host_header(headers) == "example.com"

    def test_returns_host_header_mixed_case(self):
        headers = {"HoSt": "example.com"}
        assert _get_user_provided_host_header(headers) == "example.com"

    def test_returns_host_header_from_multiple_headers(self):
        headers = {"Content-Type": "application/json", "Host": "api.example.com", "Authorization": "Bearer token"}
        assert _get_user_provided_host_header(headers) == "api.example.com"

    def test_returns_first_host_header_when_duplicates(self):
        headers = {"host": "first.com", "Host": "second.com"}
        # Should return the first one encountered (iteration order is preserved in dict)
        result = _get_user_provided_host_header(headers)
        assert result in ("first.com", "second.com")


@patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
def test_host_header_preservation_with_user_header(mock_get_client):
    """Test that user-provided Host header is preserved in the request."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.request.return_value = mock_response
    mock_get_client.return_value = mock_client

    custom_host = "custom.example.com:8080"
    response = make_request("GET", "http://example.com", headers={"Host": custom_host})

    assert response.status_code == 200
    # Verify client.request was called with the host header preserved (lowercase)
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["headers"]["host"] == custom_host


@patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
@pytest.mark.parametrize("host_key", ["host", "HOST", "Host"])
def test_host_header_preservation_case_insensitive(mock_get_client, host_key):
    """Test that Host header is preserved regardless of case."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.request.return_value = mock_response
    mock_get_client.return_value = mock_client

    response = make_request("GET", "http://example.com", headers={host_key: "api.example.com"})

    assert response.status_code == 200
    # Host header should be normalized to lowercase "host"
    call_kwargs = mock_client.request.call_args.kwargs
    assert call_kwargs["headers"]["host"] == "api.example.com"


class TestFollowRedirectsParameter:
    """Tests for follow_redirects parameter handling.

    These tests verify that follow_redirects is correctly passed to client.request().
    """

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_follow_redirects_passed_to_request(self, mock_get_client):
        """Verify follow_redirects IS passed to client.request()."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        make_request("GET", "http://example.com", follow_redirects=True)

        # Verify follow_redirects was passed to request
        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs.get("follow_redirects") is True

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_allow_redirects_converted_to_follow_redirects(self, mock_get_client):
        """Verify allow_redirects (requests-style) is converted to follow_redirects (httpx-style)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Use allow_redirects (requests-style parameter)
        make_request("GET", "http://example.com", allow_redirects=True)

        # Verify it was converted to follow_redirects
        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs.get("follow_redirects") is True
        assert "allow_redirects" not in call_kwargs

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_follow_redirects_not_set_when_not_specified(self, mock_get_client):
        """Verify follow_redirects is not in kwargs when not specified (httpx default behavior)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        make_request("GET", "http://example.com")

        # follow_redirects should not be in kwargs, letting httpx use its default
        call_kwargs = mock_client.request.call_args.kwargs
        assert "follow_redirects" not in call_kwargs

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_follow_redirects_takes_precedence_over_allow_redirects(self, mock_get_client):
        """Verify follow_redirects takes precedence when both are specified."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        # Both specified - follow_redirects should take precedence
        make_request("GET", "http://example.com", allow_redirects=False, follow_redirects=True)

        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs.get("follow_redirects") is True


def test_to_graphon_http_response_preserves_httpx_response_fields() -> None:
    response = httpx.Response(
        201,
        headers={"X-Test": "1"},
        content=b"payload",
        request=httpx.Request("GET", "https://example.com/resource"),
    )

    wrapped = _to_graphon_http_response(response)

    assert wrapped.status_code == 201
    assert wrapped.headers == {"x-test": "1", "content-length": "7"}
    assert wrapped.content == b"payload"
    assert wrapped.url == "https://example.com/resource"
    assert wrapped.reason_phrase == "Created"
    assert wrapped.text == "payload"


def test_ssrf_proxy_exposes_expected_error_types() -> None:
    proxy = SSRFProxy()

    assert proxy.max_retries_exceeded_error is max_retries_exceeded_error
    assert proxy.request_error is request_error
    assert graphon_ssrf_proxy.max_retries_exceeded_error is max_retries_exceeded_error
    assert graphon_ssrf_proxy.request_error is request_error


@pytest.mark.parametrize("method_name", ["get", "head", "post", "put", "delete", "patch"])
def test_graphon_ssrf_proxy_wraps_module_requests(method_name: str) -> None:
    response = httpx.Response(
        200,
        headers={"X-Test": "1"},
        content=b"ok",
        request=httpx.Request("GET", "https://example.com/resource"),
    )

    with patch(f"core.helper.ssrf_proxy.{method_name}", return_value=response) as mock_method:
        wrapped = getattr(graphon_ssrf_proxy, method_name)(
            "https://example.com/resource",
            max_retries=3,
            headers={"X-Test": "1"},
        )

    mock_method.assert_called_once_with(
        url="https://example.com/resource",
        max_retries=3,
        headers={"X-Test": "1"},
    )
    assert wrapped.status_code == 200
    assert wrapped.url == "https://example.com/resource"
    assert wrapped.content == b"ok"


class TestIsBlockedAddress:
    """Unit tests for `_is_blocked_address`'s private/internal-use classification."""

    @pytest.mark.parametrize(
        "address",
        [
            "10.0.0.1",  # RFC 1918 private
            "172.16.0.1",  # RFC 1918 private
            "192.168.1.1",  # RFC 1918 private
            "127.0.0.1",  # loopback
            "169.254.169.254",  # link-local / cloud metadata endpoint
            "224.0.0.1",  # multicast
            "0.0.0.0",  # unspecified
            "::1",  # IPv6 loopback
            "fe80::1",  # IPv6 link-local
            "fc00::1",  # IPv6 unique local (private)
        ],
    )
    def test_blocks_private_and_internal_use_addresses(self, address):
        assert _is_blocked_address(ipaddress.ip_address(address)) is True

    @pytest.mark.parametrize("address", ["93.184.215.14", "1.1.1.1", "8.8.8.8", "2606:4700:4700::1111"])
    def test_allows_public_addresses(self, address):
        assert _is_blocked_address(ipaddress.ip_address(address)) is False


class TestParseSsrfAllowlist:
    """Unit tests for `_parse_ssrf_allowlist`'s hostname/CIDR splitting."""

    def test_none_input_yields_empty_allowlist(self):
        hostnames, networks = _parse_ssrf_allowlist(None)
        assert hostnames == frozenset()
        assert networks == ()

    def test_empty_string_yields_empty_allowlist(self):
        hostnames, networks = _parse_ssrf_allowlist("")
        assert hostnames == frozenset()
        assert networks == ()

    def test_hostname_entries_are_lowercased(self):
        hostnames, _networks = _parse_ssrf_allowlist("Internal-API.Example.Com")
        assert hostnames == frozenset({"internal-api.example.com"})

    def test_ip_and_cidr_entries_parsed_as_networks(self):
        _hostnames, networks = _parse_ssrf_allowlist("10.20.0.0/24, 192.168.1.5")
        assert ipaddress.ip_network("10.20.0.0/24") in networks
        assert ipaddress.ip_network("192.168.1.5/32") in networks

    def test_mixed_hostnames_and_networks(self):
        hostnames, networks = _parse_ssrf_allowlist("internal-api.example.com,10.20.0.0/24")
        assert hostnames == frozenset({"internal-api.example.com"})
        assert networks == (ipaddress.ip_network("10.20.0.0/24"),)

    def test_blank_entries_are_skipped(self):
        hostnames, networks = _parse_ssrf_allowlist(" , internal.example.com , , ")
        assert hostnames == frozenset({"internal.example.com"})
        assert networks == ()


class TestIsSsrfAllowlisted:
    """Unit tests for `_is_ssrf_allowlisted`."""

    def test_hostname_match_is_case_insensitive(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", "Internal.Example.Com"):
            assert _is_ssrf_allowlisted("internal.example.com", None) is True
            assert _is_ssrf_allowlisted("INTERNAL.EXAMPLE.COM", None) is True

    def test_ip_within_allowlisted_cidr_is_allowed(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", "10.20.0.0/24"):
            assert _is_ssrf_allowlisted("some-internal-host", ipaddress.ip_address("10.20.0.5")) is True

    def test_ip_outside_allowlisted_cidr_is_not_allowed(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", "10.20.0.0/24"):
            assert _is_ssrf_allowlisted("some-internal-host", ipaddress.ip_address("10.30.0.5")) is False

    def test_no_allowlist_configured_denies_everything(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None):
            assert _is_ssrf_allowlisted("internal.example.com", ipaddress.ip_address("10.0.0.1")) is False


class TestValidateSsrfAddress:
    """Unit tests for `_validate_ssrf_address`."""

    def test_blocked_address_raises_when_not_allowlisted(self):
        with (
            patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None),
            pytest.raises(ToolSSRFError, match="blocked by SSRF protection"),
        ):
            _validate_ssrf_address(
                "internal.example.com", ipaddress.ip_address("10.0.0.1"), "http://internal.example.com"
            )

    def test_blocked_address_allowed_when_allowlisted(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", "internal.example.com"):
            _validate_ssrf_address(
                "internal.example.com", ipaddress.ip_address("10.0.0.1"), "http://internal.example.com"
            )

    def test_public_address_never_raises(self):
        with patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None):
            _validate_ssrf_address("example.com", ipaddress.ip_address("93.184.215.14"), "http://example.com")


class TestResolveAndPinHost:
    """Unit tests for `_resolve_and_pin_host` -- the core of the in-process SSRF
    address guard (layer 2): resolve once, validate, then pin the connection to
    the exact validated address so no rebinding window exists between check and
    connect."""

    def test_hostname_resolving_to_public_address_is_pinned(self):
        with patch(
            "core.helper.ssrf_proxy.socket.getaddrinfo",
            return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))],
        ):
            pinned = _resolve_and_pin_host("http://example.com/path")

        assert pinned.url.host == "93.184.215.14"
        assert pinned.host_header == "example.com"
        assert pinned.sni_hostname == "example.com"

    def test_hostname_resolving_to_private_address_is_blocked(self):
        with (
            patch(
                "core.helper.ssrf_proxy.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))],
            ),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None),
            pytest.raises(ToolSSRFError, match="blocked by SSRF protection"),
        ):
            _resolve_and_pin_host("http://metadata.internal/latest/meta-data/")

    def test_hostname_resolving_to_private_address_is_allowed_when_allowlisted(self):
        with (
            patch(
                "core.helper.ssrf_proxy.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.20.0.5", 0))],
            ),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", "internal-api.example.com"),
        ):
            pinned = _resolve_and_pin_host("http://internal-api.example.com/path")

        assert pinned.url.host == "10.20.0.5"

    def test_dns_resolution_failure_propagates(self):
        with (
            patch("core.helper.ssrf_proxy.socket.getaddrinfo", side_effect=socket.gaierror("nope")),
            pytest.raises(socket.gaierror),
        ):
            _resolve_and_pin_host("http://does-not-resolve.invalid/path")

    def test_literal_ip_host_is_validated_without_dns_lookup(self):
        with (
            patch("core.helper.ssrf_proxy.socket.getaddrinfo") as mock_getaddrinfo,
            patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None),
            pytest.raises(ToolSSRFError, match="blocked by SSRF protection"),
        ):
            _resolve_and_pin_host("http://127.0.0.1/path")
        mock_getaddrinfo.assert_not_called()

    def test_literal_public_ip_host_passes_through_unpinned(self):
        with patch("core.helper.ssrf_proxy.socket.getaddrinfo") as mock_getaddrinfo:
            pinned = _resolve_and_pin_host("http://93.184.215.14/path")

        mock_getaddrinfo.assert_not_called()
        assert pinned.url.host == "93.184.215.14"
        assert pinned.host_header is None
        assert pinned.sni_hostname is None


class TestMakeRequestSsrfAddressGuard:
    """End-to-end (through `make_request`, with the transport mocked) coverage
    for the in-process SSRF address guard being applied and skipped in the
    right circumstances."""

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_private_target_is_blocked_when_no_proxy_configured(self, mock_get_client):
        mock_get_client.return_value = MagicMock()

        with (
            patch(
                "core.helper.ssrf_proxy.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))],
            ),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_ALLOWED_HOSTS", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTP_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTPS_URL", None),
            pytest.raises(ToolSSRFError, match="blocked by SSRF protection"),
        ):
            make_request("GET", "http://metadata.internal/latest/meta-data/")

        mock_get_client.return_value.request.assert_not_called()

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_address_guard_is_skipped_when_forward_proxy_configured(self, mock_get_client):
        """When an external forward proxy is configured, the real TCP connection
        targets the proxy, not the request URL's host -- so the in-process guard
        must not run at all, even for a target that would otherwise be blocked."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        with (
            patch("core.helper.ssrf_proxy.socket.getaddrinfo") as mock_getaddrinfo,
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", "http://proxy.example.com:3128"),
        ):
            response = make_request("GET", "http://169.254.169.254/latest/meta-data/")

        assert response.status_code == 200
        mock_getaddrinfo.assert_not_called()
        # The original URL (not a pinned/rewritten one) must be what's requested,
        # since the proxy -- not this module -- resolves and connects to it.
        assert mock_client.request.call_args.kwargs["url"] == "http://169.254.169.254/latest/meta-data/"

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_public_target_is_pinned_and_request_uses_resolved_address(self, mock_get_client):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client.request.return_value = mock_response
        mock_get_client.return_value = mock_client

        with (
            patch(
                "core.helper.ssrf_proxy.socket.getaddrinfo",
                return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 0))],
            ),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTP_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTPS_URL", None),
        ):
            response = make_request("GET", "http://example.com/path")

        assert response.status_code == 200
        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs["url"].host == "93.184.215.14"
        assert call_kwargs["headers"]["host"] == "example.com"
        assert call_kwargs["extensions"]["sni_hostname"] == "example.com"

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_dns_failure_with_zero_retries_reraises_immediately(self, mock_get_client):
        mock_get_client.return_value = MagicMock()

        with (
            patch("core.helper.ssrf_proxy.socket.getaddrinfo", side_effect=socket.gaierror("temporary failure")),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTP_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTPS_URL", None),
            pytest.raises(socket.gaierror),
        ):
            make_request("GET", "http://does-not-resolve.invalid/path", max_retries=0)

        # Not silently swallowed as a ToolSSRFError -- a DNS failure is a
        # transient network condition, not a definitive policy block.
        mock_get_client.return_value.request.assert_not_called()

    @patch("core.helper.ssrf_proxy._get_ssrf_client", autospec=True)
    def test_dns_failure_is_retried_with_backoff_before_giving_up(self, mock_get_client):
        mock_get_client.return_value = MagicMock()

        with (
            patch("core.helper.ssrf_proxy.socket.getaddrinfo", side_effect=socket.gaierror("temporary failure")),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_ALL_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTP_URL", None),
            patch("core.helper.ssrf_proxy.dify_config.SSRF_PROXY_HTTPS_URL", None),
            patch("core.helper.ssrf_proxy.time.sleep") as mock_sleep,
            pytest.raises(max_retries_exceeded_error),
        ):
            make_request("GET", "http://does-not-resolve.invalid/path", max_retries=2)

        # Retried (with backoff sleeps) rather than failing on the first
        # attempt, and never reached the transport layer at all.
        assert mock_sleep.call_count == 2
        mock_get_client.return_value.request.assert_not_called()
