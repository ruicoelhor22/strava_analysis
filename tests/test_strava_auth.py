from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest

from endurance_lab.cli import build_parser
from endurance_lab.config import project_path
from endurance_lab.strava_auth import (
    AUTH_REQUIRED_MESSAGE,
    AuthenticationStatus,
    PlaywrightTransport,
    check_authentication,
    interactive_login,
    is_authentication_url,
    profile_path,
    reset_authentication_profile,
)
from endurance_lab.strava_export import StravaDownloader, build_authenticated_session
from endurance_lab.strava_manifest import ManifestRow, load_manifest, save_manifest


@pytest.fixture
def private_dir() -> Path:
    path = Path("data/test-runs") / uuid4().hex
    path.mkdir(parents=True)
    return path


@pytest.fixture
def auth_profile() -> Path:
    return Path("training_data/auth") / f"test-profile-{uuid4().hex}"


@dataclass
class FakeNavigationResponse:
    status: int = 200


class FakePage:
    def __init__(self, destinations, *, login_form: bool = False):
        self.destinations = list(destinations)
        self.url = "about:blank"
        self.login_form = login_form
        self.closed = False

    def goto(self, url, **kwargs):
        destination, status = self.destinations.pop(0) if self.destinations else (url, 200)
        self.url = destination
        return FakeNavigationResponse(status)

    def query_selector(self, selector):
        if self.login_form and ("password" in selector or "email" in selector):
            return object()
        return None

    def close(self):
        self.closed = True


class FakeAPIResponse:
    def __init__(self, status=200, body=b"ok", url="https://www.strava.com/activities/123/export_original"):
        self.status = status
        self._body = body
        self.url = url
        self.headers = {"content-type": "application/octet-stream"}

    def body(self):
        return self._body


class FakeRequestContext:
    def __init__(self, responses=None):
        self.responses = list(responses or [FakeAPIResponse()])
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class FakeBrowserContext:
    def __init__(self, page=None, responses=None):
        self.pages = [page] if page else []
        self.request = FakeRequestContext(responses)
        self.closed = False

    def new_page(self):
        page = FakePage([("https://www.strava.com/dashboard", 200)])
        self.pages.append(page)
        return page

    def close(self):
        self.closed = True


class FakeChromium:
    def __init__(self, contexts):
        self.contexts = list(contexts)
        self.launches = []

    def launch_persistent_context(self, **kwargs):
        self.launches.append(kwargs)
        return self.contexts.pop(0)


class FakePlaywright:
    def __init__(self, chromium):
        self.chromium = chromium
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeManager:
    def __init__(self, playwright):
        self.playwright = playwright

    def start(self):
        return self.playwright

    def __exit__(self, *args):
        return None


def test_profile_path_is_created_and_reused_with_windows_safe_path(auth_profile):
    profile = auth_profile.with_name(auth_profile.name + " with spaces")
    contexts = [FakeBrowserContext(), FakeBrowserContext()]
    chromium = FakeChromium(contexts)
    playwrights = []

    def factory():
        playwright = FakePlaywright(chromium)
        playwrights.append(playwright)
        return FakeManager(playwright)

    for _ in range(2):
        with PlaywrightTransport(profile, playwright_factory=factory) as transport:
            assert transport.context is not None
    assert profile.is_dir()
    assert len(chromium.launches) == 2
    assert all(call["user_data_dir"] == profile.resolve() for call in chromium.launches)
    assert all(call["headless"] is True for call in chromium.launches)
    assert all(playwright.stopped for playwright in playwrights)


def test_authenticated_session_detection():
    page = FakePage([("https://www.strava.com/dashboard?feed_type=following", 200)])
    status = check_authentication(FakeBrowserContext(page), page=page)
    assert status == AuthenticationStatus(
        True,
        True,
        "authenticated Strava dashboard loaded",
        "https://www.strava.com/dashboard?feed_type=following",
    )


@pytest.mark.parametrize(
    "url,login_form,reason",
    [
        ("https://www.strava.com/login?returnUrl=%2Fdashboard", False, "redirected to Strava login"),
        ("https://www.strava.com/dashboard", True, "Strava login form is present"),
    ],
)
def test_unauthenticated_session_detection(url, login_form, reason):
    page = FakePage([(url, 200)], login_form=login_form)
    status = check_authentication(FakeBrowserContext(page), page=page)
    assert not status.authenticated
    assert status.confident
    assert status.reason == reason


def test_uncertain_authentication_is_not_reported_as_success():
    page = FakePage([("https://www.strava.com/maintenance", 200)])
    status = check_authentication(FakeBrowserContext(page), page=page)
    assert not status.authenticated
    assert not status.confident


def test_interactive_login_never_handles_credentials(auth_profile):
    page = FakePage(
        [
            ("https://www.strava.com/login", 200),
            ("https://www.strava.com/login", 200),
            ("https://www.strava.com/dashboard", 200),
        ]
    )
    context = FakeBrowserContext(page)
    prompts = []
    output = []

    class FakeTransport:
        def __init__(self, profile, headless):
            assert headless is False
            self.context = context

        def __enter__(self):
            return self

        def __exit__(self, *args):
            context.close()

    status = interactive_login(
        auth_profile,
        emit=output.append,
        confirm=lambda prompt: prompts.append(prompt) or "",
        transport_factory=FakeTransport,
    )
    assert status.authenticated
    assert len(prompts) == 1
    assert any("manually" in line for line in output)
    assert context.closed


def test_playwright_transport_converts_login_redirect_to_401(auth_profile):
    response = FakeAPIResponse(
        200,
        b"<html>Log in</html>",
        "https://www.strava.com/login?returnUrl=%2Factivities%2F123%2Fexport_original",
    )
    context = FakeBrowserContext(responses=[response])
    chromium = FakeChromium([context])
    playwright = FakePlaywright(chromium)
    with PlaywrightTransport(
        auth_profile, playwright_factory=lambda: FakeManager(playwright)
    ) as transport:
        adapted = transport.get("https://www.strava.com/activities/123/export_original", timeout=30)
    assert adapted.status_code == 401
    assert adapted.final_url.startswith("https://www.strava.com/login")


def test_expired_playwright_session_stops_download_and_preserves_pending_rows(private_dir):
    manifest = private_dir / "manifest.csv"
    save_manifest([ManifestRow("123456789"), ManifestRow("123456790")], manifest)

    class ExpiredTransport:
        def get(self, url, **kwargs):
            return type(
                "Response",
                (),
                {"status_code": 401, "content": b"", "headers": {}},
            )()

    summary = StravaDownloader(
        ExpiredTransport(), delay_seconds=0, max_retries=0, sleep=lambda _: None
    ).run(manifest, private_dir / "import")
    rows = load_manifest(manifest)
    assert summary.authentication_required
    assert rows[0].download_status == "authentication_required"
    assert rows[1].download_status == "pending"
    assert "python -m endurance_lab strava-login" in AUTH_REQUIRED_MESSAGE


def test_cookies_authentication_remains_available(private_dir):
    cookies = private_dir / "cookies.txt"
    cookies.write_text(
        "# Netscape HTTP Cookie File\n"
        ".strava.com\tTRUE\t/\tTRUE\t2147483647\t_session\ttest-value\n",
        encoding="utf-8",
    )
    session = build_authenticated_session(cookies)
    assert session.cookies.get("_session", domain=".strava.com") == "test-value"
    session.close()


def test_playwright_is_default_and_cookies_are_explicit_fallback():
    parser = build_parser()
    default = parser.parse_args(["strava-download", "--dry-run"])
    fallback = parser.parse_args(["strava-download", "--auth", "cookies", "--dry-run"])
    assert default.auth == "playwright"
    assert fallback.auth == "cookies"


def test_profile_is_under_private_ignored_tree():
    resolved = profile_path()
    private_root = project_path("training_data/auth").resolve()
    assert resolved.is_relative_to(private_root)
    ignore = Path(".gitignore").read_text(encoding="utf-8")
    assert "training_data/" in ignore
    assert "browser-profile/" in ignore


def test_auth_reset_requires_confirmation_and_only_removes_profile():
    profile = Path("training_data/auth") / f"test-reset-{uuid4().hex}"
    profile.mkdir(parents=True)
    (profile / "session-state").write_text("private", encoding="utf-8")
    assert not reset_authentication_profile(profile, confirm=lambda _: "no", emit=lambda _: None)
    assert profile.exists()
    assert reset_authentication_profile(profile, confirm=lambda _: "RESET", emit=lambda _: None)
    assert not profile.exists()


def test_auth_reset_refuses_paths_outside_private_auth_tree(private_dir):
    with pytest.raises(ValueError, match="private Git-ignored"):
        reset_authentication_profile(private_dir, confirm=lambda _: "RESET")


def test_authentication_url_detection_is_strava_specific():
    assert is_authentication_url("https://www.strava.com/login?x=1")
    assert not is_authentication_url("https://example.com/login")
    assert not is_authentication_url("https://www.strava.com/dashboard")
