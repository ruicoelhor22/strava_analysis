from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Callable, Protocol
from urllib.parse import urlparse

from endurance_lab.config import project_path
from endurance_lab.strava_export import TransportError


DEFAULT_PROFILE = Path("training_data/auth/strava-browser-profile")
STRAVA_HOME_URL = "https://www.strava.com/"
STRAVA_LOGIN_URL = "https://www.strava.com/login"
STRAVA_AUTH_CHECK_URL = "https://www.strava.com/dashboard"
AUTH_REQUIRED_MESSAGE = (
    "Strava authentication is required. Run:\n\n"
    "python -m endurance_lab strava-login"
)


@dataclass(frozen=True)
class AuthenticationStatus:
    authenticated: bool
    confident: bool
    reason: str
    final_url: str = ""


@dataclass(frozen=True)
class BrowserResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]
    final_url: str


class BrowserTransportLike(Protocol):
    context: object

    def __enter__(self): ...
    def __exit__(self, exc_type, exc, traceback): ...


def profile_path(value: str | Path | None = None) -> Path:
    resolved = project_path(value or DEFAULT_PROFILE).resolve()
    private_auth_root = project_path("training_data/auth").resolve()
    if resolved == private_auth_root or not resolved.is_relative_to(private_auth_root):
        raise ValueError(
            f"Browser profiles must be a child of the private Git-ignored directory "
            f"{private_auth_root}: {resolved}"
        )
    return resolved


class PlaywrightTransport:
    """Authenticated HTTP-like transport backed by a persistent Chromium context."""

    def __init__(
        self,
        profile: str | Path | None = None,
        *,
        headless: bool = True,
        playwright_factory: Callable[[], object] | None = None,
    ) -> None:
        self.profile = profile_path(profile)
        self.headless = headless
        self.playwright_factory = playwright_factory
        self._manager = None
        self._playwright = None
        self.context = None

    def __enter__(self) -> "PlaywrightTransport":
        self.profile.mkdir(parents=True, exist_ok=True)
        factory = self.playwright_factory
        if factory is None:
            try:
                from playwright.sync_api import sync_playwright
            except ImportError as exc:
                raise RuntimeError(
                    "Playwright is not installed. Run: python -m pip install -e ."
                ) from exc
            factory = sync_playwright
        try:
            self._manager = factory()
            self._playwright = self._manager.start()
            self.context = self._playwright.chromium.launch_persistent_context(
                user_data_dir=self.profile,
                headless=self.headless,
                accept_downloads=False,
                no_viewport=True,
            )
        except Exception as exc:
            try:
                self.close()
            except Exception:
                self.context = None
                self._manager = None
                self._playwright = None
            raise RuntimeError(
                "Could not launch Playwright Chromium. Install it with: "
                "python -m playwright install chromium. "
                f"Details: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if self.context is not None:
            try:
                self.context.close()
            finally:
                self.context = None
        if self._playwright is not None:
            try:
                self._playwright.stop()
            finally:
                self._manager = None
                self._playwright = None
        elif self._manager is not None:
            try:
                self._manager.__exit__(None, None, None)
            except Exception:
                # A partially started driver may not have a transport to stop.
                pass
            finally:
                self._manager = None

    def get(self, url: str, **kwargs) -> BrowserResponse:
        if self.context is None:
            raise TransportError("Playwright browser context is not open")
        timeout_seconds = float(kwargs.get("timeout", 60.0))
        try:
            response = self.context.request.get(
                url,
                timeout=timeout_seconds * 1000,
                fail_on_status_code=False,
                max_redirects=20,
            )
            content = response.body()
            final_url = str(response.url)
            status = int(response.status)
            if is_authentication_url(final_url):
                status = 401
            return BrowserResponse(status, content, dict(response.headers), final_url)
        except Exception as exc:
            raise TransportError(f"Playwright request failed: {exc}") from exc


def check_authentication(context, page=None, timeout_seconds: float = 30.0) -> AuthenticationStatus:
    own_page = page is None
    active_page = page or context.new_page()
    try:
        response = active_page.goto(
            STRAVA_AUTH_CHECK_URL,
            wait_until="domcontentloaded",
            timeout=timeout_seconds * 1000,
        )
        final_url = str(active_page.url)
        status = int(response.status) if response is not None else None
        if is_authentication_url(final_url):
            return AuthenticationStatus(False, True, "redirected to Strava login", final_url)
        if _has_login_form(active_page):
            return AuthenticationStatus(False, True, "Strava login form is present", final_url)
        if status in {401, 403}:
            return AuthenticationStatus(False, True, f"authentication check returned HTTP {status}", final_url)
        parsed = urlparse(final_url)
        if (
            parsed.hostname
            and parsed.hostname.lower().endswith("strava.com")
            and parsed.path.rstrip("/").startswith("/dashboard")
            and status is not None
            and 200 <= status < 400
        ):
            return AuthenticationStatus(True, True, "authenticated Strava dashboard loaded", final_url)
        return AuthenticationStatus(
            False,
            False,
            f"could not confidently verify authentication (HTTP {status}, URL {final_url})",
            final_url,
        )
    except Exception as exc:
        return AuthenticationStatus(False, False, f"authentication check failed: {exc}")
    finally:
        if own_page:
            active_page.close()


def interactive_login(
    profile: str | Path | None = None,
    *,
    emit: Callable[[str], None] = print,
    confirm: Callable[[str], str] = input,
    transport_factory: Callable[..., BrowserTransportLike] = PlaywrightTransport,
) -> AuthenticationStatus:
    resolved = profile_path(profile)
    with transport_factory(resolved, headless=False) as transport:
        context = transport.context
        page = context.pages[0] if context.pages else context.new_page()
        current = check_authentication(context, page=page)
        if current.authenticated:
            emit("Strava session is already authenticated.")
            return current
        page.goto(STRAVA_LOGIN_URL, wait_until="domcontentloaded")
        emit("A visible Chromium window is open at Strava.")
        emit("Log in manually and complete any MFA or CAPTCHA yourself.")
        confirm("After Strava shows your authenticated account, return here and press Enter...")
        result = check_authentication(context, page=page)
        if result.authenticated:
            emit("Strava authentication confirmed. The persistent profile has been saved.")
        elif result.confident:
            emit("Strava is not authenticated. Run strava-login again when ready.")
        else:
            emit(f"Authentication could not be confidently confirmed: {result.reason}")
        return result


def reset_authentication_profile(
    profile: str | Path | None = None,
    *,
    confirm: Callable[[str], str] = input,
    emit: Callable[[str], None] = print,
) -> bool:
    resolved = profile_path(profile)
    if not resolved.exists():
        emit(f"No authentication profile exists at: {resolved}")
        return False
    answer = confirm(
        f"This will permanently remove only the Strava browser profile at:\n{resolved}\n"
        "Type RESET to continue: "
    )
    if answer.strip() != "RESET":
        emit("Authentication reset cancelled.")
        return False
    shutil.rmtree(resolved)
    emit("Strava authentication profile removed. Downloaded activities were not touched.")
    return True


def is_authentication_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname or not parsed.hostname.lower().endswith("strava.com"):
        return False
    path = parsed.path.lower().rstrip("/")
    return path in {"/login", "/register", "/session", "/account/recover"} or path.startswith("/login/")


def _has_login_form(page) -> bool:
    return page.query_selector("input[type='password']") is not None and (
        page.query_selector("input[type='email']") is not None
        or page.query_selector("input[name='email']") is not None
    )
