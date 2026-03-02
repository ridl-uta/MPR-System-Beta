from __future__ import annotations

import re
import time

try:
    import telnetlib  # Python <= 3.12
except ModuleNotFoundError:  # pragma: no cover - Python 3.13+
    telnetlib = None  # type: ignore[assignment]

PROMPT_RE_BYTES = re.compile(br"[>#]\s*$")
PROMPT_RE_TEXT = re.compile(r"[>#]\s*$")
APC_PROMPT_BYTES = re.compile(br"(?:\r?\n)?apc>\s*$", re.M)
APC_PROMPT_TEXT = re.compile(r"(?:\r?\n)?apc>\s*$", re.M)


class APCPDUClient:
    """Minimal APC telnet client for polling outlet power."""

    def __init__(
        self,
        *,
        host: str,
        username: str,
        password: str,
        port: int = 23,
        timeout_s: float = 8.0,
    ) -> None:
        self.host = host
        self.username = username
        self.password = password
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self._tn: telnetlib.Telnet | None = None
        self._prompts = [APC_PROMPT_BYTES, PROMPT_RE_BYTES]

    def connect(self) -> None:
        if telnetlib is None:
            raise RuntimeError(
                "telnetlib is unavailable in this Python runtime. "
                "Use Python <= 3.12 for APC telnet support."
            )
        self.close()
        self._tn = telnetlib.Telnet(self.host, port=self.port, timeout=self.timeout_s)
        try:
            self._tn.expect([b"User Name :", b"Username:"], self.timeout_s)
            self._tn.write(self.username.encode("utf-8") + b"\r\n")
            self._tn.expect([b"Password  :", b"Password:"], self.timeout_s)
            self._tn.write(self.password.encode("utf-8") + b"\r\n")
            idx, _, _ = self._tn.expect(self._prompts, self.timeout_s)
            if idx == -1:
                raise TimeoutError(f"login prompt not seen for {self.host}:{self.port}")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._tn is None:
            return
        try:
            self._tn.write(b"exit\r\n")
        except Exception:
            pass
        try:
            self._tn.close()
        finally:
            self._tn = None

    def ensure_alive(self) -> None:
        if self._tn is None:
            self.connect()
            return
        try:
            self._tn.write(b"\r\n")
            idx, _, _ = self._tn.expect(self._prompts, 2.0)
            if idx == -1:
                raise TimeoutError("prompt not seen")
        except Exception:
            self.connect()

    def run(self, command: str, deadline_s: float = 15.0) -> str:
        self.ensure_alive()
        try:
            return self._run_once(command, deadline_s)
        except (TimeoutError, ConnectionError, EOFError):
            self.connect()
            return self._run_once(command, deadline_s)

    def _run_once(self, command: str, deadline_s: float) -> str:
        if self._tn is None:
            raise ConnectionError("telnet session is not connected")

        end_time = time.monotonic() + deadline_s
        self._tn.write(command.encode("utf-8") + b"\r\n")

        chunks: list[bytes] = []
        while True:
            remaining = max(0.1, end_time - time.monotonic())
            try:
                idx, _, text = self._tn.expect(self._prompts, remaining)
            except EOFError as exc:
                raise ConnectionError("telnet connection closed") from exc

            if text:
                chunks.append(text)
            if idx != -1:
                break
            if remaining <= 0.11:
                raise TimeoutError("prompt not seen before deadline")

        out = b"".join(chunks).decode("utf-8", errors="replace")
        out = APC_PROMPT_TEXT.sub("", out)
        out = PROMPT_RE_TEXT.sub("", out).strip()

        first_token = command.split()[0] if command.split() else ""
        if first_token:
            echo_re = re.compile(rf"^(?:apc>\s*)?{re.escape(first_token)}\b.*\r?\n", re.I | re.M)
            out = echo_re.sub("", out, count=1).strip()
        return out
