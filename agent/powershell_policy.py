"""Risk classification for PowerShell commands requiring user confirmation."""
import re


_RISKY_COMMAND = re.compile(
    r"""(?ix)
    \b(?:remove-item|rm|ri|del(?:ete)?|erase|rd|rmdir|kill|clear-content|clear-item|
       set-content|add-content|out-file|new-item|move-item|copy-item|rename-item|
       format-volume|format-disk|clear-disk|initialize-disk|new-partition|
       remove-partition|stop-computer|restart-computer|stop-process|
       stop-service|set-service|disable-computerrestore|disable-windowsoptionalfeature|
       set-executionpolicy|invoke-expression|iex|add-mppreference|
       set-mppreference|reg(?:\.exe)?\s+(?:delete|add)|sc(?:\.exe)?\s+delete|
       bcdedit|diskpart|cipher\s+/w|invoke-webrequest|iwr|invoke-restmethod|irm|
       start-bitstransfer|add-type|downloadstring|frombase64string)\b
    |
    \b(?:start-process)\s+(?:powershell|pwsh|cmd)(?:\.exe)?\b
    |
    \b(?:cmd(?:\.exe)?)\s+/c\b[^\r\n|;]*\b(?:del|erase|rmdir|format)\b
    |
    \b(?:curl|wget)\b[^\r\n|;]*\|\s*(?:invoke-expression|iex)\b
    |
    \b(?:shutdown|logoff)\b
    |
    (?:^|\s)-(?:encodedcommand|enc)\b
    """,
)


def requires_powershell_confirmation(command: str) -> bool:
    """Require explicit user approval for commands with destructive/system effects."""
    return bool(_RISKY_COMMAND.search(command or ""))
