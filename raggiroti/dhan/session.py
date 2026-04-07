from __future__ import annotations


class DhanUnavailable(Exception):
    pass


def create_dhan_client(client_id: str, access_token: str):
    """
    Optional integration.

    Install Dhan SDK first (example):
      py -m pip install dhanhq
    """
    try:
        from dhanhq import dhanhq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise DhanUnavailable("Dhan SDK not installed. Install 'dhanhq' to enable.") from e

    return dhanhq(client_id, access_token)

