import logging
import logging.config
import re
from typing import Any

import structlog
from structlog.types import Processor

from app.core.config import get_settings

_REQ_LINE_RE = re.compile(r"(?P<method>[A-Z]+)\s+(?P<path>\S+)\s+(?P<protocol>HTTP/\d\.\d)")


def add_request_context(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return event_dict


def uvicorn_access_to_structured(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    record = event_dict.get("_record")
    if not record or record.name != "uvicorn.access":
        return event_dict

    try:
        client, request_line, status = (None, None, None)
        if isinstance(record.args, tuple) and len(record.args) == 3:
            client, request_line, status = record.args

        if client:
            if isinstance(client, (list, tuple)):
                client = ":".join(str(x) for x in client)
            client_ip, _, client_port = str(client).partition(":")
            event_dict["client_ip"] = client_ip
            if client_port:
                event_dict["client_port"] = client_port

        if request_line:
            m = _REQ_LINE_RE.match(str(request_line))
            if m:
                event_dict.update(m.groupdict())

        if status is not None:
            event_dict["status_code"] = int(status)

        event_dict.setdefault("event", "http_access")
    except Exception as e:
        _logger.error("Failed to process uvicorn access log", exc_info=e)
    return event_dict


def rename_event_to_message(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    if "event" in event_dict and "message" not in event_dict:
        event_dict["message"] = event_dict.pop("event")
    return event_dict


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    pre_chain: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        uvicorn_access_to_structured,
        add_request_context,
        structlog.stdlib.add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", key="timestamp"),
    ]

    if settings.log_format == "json":
        tails: list[Processor] = [
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            rename_event_to_message,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ]
    else:
        tails = [
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=[
            *pre_chain,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structlog": {
                    "()": structlog.stdlib.ProcessorFormatter,
                    "foreign_pre_chain": pre_chain,
                    "processors": tails,
                },
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                    "formatter": "structlog",
                }
            },
            "loggers": {
                "": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.asgi": {"handlers": ["default"], "level": level, "propagate": False},
            },
        }
    )


def get_logger(name: str = "") -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
