# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""HTTP-based message source for polling remote services."""

import logging
from typing import Generic
from urllib.parse import urljoin

import requests

from ..core.message import Message, MessageSource, T
from .serialization import MessageSerializer


class HTTPMessageSource(MessageSource[Message[T]], Generic[T]):
    """
    Message source that polls messages from an HTTP endpoint.

    Polls a remote service's GET /messages endpoint and deserializes
    the response into Message objects.

    Parameters
    ----------
    base_url:
        Base URL of the service (e.g., "http://localhost:8000").
    serializer:
        Serializer to use for deserializing messages.
    endpoint:
        Endpoint path for getting messages. Defaults to "/messages".
    timeout:
        Timeout in seconds for HTTP requests.
    logger:
        Optional logger instance.
    """

    def __init__(
        self,
        base_url: str,
        serializer: MessageSerializer[T],
        endpoint: str = '/messages',
        timeout: float = 5.0,
        logger: logging.Logger | None = None,
    ):
        self._base_url = base_url
        self._serializer = serializer
        self._endpoint = endpoint
        self._timeout = timeout
        self._logger = logger or logging.getLogger(__name__)
        self._session = requests.Session()

    def get_messages(self) -> list[Message[T]]:
        """
        Poll the HTTP endpoint for messages.

        Returns
        -------
        :
            List of messages received from the endpoint. Empty list if
            request fails or no messages are available.
        """
        url = urljoin(self._base_url, self._endpoint)

        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()

            if response.status_code == 204:  # No content
                return []

            if not response.content:
                return []

            messages = self._serializer.deserialize(response.content)
            self._logger.debug("Received %d messages from %s", len(messages), url)
            return messages

        except requests.Timeout:
            self._logger.warning("Timeout polling %s", url)
            return []
        except requests.RequestException as e:
            self._logger.error("Error polling %s: %s", url, e)
            return []
        except Exception as e:
            self._logger.exception("Unexpected error deserializing messages: %s", e)
            return []

    def close(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close session."""
        self.close()


class MultiHTTPSource(MessageSource[Message[T]], Generic[T]):
    """
    Message source that combines multiple HTTP sources.

    Polls multiple HTTP endpoints and combines their messages. Useful for
    consuming from multiple upstream services (e.g., raw data + config).

    Parameters
    ----------
    sources:
        List of HTTP message sources to poll
    """

    def __init__(self, sources: list[HTTPMessageSource[T]]):
        self._sources = sources

    def get_messages(self) -> list[Message[T]]:
        """
        Poll all HTTP sources and combine messages.

        Returns
        -------
        :
            Combined list of messages from all sources
        """
        messages = []
        for source in self._sources:
            messages.extend(source.get_messages())
        return messages

    def close(self) -> None:
        """Close all HTTP sessions."""
        for source in self._sources:
            source.close()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close all sessions."""
        self.close()
