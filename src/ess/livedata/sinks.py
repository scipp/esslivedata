# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
import structlog

from .core import Message, MessageSink, compact_messages


class PlotToPngSink(MessageSink[sc.DataArray]):
    def __init__(self) -> None:
        self.logger = structlog.get_logger()

    def publish_messages(self, messages: Message[sc.DataArray]) -> None:
        for msg in compact_messages(messages):
            title = f"{msg.stream.kind} - {msg.stream.name}"
            # Normalize the source name to be a valid filename and not a directory
            filename = f"{msg.stream.kind}_{msg.stream.name.replace('/', '_')}.png"
            try:
                msg.value.plot(title=title).save(filename)
            except Exception:
                self.logger.exception("Plotting to PNG failed")
                pass
