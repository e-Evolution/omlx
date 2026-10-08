# SPDX-License-Identifier: Apache-2.0
"""One truthy set for the on/off environment switches.

``OMLX_MEMORY_PRESSURE_PARK``, ``OMLX_MEMORY_ABORT_SINGLE_VICTIM`` and
``OMLX_MEDIA_TEXT_SUFFIX_CHUNKED`` are on for ``true``, ``1``, ``yes`` or
``on`` (case and surrounding spaces ignored); anything else is off.
"""

TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def env_flag_on(raw: str | None) -> bool:
    """Whether an environment switch value is on (see ``TRUTHY_ENV_VALUES``)."""
    return (raw or "").strip().lower() in TRUTHY_ENV_VALUES
