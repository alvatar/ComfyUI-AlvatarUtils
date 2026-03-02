"""
Continue - Synchronization barrier with dynamic passthrough channels.

Behavior:
- Starts with 2 required inputs: input1, input2.
- As additional input sockets are added by the frontend extension (input3, input4, ...),
  this node accepts them dynamically and passes them through in order.
- Ensures a dependency checkpoint so all connected upstream branches complete before
  downstream execution continues.

Implementation notes:
- Dynamic sockets in ComfyUI require a hybrid approach:
  1) Frontend JS mutates visible inputs/outputs as links are made/removed.
  2) Backend INPUT_TYPES must tolerate dynamic input names during validation.
- This file implements the backend half and follows the same validation bypass pattern
  used by ImpactSwitch-style dynamic nodes.
"""

from __future__ import annotations

import inspect
import re


class AnyType(str):
    """Wildcard type accepted by ComfyUI type checks."""

    def __ne__(self, __value: object) -> bool:  # noqa: D401
        return False


class TautologyStr(str):
    """String type that is equal to any other string for output type indexing."""

    def __ne__(self, other):  # noqa: D401
        return False


class ByPassTypeTuple(tuple):
    """Tuple that repeats index 0 type for any output index > 0."""

    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


any_typ = AnyType("*")
_INPUT_RE = re.compile(r"^input(\d+)$")


class _DynamicInputContainer:
    """Validation bypass container for dynamic inputN slots (N >= 3)."""

    def __contains__(self, item):
        if not isinstance(item, str) or not item.startswith("input"):
            return False
        suffix = item[5:]
        return suffix.isdigit() and int(suffix) >= 3

    def __getitem__(self, key):
        return any_typ, {"lazy": True}


class Continue:
    """Dynamic synchronization barrier with passthrough outputs."""

    @classmethod
    def INPUT_TYPES(cls):
        # Base UI shape: 2 required inputs + one optional seed for dynamic growth.
        optional_inputs = {
            "input3": (any_typ, {
                "lazy": True,
                "tooltip": "Dynamic passthrough input. Additional inputs are added automatically as you connect slots."
            }),
        }

        # During backend input-info validation, allow any inputN >= 3.
        # This mirrors ImpactSwitch-style dynamic slot validation bypass.
        if any(frame.function == "get_input_info" for frame in inspect.stack()):
            optional_inputs = _DynamicInputContainer()

        return {
            "required": {
                "input1": (any_typ, {
                    "tooltip": "Passthrough input 1. Connect to enable synchronization barrier behavior."
                }),
                "input2": (any_typ, {
                    "tooltip": "Passthrough input 2. When both input1 and input2 are connected, a new input slot appears."
                }),
            },
            "optional": optional_inputs,
        }

    RETURN_TYPES = ByPassTypeTuple((any_typ, any_typ))
    RETURN_NAMES = ("output1", "output2")
    OUTPUT_TOOLTIPS = (
        "Passthrough of input1.",
        "Passthrough of input2.",
    )
    FUNCTION = "execute"
    CATEGORY = "Alvatar/Utils"
    DESCRIPTION = (
        "Synchronization barrier with dynamic inputs/outputs. "
        "Starts with 2 inputs, then grows as connections are made."
    )

    def execute(self, input1, input2, **kwargs):
        """
        Pass through inputs in slot order.

        Dynamic slots are expected as kwargs named input3, input4, ...
        Missing intermediate dynamic slots are returned as None to preserve index order.
        """
        dynamic_inputs = {}
        max_index = 2

        for key, value in kwargs.items():
            match = _INPUT_RE.match(key)
            if not match:
                continue
            idx = int(match.group(1))
            if idx < 3:
                continue
            dynamic_inputs[idx] = value
            if idx > max_index:
                max_index = idx

        outputs = [input1, input2]
        for idx in range(3, max_index + 1):
            outputs.append(dynamic_inputs.get(idx))

        return tuple(outputs)
