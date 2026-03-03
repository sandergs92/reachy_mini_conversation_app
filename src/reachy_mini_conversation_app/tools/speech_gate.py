import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class SpeechGateTool(Tool):
    """Decide whether to respond, stay silent, or process creative direction."""

    name = "speech_gate"
    description = (
        "Decide whether to speak, stay silent, or process creative direction. "
        "DEFAULT is 'silent'. Only choose 'respond' if someone is clearly "
        "talking to you (by name or obviously directing speech at you). "
        "Choose 'creative_direction' if a speaker is giving you creative "
        "direction (tone, style, scene changes). If unsure, choose 'silent'."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["silent", "respond", "creative_direction"],
                "description": (
                    "'silent' (DEFAULT) — speakers talking to each other, "
                    "monologuing, telling you to be quiet, mentioning you "
                    "in passing, or any ambiguity. "
                    "'respond' — someone is clearly talking TO you and "
                    "expects a reply. "
                    "'creative_direction' — a speaker is giving creative "
                    "direction (tone, style, scene, expression changes)."
                ),
            },
            "draft_response": {
                "type": "string",
                "description": "If decision is 'respond', what you want to say. Leave empty otherwise.",
            },
        },
        "required": ["decision"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        decision = kwargs.get("decision", "silent")
        draft_response = kwargs.get("draft_response", "")
        logger.info(
            "Tool call: speech_gate — decision=%s, draft=%s",
            decision,
            draft_response[:80] if draft_response else "",
        )
        return {
            "status": "ok",
            "decision": decision,
            "draft_response": draft_response,
        }
