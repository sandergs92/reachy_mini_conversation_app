import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class CreativeDirectionTool(Tool):
    """Update the creative session state for real-time creative direction."""

    name = "creative_direction_tool"
    description = (
        "Update the creative session state. Call this when you receive a "
        "creative direction update. Provide the scene, tone_and_style, or "
        "immediate_expression fields from the direction."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "scene": {
                "type": "string",
                "description": "The temporal and spatial context to embody.",
            },
            "tone_and_style": {
                "type": "string",
                "description": "Emotional quality, energy, and manner.",
            },
            "immediate_expression": {
                "type": "string",
                "description": "Short-term emotional or behavioral emphasis.",
            },
        },
        "required": ["scene", "tone_and_style", "immediate_expression"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        scene = kwargs.get("scene", "")
        tone_and_style = kwargs.get("tone_and_style", "")
        immediate_expression = kwargs.get("immediate_expression", "")

        deps.creative_session_state = {
            "scene": scene,
            "tone_and_style": tone_and_style,
            "immediate_expression": immediate_expression,
        }

        logger.info(
            "Tool call: creative_direction_tool updated — scene=%s, tone=%s, expression=%s",
            scene,
            tone_and_style,
            immediate_expression,
        )
        return {
            "status": "creative session state updated",
            "scene": scene,
            "tone_and_style": tone_and_style,
            "immediate_expression": immediate_expression,
        }
