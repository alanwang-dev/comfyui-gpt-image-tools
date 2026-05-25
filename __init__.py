"""
comfyui_gpt_image_tools
=======================
Custom ComfyUI nodes for GPT Image generation and ESRGAN upscaling.

Nodes provided:
  • GPT Image: Turnaround Generator  —  single gpt-image-2 call → multi-view sheet
  • ESRGAN Upscaler                  —  ESRGAN / Real-ESRGAN upscaling with tiled OOM-retry
"""

try:
    from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    _names = list(NODE_DISPLAY_NAME_MAPPINGS.values())
    print(f"\033[32m[GPT Image Tools] Loaded {len(_names)} node(s): {', '.join(_names)}\033[0m")

except Exception as _err:
    import traceback
    print(f"\033[31m[GPT Image Tools] Failed to load nodes: {_err}\033[0m")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS        = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
