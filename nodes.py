

# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI registration
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "GPTImageTurnaroundGenerator": GPTImageTurnaroundGenerator,
    "ESRGANUpscalerNode":          ESRGANUpscalerNode,
    "GPTImageMultiViewSheet":      GPTImageMultiViewSheet,
    "GPTImageMultiViewCropper":    GPTImageMultiViewCropper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageTurnaroundGenerator": "GPT Image: Turnaround Generator",
    "ESRGANUpscalerNode":          "ESRGAN Upscaler",
    "GPTImageMultiViewSheet":      "GPT Image: Multi-View Sheet (6-view Hy3D)",
    "GPTImageMultiViewCropper":    "GPT Image: Multi-View Cropper",
}