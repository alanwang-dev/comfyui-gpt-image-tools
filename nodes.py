"""
comfyui_gpt_image_tools — Two custom nodes:

  1. GPTImageTurnaroundGenerator
       Calls the OpenAI gpt-image-2 API to produce a multi-view character
       turnaround reference sheet in a single API request.

  2. ESRGANUpscalerNode
       Loads any ESRGAN / Real-ESRGAN model from models/upscale_models/ and
       upscales the input image with automatic OOM-retry tiling.
"""

import base64
import io
import json
import math
import os
import urllib.request

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.utils
import folder_paths


def _resolve_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if key.lower().startswith("env:"):
        env_name = key.split(":", 1)[1].strip()
        if env_name:
            key = os.environ.get(env_name, "").strip()
        else:
            key = ""
    else:
        key = key or os.environ.get("OPENAI_API_KEY", "").strip()

    if not key:
        return ""

    # Prevent opaque httpx ascii-header crashes by validating early.
    try:
        key.encode("ascii")
    except UnicodeEncodeError as e:
        raise ValueError(
            "OPENAI_API_KEY contains non-ASCII characters. "
            "Please provide a valid key (starts with sk-)."
        ) from e

    if not key.startswith("sk-"):
        raise ValueError("OPENAI_API_KEY format is invalid. Expected key starting with 'sk-'.")

    return key


def _decode_generated_image(result) -> Image.Image:
    b64_json = getattr(result, "b64_json", None)
    if b64_json:
        img_bytes = base64.b64decode(b64_json)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    image_url = getattr(result, "url", None)
    if image_url:
        with urllib.request.urlopen(image_url) as resp:
            img_bytes = resp.read()
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    raise ValueError("OpenAI image response did not include image data.")


# ═══════════════════════════════════════════════════════════════════════════════
# Node 1 — GPT Image Turnaround Generator
# ═══════════════════════════════════════════════════════════════════════════════

_VIEWS_DEFAULT = (
    "front, back, left side, right side, front-left 45°, front-right 45°"
)


class GPTImageTurnaroundGenerator:
    """
    Single-call GPT Image 2 turnaround sheet.

    Builds a structured prompt that asks the model to lay out N character
    views in a grid, computes valid image dimensions automatically, and
    returns the result as a ComfyUI IMAGE tensor.
    """

    # ── gpt-image-2 hard limits ───────────────────────────────────────────────
    _MAX_PX   = 8_294_400   # 8.29 MP
    _MIN_PX   = 655_360
    _MAX_EDGE = 3_840
    _MAX_RATIO = 3.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A female warrior in ornate silver plate armor, "
                        "red braided hair, confident upright stance, "
                        "fantasy RPG style"
                    ),
                    "tooltip": "Describe the character's appearance, clothing, and style.",
                }),
                "views": ("STRING", {
                    "multiline": False,
                    "default": _VIEWS_DEFAULT,
                    "tooltip": (
                        "Comma-separated view labels, left→right then top→bottom. "
                        "Example: front, back, left side, right side"
                    ),
                }),
                "cols": ("INT", {
                    "default": 3, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Number of columns in the grid.",
                }),
                "tile_width": ("INT", {
                    "default": 1024, "min": 512, "max": 1536, "step": 16,
                    "tooltip": (
                        "Width of each view cell (px). "
                        "Total image size is auto-clamped to API limits."
                    ),
                }),
                "tile_height": ("INT", {
                    "default": 1024, "min": 512, "max": 1536, "step": 16,
                    "tooltip": "Height of each view cell (px).",
                }),
                "background": ("STRING", {
                    "default": "pure white background, no ground shadow",
                    "multiline": False,
                }),
                "quality": (["high", "medium", "low", "auto"],),
                "model": (["gpt-image-2", "gpt-image-1.5", "gpt-image-1"],),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "OpenAI API key. "
                        "Leave blank to use the OPENAI_API_KEY environment variable."
                    ),
                }),
            },
            "optional": {
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Appended verbatim at the end of the generated prompt.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("turnaround_sheet", "prompt_used")
    FUNCTION      = "generate"
    CATEGORY      = "GPT Image"
    OUTPUT_NODE   = False

    # ── size helpers ─────────────────────────────────────────────────────────

    def _compute_total_size(self, cols: int, rows: int,
                            tile_w: int, tile_h: int) -> tuple[int, int]:
        """
        Scale tile dimensions down (proportionally) until the total image
        satisfies all gpt-image-2 constraints.
        Both returned values are multiples of 16.
        """
        # Start from snapped tile values
        tw = max(16, (tile_w // 16) * 16)
        th = max(16, (tile_h // 16) * 16)

        scale = 1.0
        total_w = cols * tw
        total_h = rows * th

        if total_w > self._MAX_EDGE:
            scale = min(scale, self._MAX_EDGE / total_w)
        if total_h > self._MAX_EDGE:
            scale = min(scale, self._MAX_EDGE / total_h)

        total_px = total_w * total_h
        if total_px > self._MAX_PX:
            scale = min(scale, math.sqrt(self._MAX_PX / total_px))

        ratio = max(total_w, total_h) / min(total_w, total_h)
        if ratio > self._MAX_RATIO:
            longer = max(total_w, total_h)
            shorter = min(total_w, total_h)
            scale = min(scale, (self._MAX_RATIO * shorter) / longer)

        if scale < 1.0:
            tw = max(16, (int(tw * scale) // 16) * 16)
            th = max(16, (int(th * scale) // 16) * 16)
            total_w = cols * tw
            total_h = rows * th

        return total_w, total_h

    # ── prompt builder ───────────────────────────────────────────────────────

    def _build_prompt(self, view_list: list[str], cols: int, rows: int,
                      total_w: int, total_h: int, tile_w: int, tile_h: int,
                      background: str, character_description: str,
                      extra: str) -> str:
        n = len(view_list)

        row_lines = []
        for r in range(rows):
            chunk = view_list[r * cols: r * cols + cols]
            row_lines.append(f"  Row {r + 1}: {',  '.join(chunk)}")

        lines = [
            f"Game character turnaround reference sheet — {n} views arranged in a"
            f" {cols}-column × {rows}-row grid.",
            f"Canvas: {total_w}×{total_h} px  |  each cell: {tile_w}×{tile_h} px  |  no separator gaps",
            "",
            "Grid layout (left to right, top to bottom):",
            *row_lines,
            "",
            f"Background: {background}.",
            "",
            "Character:",
            f"  {character_description}",
            "",
            "Strict requirements:",
            "  • Every cell shows the SAME character: identical colors, proportions,"
            " silhouette, and clothing detail — only the camera angle differs.",
            "  • Each view shows the full body from head to feet, no cropping.",
            "  • Flat or soft uniform studio lighting, no cast shadows on the floor.",
            "  • No text labels, no cell borders, no decorative background elements.",
            "  • All cells are equal in size and perfectly aligned to a grid.",
            "  • Style: clean digital game concept art / character sheet.",
        ]

        if extra.strip():
            lines += ["", "Additional instructions:", f"  {extra.strip()}"]

        return "\n".join(lines)

    # ── main entry ───────────────────────────────────────────────────────────

    def generate(self, character_description: str, views: str,
                 cols: int, tile_width: int, tile_height: int,
                 background: str, quality: str, model: str, api_key: str,
                 extra_instructions: str = "") -> tuple:

        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "The 'openai' package is not installed in this Python environment. "
                "Run:  pip install openai"
            )

        # Parse views
        view_list = [v.strip() for v in views.split(",") if v.strip()]
        if not view_list:
            raise ValueError("'views' must contain at least one view angle.")

        n_views = len(view_list)
        cols = min(cols, n_views)
        rows = math.ceil(n_views / cols)

        total_w, total_h = self._compute_total_size(cols, rows, tile_width, tile_height)
        size_str = f"{total_w}x{total_h}"

        tile_w = total_w // cols
        tile_h = total_h // rows
        prompt = self._build_prompt(
            view_list, cols, rows, total_w, total_h, tile_w, tile_h,
            background, character_description, extra_instructions
        )

        # Resolve API key
        key = _resolve_api_key(api_key)
        if not key:
            raise ValueError(
                "No API key found. Provide it via the 'api_key' input "
                "or set the OPENAI_API_KEY environment variable."
            )

        print(
            f"[GPTImageTurnaround] model={model}  size={size_str}  "
            f"quality={quality}  views={n_views} ({cols}×{rows} grid)"
        )

        client = OpenAI(api_key=key)
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size_str,
            quality=quality,
            n=1,
        )

        pil_img = _decode_generated_image(response.data[0])

        arr    = np.array(pil_img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)       # [1, H, W, 3]

        print(f"[GPTImageTurnaround] Done — output {total_w}×{total_h}")
        return (tensor, prompt)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 2 — ESRGAN Upscaler (load + upscale in a single node)
# ═══════════════════════════════════════════════════════════════════════════════

class ESRGANUpscalerNode:
    """
    All-in-one ESRGAN / Real-ESRGAN upscaler.

    Loads any model found in   ComfyUI/models/upscale_models/
    and upscales the input image with tiled inference.
    Automatically retries with a smaller tile on GPU OOM.

    Recommended models to download:
      • 4x-UltraSharp.pth         — best for game/concept art, sharp edges
      • 4x-AnimeSharp.pth         — anime/stylized characters
      • RealESRGAN_x4plus_anime_6B.pth — fast, soft anime upscaling
    Place them in:  ComfyUI/models/upscale_models/
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_list = folder_paths.get_filename_list("upscale_models")
        if not model_list:
            model_list = ["(no models — place .pth files in models/upscale_models/)"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list,),
                "tile_size": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 64,
                    "tooltip": (
                        "Inference tile size (px). Smaller = less VRAM. "
                        "512 is safe for 8 GB GPU. 0 = no tiling (fastest, most VRAM)."
                    ),
                }),
                "tile_overlap": ("INT", {
                    "default": 32, "min": 8, "max": 256, "step": 8,
                    "tooltip": "Tile overlap to avoid seam artifacts. 32 is a good default.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("upscaled_image", "info")
    FUNCTION      = "upscale"
    CATEGORY      = "GPT Image"

    def upscale(self, image: torch.Tensor, model_name: str,
                tile_size: int, tile_overlap: int) -> tuple:

        try:
            from spandrel import ModelLoader
        except ImportError:
            raise RuntimeError(
                "spandrel is not installed. It is normally bundled with ComfyUI — "
                "ensure your ComfyUI installation is up to date."
            )

        # ── load model ───────────────────────────────────────────────────────
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        if model_path is None:
            raise FileNotFoundError(
                f"Model '{model_name}' not found in models/upscale_models/. "
                "Download it and place the .pth file in that folder."
            )

        sd = comfy.utils.load_torch_file(model_path, safe_load=True)

        # Some SwinIR / HAT models are saved with a 'module.' prefix
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        upscale_model = ModelLoader().load_from_state_dict(sd).eval()

        # ── VRAM management ──────────────────────────────────────────────────
        device = comfy.model_management.get_torch_device()
        mem_needed = (
            comfy.model_management.module_size(upscale_model.model)
            + image.nelement() * image.element_size()
            * max(upscale_model.scale, 1.0) ** 2
            * 1.5   # safety headroom
        )
        comfy.model_management.free_memory(mem_needed, device)
        upscale_model.to(device)

        # ── tiled inference with OOM-retry ───────────────────────────────────
        in_img = image.movedim(-1, -3).to(device)   # [B, C, H, W]
        tile   = tile_size if tile_size > 0 else max(image.shape[1], image.shape[2])
        oom    = True

        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2],
                    tile_x=tile, tile_y=tile, overlap=tile_overlap,
                )
                pbar = comfy.utils.ProgressBar(steps)
                out  = comfy.utils.tiled_scale(
                    in_img,
                    lambda x: upscale_model(x),
                    tile_x=tile, tile_y=tile,
                    overlap=tile_overlap,
                    upscale_amount=upscale_model.scale,
                    pbar=pbar,
                )
                oom = False
            except comfy.model_management.OOM_EXCEPTION as err:
                tile //= 2
                print(f"[ESRGANUpscaler] OOM — retrying with tile_size={tile}")
                if tile < 64:
                    raise RuntimeError(
                        "GPU out of memory even at tile_size=64. "
                        "Try a smaller input image or switch to CPU inference."
                    ) from err

        upscale_model.cpu()

        out_img = out.movedim(-3, -1).clamp(0.0, 1.0).float()   # [B, H, W, C]

        h_in,  w_in  = image.shape[1],   image.shape[2]
        h_out, w_out = out_img.shape[1], out_img.shape[2]
        info = (
            f"{w_in}×{h_in}  →  {w_out}×{h_out}  "
            f"(×{upscale_model.scale}  |  {model_name})"
        )
        print(f"[ESRGANUpscaler] {info}")

        return (out_img, info)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 3 — GPT Image Multi-View Sheet  (3×2 grid, Hy3D-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class GPTImageMultiViewSheet:
    """
    Single gpt-image-2 call → 3×2 grid of 6 character views.

    Grid layout (matches Hy3DBakeFromMultiview default camera order:
    azims=[0,90,180,270,0,180]  elevs=[0,0,0,0,90,-90]):

    ┌──────────┬──────────┬──────────┐
    │  Front   │  Right   │  Back    │  Row 0
    │  az=0°   │  az=90°  │  az=180° │
    ├──────────┼──────────┼──────────┤
    │  Left    │  Top     │  Bottom  │  Row 1
    │  az=270° │  el=+90° │  el=-90° │
    └──────────┴──────────┴──────────┘

    Returns the full sheet image, the exact prompt text, and a JSON string
    with pixel crop coordinates for each of the 6 tiles.
    """

    # Order matches Hy3DBakeFromMultiview defaults exactly
    _VIEWS = [
        ("front",  "FRONT VIEW — character faces directly toward camera (azimuth 0°, elevation 0°)",        0,   0),
        ("right",  "RIGHT SIDE — camera to the character's right (azimuth 90°, elevation 0°)",            90,   0),
        ("back",   "BACK VIEW  — character faces directly away from camera (azimuth 180°, elevation 0°)", 180,   0),
        ("left",   "LEFT SIDE  — camera to the character's left (azimuth 270°, elevation 0°)",           270,   0),
        ("top",    "TOP-DOWN   — camera directly overhead, looking straight down (elevation +90°)",         0,  90),
        ("bottom", "BOTTOM-UP  — camera directly below, looking straight up (elevation -90°)",           180, -90),
    ]

    _MAX_PX    = 8_294_400
    _MAX_EDGE  = 3_840
    _MAX_RATIO = 3.0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A fantasy warrior in full-body A-pose:\n"
                        "- Silver plate armor with blue gemstone chest piece\n"
                        "- Red flowing cape, slightly tattered at the hem\n"
                        "- One-handed broadsword sheathed at left hip\n"
                        "- Closed-toe armored boots, knee-high with brass clasps\n"
                        "- Human male, medium athletic build, approximately 180 cm tall\n"
                        "- Short dark brown hair, clean-shaven face, blue eyes\n"
                        "- Leather belt with two small pouches on right side\n"
                        "- Dark navy tunic visible at collar and wrists"
                    ),
                    "tooltip": (
                        "Detailed physical description of the character. "
                        "Be specific: colors, materials, accessories, proportions."
                    ),
                }),
                "art_style": ("STRING", {
                    "multiline": False,
                    "default": (
                        "game-ready PBR character concept art, semi-realistic, "
                        "clean digital illustration, uniform soft studio lighting"
                    ),
                    "tooltip": "Art direction appended after the character description.",
                }),
                "tile_size": ("INT", {
                    "default": 1152, "min": 256, "max": 1536, "step": 16,
                    "tooltip": (
                        "Each view cell is tile_size × tile_size pixels. "
                        "1152 is the API maximum for a 3×2 grid with gap=32 "
                        "(canvas 3520×2336, just under the 8.29 MP limit). "
                        "Total canvas is auto-clamped to gpt-image-2 limits."
                    ),
                }),
                "gap_size": ("INT", {
                    "default": 32, "min": 0, "max": 128, "step": 16,
                    "tooltip": (
                        "White separator strip between cells (px). "
                        "Must be a multiple of 16. 32 ≈ 20px visually. "
                        "Use these pixel offsets when cropping individual views."
                    ),
                }),
                "quality": (["high", "medium", "low", "auto"], {"default": "high"}),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "OpenAI API key. Leave blank to use OPENAI_API_KEY env var.",
                }),
            },
            "optional": {
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Additional requirements appended verbatim to the prompt.",
                }),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES  = ("sheet_image", "prompt_used", "crop_coords_json")
    FUNCTION      = "generate"
    CATEGORY      = "GPT Image"
    OUTPUT_NODE   = False

    def _clamp_to_api_limits(self, w: int, h: int) -> tuple[int, int]:
        """Scale down proportionally until within gpt-image-2 constraints."""
        scale = 1.0
        if w > self._MAX_EDGE:
            scale = min(scale, self._MAX_EDGE / w)
        if h > self._MAX_EDGE:
            scale = min(scale, self._MAX_EDGE / h)
        if w * h > self._MAX_PX:
            scale = min(scale, math.sqrt(self._MAX_PX / (w * h)))
        ratio = max(w, h) / max(min(w, h), 1)
        if ratio > self._MAX_RATIO:
            scale = min(scale, self._MAX_RATIO * min(w, h) / max(w, h))
        if scale < 1.0:
            w = max(16, (int(w * scale) // 16) * 16)
            h = max(16, (int(h * scale) // 16) * 16)
        return w, h

    def generate(self, character_description, art_style, tile_size, gap_size,
                 quality, api_key, extra_instructions=""):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Run:  pip install openai")

        # Snap to multiples of 16 (API requirement)
        tile_size = max(16, (tile_size // 16) * 16)
        gap_size  = (gap_size // 16) * 16

        raw_w = 3 * tile_size + 2 * gap_size   # 3 columns
        raw_h = 2 * tile_size + 1 * gap_size   # 2 rows
        total_w, total_h = self._clamp_to_api_limits(raw_w, raw_h)

        # Recalculate if clamped
        if total_w != raw_w or total_h != raw_h:
            scale     = min(total_w / raw_w, total_h / raw_h)
            tile_size = max(16, (int(tile_size * scale) // 16) * 16)
            gap_size  = max(0,  (int(gap_size  * scale) // 16) * 16)
            total_w   = 3 * tile_size + 2 * gap_size
            total_h   = 2 * tile_size + 1 * gap_size

        # Build per-tile crop coordinates
        tiles_info = []
        for i, (name, desc, azim, elev) in enumerate(self._VIEWS):
            col = i % 3
            row = i // 3
            tiles_info.append({
                "name":     name,
                "label":    desc,
                "azim_deg": azim,
                "elev_deg": elev,
                "col": col, "row": row,
                "x": col * (tile_size + gap_size),
                "y": row * (tile_size + gap_size),
                "w": tile_size, "h": tile_size,
            })

        crop_json = json.dumps({
            "cols": 3, "rows": 2,
            "tile_w": tile_size, "tile_h": tile_size,
            "gap":    gap_size,
            "total_w": total_w, "total_h": total_h,
            "tiles":  tiles_info,
        }, indent=2)

        # Build prompt
        view_block = "\n".join(
            f"  [Row {t['row']}, Col {t['col']}]  {t['label']}"
            for t in tiles_info
        )
        prompt = (
            f"CHARACTER TURNAROUND REFERENCE SHEET — 6 ORTHOGRAPHIC VIEWS\n"
            f"Canvas: {total_w}\u00d7{total_h} px  |  3 columns \u00d7 2 rows\n"
            f"Each view cell: {tile_size}\u00d7{tile_size} px  "
            f"|  White separator gap: {gap_size}px\n"
            f"\nCELL CONTENTS:\n{view_block}\n"
            f"\nCHARACTER:\n{character_description}\n"
            f"\nSTYLE: {art_style}\n"
            f"\nMANDATORY REQUIREMENTS:\n"
            f"\u2022 POSE: A-pose (arms angled ~45\u00b0 down from shoulders) in ALL 6 views\n"
            f"\u2022 BACKGROUND: pure white (#FFFFFF) in every cell \u2014 no gradients, shadows, or floor plane\n"
            f"\u2022 FULL BODY: entire character head-to-toe visible in every cell, no clipping\n"
            f"\u2022 CONSISTENCY: identical colors, proportions, costume details in all 6 cells \u2014 only angle differs\n"
            f"\u2022 GAPS: separator strips must be pure white (#FFFFFF), not gray or textured\n"
            f"\u2022 NO text, labels, cell borders, watermarks, or decorative backgrounds\n"
            f"\u2022 Character fills ~80% of each cell height, centered with equal top/bottom margin\n"
            f"\u2022 Soft diffuse studio lighting, minimal cast shadows, consistent across all views\n"
        )
        if extra_instructions.strip():
            prompt += f"\nADDITIONAL:\n{extra_instructions.strip()}\n"

        key = _resolve_api_key(api_key)
        if not key:
            raise ValueError(
                "No API key. Provide via api_key input or set OPENAI_API_KEY env var."
            )

        print(
            f"[GPTImageMultiViewSheet] gpt-image-2  {total_w}\u00d7{total_h}  "
            f"quality={quality}  tile={tile_size}  gap={gap_size}"
        )

        client = OpenAI(api_key=key)
        response = client.images.generate(
            model="gpt-image-2",
            prompt=prompt,
            size=f"{total_w}x{total_h}",
            quality=quality,
            n=1,
        )

        pil_img   = _decode_generated_image(response.data[0])
        arr       = np.array(pil_img, dtype=np.float32) / 255.0
        tensor    = torch.from_numpy(arr).unsqueeze(0)   # [1, H, W, 3]

        print(f"[GPTImageMultiViewSheet] Done \u2014 {total_w}\u00d7{total_h}")
        return (tensor, prompt, crop_json)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 4 — GPT Image Multi-View Sheet Cropper
# ═══════════════════════════════════════════════════════════════════════════════

class GPTImageMultiViewCropper:
    """
    Splits a 3×2 multi-view sheet (from GPTImageMultiViewSheet) into 6
    individual IMAGE tensors using the crop_coords_json metadata.

    Output slot order: front, right, back, left, top, bottom
    — matches Hy3DGenerateMeshMultiView (front/left/right/back) and
      Hy3DBakeFromMultiview (front/right/back/left/top/bottom) camera order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sheet_image":      ("IMAGE",),
                "crop_coords_json": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("front", "right", "back", "left", "top", "bottom")
    FUNCTION      = "crop"
    CATEGORY      = "GPT Image"
    OUTPUT_NODE   = False

    def crop(self, sheet_image: torch.Tensor, crop_coords_json: str):
        coords = json.loads(crop_coords_json)
        tile_lookup = {t["name"]: t for t in coords["tiles"]}

        # sheet_image: [B, H, W, 3]  — B=1 from GPTImageMultiViewSheet
        img = sheet_image[0]   # [H, W, 3]

        def cut(name: str) -> torch.Tensor:
            t = tile_lookup[name]
            x, y, w, h = t["x"], t["y"], t["w"], t["h"]
            tile = img[y : y + h, x : x + w, :]   # [h, w, 3]
            return tile.unsqueeze(0)                # [1, h, w, 3]

        front  = cut("front")
        right  = cut("right")
        back   = cut("back")
        left   = cut("left")
        top    = cut("top")
        bottom = cut("bottom")

        print(
            f"[GPTImageMultiViewCropper] "
            f"tile={coords.get('tile_w')}×{coords.get('tile_h')}  "
            f"gap={coords.get('gap')}  "
            f"sheet={coords.get('total_w')}×{coords.get('total_h')}"
        )

        return (front, right, back, left, top, bottom)


# ═══════════════════════════════════════════════════════════════════════════════
# Node 5 — Universal LLM Node (OpenAI / Gemini / Anthropic)
# ═══════════════════════════════════════════════════════════════════════════════

class UniversalLLMNode:
    """
    Single node that calls OpenAI, Google Gemini, or Anthropic Claude with a
    text prompt and an optional vision image.

    Provider routing:
      openai    — openai.OpenAI client → api.openai.com
      gemini    — openai.OpenAI client → Google's OpenAI-compatible endpoint
      anthropic — anthropic.Anthropic client → api.anthropic.com

    API key resolution (in priority order):
      1. Explicit value in the api_key field
      2. "env:VAR_NAME" syntax → reads that env variable
      3. Empty field → auto-reads OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY
    """

    CATEGORY = "GPT Image Tools"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("response",)
    OUTPUT_TOOLTIPS = ("The model's text response.",)
    FUNCTION = "run"

    _ENV_DEFAULTS: dict[str, str] = {
        "openai":    "OPENAI_API_KEY",
        "gemini":    "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }
    _GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "provider": (["openai", "gemini", "anthropic"],),
                "model": ("STRING", {
                    "default": "gpt-4o",
                    "multiline": False,
                    "tooltip": (
                        "Model name for the selected provider. "
                        "Examples: gpt-4o, gemini-2.0-flash, claude-opus-4-5"
                    ),
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a helpful assistant.",
                    "multiline": True,
                }),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "User prompt. Overridden by prompt_override input when connected.",
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "API key, or 'env:VAR_NAME' to read from an env variable. "
                        "Leave blank to auto-read OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY."
                    ),
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 1, "max": 32768, "step": 64,
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional vision image (first frame used). Not all models support vision.",
                }),
                "prompt_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "When connected, replaces the prompt widget value.",
                }),
            },
        }

    # ── internal helpers ─────────────────────────────────────────────────────

    def _resolve_key(self, api_key: str, provider: str) -> str:
        key = (api_key or "").strip()
        if key.lower().startswith("env:"):
            key = os.environ.get(key[4:].strip(), "").strip()
        if not key:
            key = os.environ.get(self._ENV_DEFAULTS.get(provider, ""), "").strip()
        if not key:
            raise ValueError(
                f"No API key for provider '{provider}'. "
                f"Set {self._ENV_DEFAULTS.get(provider, 'the API key env var')} "
                f"or fill the api_key field."
            )
        return key

    @staticmethod
    def _image_to_b64(image_tensor) -> str:
        arr = (image_tensor[0].cpu().numpy() * 255).clip(0, 255).astype("uint8")
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ── provider backends ────────────────────────────────────────────────────

    def _call_openai_compat(
        self,
        api_key: str,
        base_url: str,
        model: str,
        system_prompt: str,
        prompt: str,
        image_b64: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)

        user_content: list = []
        if image_b64:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            })
        user_content.append({"type": "text", "text": prompt})

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        prompt: str,
        image_b64: str | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for Anthropic/Claude. "
                "Install it with: pip install anthropic"
            ) from exc

        client = Anthropic(api_key=api_key)

        user_content: list = []
        if image_b64:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64,
                },
            })
        user_content.append({"type": "text", "text": prompt})

        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            temperature=temperature,
            messages=[{"role": "user", "content": user_content}],
        )
        return msg.content[0].text

    # ── entry point ──────────────────────────────────────────────────────────

    def run(
        self,
        provider: str,
        model: str,
        system_prompt: str,
        prompt: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        image=None,
        prompt_override: str | None = None,
    ) -> tuple[str]:
        actual_prompt = (prompt_override or "").strip() or prompt.strip()
        if not actual_prompt:
            raise ValueError(
                "UniversalLLMNode: prompt is empty. "
                "Fill the prompt widget or connect a prompt_override."
            )

        key       = self._resolve_key(api_key, provider)
        image_b64 = self._image_to_b64(image) if image is not None else None

        if provider == "openai":
            response = self._call_openai_compat(
                api_key=key, base_url="https://api.openai.com/v1",
                model=model, system_prompt=system_prompt, prompt=actual_prompt,
                image_b64=image_b64, temperature=temperature, max_tokens=max_tokens,
            )
        elif provider == "gemini":
            response = self._call_openai_compat(
                api_key=key, base_url=self._GEMINI_BASE_URL,
                model=model, system_prompt=system_prompt, prompt=actual_prompt,
                image_b64=image_b64, temperature=temperature, max_tokens=max_tokens,
            )
        elif provider == "anthropic":
            response = self._call_anthropic(
                api_key=key, model=model, system_prompt=system_prompt, prompt=actual_prompt,
                image_b64=image_b64, temperature=temperature, max_tokens=max_tokens,
            )
        else:
            raise ValueError(f"UniversalLLMNode: unknown provider {provider!r}")

        return (response,)


# ═══════════════════════════════════════════════════════════════════════════════
# ComfyUI registration
# ═══════════════════════════════════════════════════════════════════════════════

NODE_CLASS_MAPPINGS = {
    "GPTImageTurnaroundGenerator": GPTImageTurnaroundGenerator,
    "ESRGANUpscalerNode":          ESRGANUpscalerNode,
    "GPTImageMultiViewSheet":      GPTImageMultiViewSheet,
    "GPTImageMultiViewCropper":    GPTImageMultiViewCropper,
    "UniversalLLMNode":            UniversalLLMNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPTImageTurnaroundGenerator": "GPT Image: Turnaround Generator",
    "ESRGANUpscalerNode":          "ESRGAN Upscaler",
    "GPTImageMultiViewSheet":      "GPT Image: Multi-View Sheet (6-view Hy3D)",
    "GPTImageMultiViewCropper":    "GPT Image: Multi-View Cropper",
    "UniversalLLMNode":            "Universal LLM (OpenAI / Gemini / Anthropic)",
}
