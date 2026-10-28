"""Optional SVG-to-PNG/GIF exporter (system GI, Cairo, Rsvg and Pillow)."""

from pathlib import Path
import io
import cairo
import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg  # noqa: E402 — version must be selected first
from PIL import Image  # noqa: E402

root = Path(__file__).resolve().parents[1]


def raster(file):
    handle = Rsvg.Handle.new_from_file(str(file))
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1200, 620)
    context = cairo.Context(surface)
    viewport = Rsvg.Rectangle()
    viewport.x = 0
    viewport.y = 0
    viewport.width = 1200
    viewport.height = 620
    handle.render_document(context, viewport)
    output = io.BytesIO()
    surface.write_to_png(output)
    output.seek(0)
    return Image.open(output).convert("RGB")


snapshot = root / "artifacts/scenes/snapshot.svg"
if snapshot.exists():
    raster(snapshot).save(root / "docs/assets/driving-scene.png", optimize=True)
files = sorted((root / "artifacts/scenes").glob("frame-*.svg"))
if files:
    frames = [raster(file).resize((900, 465)) for file in files]
    frames[0].save(
        root / "docs/assets/driving-preview.gif",
        save_all=True,
        append_images=frames[1:],
        duration=[100] * (len(frames) - 1) + [1600],
        loop=0,
        optimize=True,
    )
print("Rasterized the actual scene renderer outputs.")
