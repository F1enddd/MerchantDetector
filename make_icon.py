
import cairosvg
from PIL import Image
from io import BytesIO

svg_path = "Frontend/favicon.svg"

sizes = [16, 24, 32, 48, 64, 128, 256]

images = []

for size in sizes:
    png_data = cairosvg.svg2png(
        url=svg_path,
        output_width=size,
        output_height=size
    )

    image = Image.open(BytesIO(png_data)).convert("RGBA")
    images.append(image)

images[-1].save(
    "icon.ico",
    format="ICO",
    append_images=images[:-1],
    sizes=[(s, s) for s in sizes]
)

print("Готово! Создан icon.ico")
