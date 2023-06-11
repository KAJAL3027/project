from PIL import Image, ImageDraw

def generate_grid_image(width=800, height=800, grid_size=50):
    # Create a new transparent image with RGBA mode
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Draw vertical grid lines
    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=(255, 255, 255, 128), width=2)

    # Draw horizontal grid lines
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=(255, 255, 255, 128), width=2)

    return image

