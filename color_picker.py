from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import colorsys



def extract_dominant_color_with_priority(image_path, n_colors=5, image_resize=(100, 100)):
    image = Image.open(image_path).convert("RGB").resize(image_resize)
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)

    colors = kmeans.cluster_centers_.astype(int)

    def rgb_to_hsv(color):
        return colorsys.rgb_to_hsv(color[0] / 255, color[1] / 255, color[2] / 255)

    sorted_colors = sorted(
        colors,
        key=lambda c: (
            rgb_to_hsv(c)[1],
            -abs(c[0] - c[1]) - abs(c[1] - c[2]) - abs(c[0] - c[2]),
            rgb_to_hsv(c)[2]
        ),
        reverse=True
    )

    return tuple(map(int, sorted_colors[0]))