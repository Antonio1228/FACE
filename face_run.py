from PIL import Image
import matplotlib.pyplot as plt
from facefuncs import get_path_return_output

image_path1 = "human_picture\\000100.jpg"
highest_score_image, highest_score_image_path = get_path_return_output(
    image_path1)
img = Image.open(highest_score_image_path)
plt.imshow(img)
plt.axis('off')
plt.title(f"Most similar image: {highest_score_image}")
plt.show()
