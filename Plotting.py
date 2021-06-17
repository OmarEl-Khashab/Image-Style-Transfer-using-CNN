import matplotlib.pyplot as plt
from PIL import Image
import os

# content_image = os.path.join("D:\Style Transfer", "mypic.jpeg")
# style_image = os.path.join("D:\Style Transfer", "Vincent_van_Gogh_-_Self-Portrait_Art_Project.jpg")
content_image = os.path.join("/home/abdalla/Omar/Style Transfer", "mypic.jpeg")
style_image = os.path.join("/home/abdalla/Omar/Style Transfer", "Vincent_van_Gogh_-_Self-Portrait_Art_Project.jpg")
Art = os.path.join("/home/abdalla/Omar/Style Transfer", "gen.png")
img1 = Image.open(content_image).resize((512, 512))
img2 = Image.open(style_image).resize((512, 512))
img3 = Image.open(Art)

plt.figure()

plt.subplot(1, 3, 1)
plt.title("My Pic")
plt.imshow(img1)

plt.subplot(1, 3, 2)
plt.title("Van Gogh Portrait")
plt.imshow(img2)

plt.subplot(1, 3, 3)
plt.title("Art")
plt.imshow(img3)

plt.show()
