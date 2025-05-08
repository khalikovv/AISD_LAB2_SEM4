from PIL import Image
import numpy as np

def generate_nontrivial_color_image(width=2048, height=2048):
    # Create an empty image array
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Generate colorful gradient pattern with sine waves and color variations
    for y in range(height):
        for x in range(width):
            r = int((np.sin(x * 0.01) + 1) * 127.5)
            g = int((np.sin(y * 0.01) + 1) * 127.5)
            b = int((np.sin((x + y) * 0.01) + 1) * 127.5)
            img_array[y, x] = [r, g, b]

    # Convert to PIL Image and return
    img = Image.fromarray(img_array, 'RGB')
    return img

def convert_to_grayscale(image):
    return image.convert('L')

def convert_to_bw_dithered(image):
    return image.convert('1')

def convert_to_bw(image):
    # Convert to grayscale first
    gray = image.convert('L')
    # Apply simple threshold to convert to black and white without dithering
    bw = gray.point(lambda x: 255 if x > 128 else 0, mode='1')
    return bw

if __name__ == "__main__":
    img = generate_nontrivial_color_image()
    img.save("test/test_image.png")
    print("Non-trivial color image generated and saved as test/test_image.png")

    gray_img = convert_to_grayscale(img)
    gray_img.save("test/test_image_grayscale.png")
    print("Grayscale image saved as test/test_image_grayscale.png")

    bw_dithered_img = convert_to_bw_dithered(img)
    bw_dithered_img.save("test/test_image_bw_dithered.png")
    print("Black and white dithered image saved as test/test_image_bw_dithered.png")

    bw_img = convert_to_bw(img)
    bw_img.save("test/test_image_bw.png")
    print("Black and white image saved as test/test_image_bw.png")
