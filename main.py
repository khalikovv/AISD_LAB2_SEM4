import os
from PIL import Image
import matplotlib.pyplot as plt
from jpeg_codec import jpeg_compress, jpeg_decompress

def convert_to_grayscale(image):
    return image.convert('L')

def convert_to_dithered_grayscale(image):
    return image.convert('1') 

def save_image(image, path):
    image.save(path)

def plot_compression_results(results, output_path):
    plt.figure()
    for image_name, data in results.items():
        qualities = sorted(data.keys())
        sizes = [data[q] for q in qualities]
        plt.plot(qualities, sizes, label=image_name)
    plt.xlabel('Quality')
    plt.ylabel('Size (bytes)')
    plt.title('Compression Size and Quality')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def prepare_test_images():
    os.makedirs('test_images', exist_ok=True)
    
    lenna_path = 'Lenna.png'
    lenna_img = Image.open(lenna_path).convert('RGB')
    lenna_img.save('test_images/Lenna_color.png')

    
    lenna_gray = convert_to_grayscale(lenna_img)
    lenna_gray.save('test_images/Lenna_gray.png')
    lenna_dither = convert_to_dithered_grayscale(lenna_img)
    lenna_dither.save('test_images/Lenna_dither.png')

    # Copy new generated test images
    from shutil import copyfile
    copyfile('test/test_image.png', 'test_images/test_image.png')
    copyfile('test/test_image_grayscale.png', 'test_images/test_image_grayscale.png')
    copyfile('test/test_image_bw_dithered.png', 'test_images/test_image_bw_dithered.png')
    copyfile('test/test_image_bw.png', 'test_images/test_image_bw.png')

def run_compression_tests():
    qualities = [0, 20, 40, 60, 80, 100]
    image_files = [
        ('test_images/Lenna_color.png', 'Lenna_color'),
        ('test_images/Lenna_gray.png', 'Lenna_gray'),
        ('test_images/Lenna_dither.png', 'Lenna_dither'),
        ('test_images/test_image.png', 'test_image_color'),
        ('test_images/test_image_grayscale.png', 'test_image_grayscale'),
        ('test_images/test_image_bw_dithered.png', 'test_image_bw_dithered'),
        ('test_images/test_image_bw.png', 'test_image_bw'),
    ]

    results = {name: {} for _, name in image_files}

    for image_path, name in image_files:
        os.makedirs(f'output/{name}', exist_ok=True)
        for q in qualities:
            if q == 0:
                quality = 1
            else:
                quality = q
            compressed_path = f'output/{name}/{name}_q{quality}.myjpeg'
            decompressed_path = f'output/{name}/{name}_q{quality}_decompressed.png'

            jpeg_compress(image_path, compressed_path, quality=quality)
            jpeg_decompress(compressed_path, decompressed_path)

            size = os.path.getsize(compressed_path)
            results[name][q] = size

    plot_compression_results(results, 'output/compression_size_vs_quality.png')

if __name__ == '__main__':
    prepare_test_images()
    run_compression_tests()
