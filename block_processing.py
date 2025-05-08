import numpy as np

def split_into_blocks(image_channel, block_size, fill_value=0):
    if not isinstance(image_channel, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if image_channel.ndim != 2:
        raise ValueError("Input must be a 2D array.")
    if not isinstance(block_size, int) or block_size <= 0:
        raise ValueError("block_size must be a positive integer.")

    height, width = image_channel.shape

    pad_height = (block_size - (height % block_size)) % block_size
    pad_width = (block_size - (width % block_size)) % block_size

    if pad_height > 0 or pad_width > 0:
        padded_image = np.pad(image_channel,
                              ((0, pad_height), (0, pad_width)),
                              mode='constant',
                              constant_values=fill_value)
    else:
        padded_image = image_channel

    padded_height, padded_width = padded_image.shape

    blocks = []
    for row in range(0, padded_height, block_size):
        for col in range(0, padded_width, block_size):
            block = padded_image[row:row+block_size, col:col+block_size]
            blocks.append(block)

    return blocks

def reassemble_from_blocks(blocks, padded_height, padded_width):
    if not blocks:
        return np.array([], dtype=np.uint8).reshape(0, 0)

    block_size = blocks[0].shape[0]
    if any(block.shape != (block_size, block_size) for block in blocks):
        raise ValueError("All blocks must be square and of the same size.")

    if padded_height % block_size != 0 or padded_width % block_size != 0:
        raise ValueError("padded_height and padded_width must be multiples of block_size.")

    num_blocks_vert = padded_height // block_size
    num_blocks_horz = padded_width // block_size

    if len(blocks) != num_blocks_vert * num_blocks_horz:
        raise ValueError("Number of blocks does not match padded dimensions.")

    image = np.zeros((padded_height, padded_width), dtype=blocks[0].dtype)

    idx = 0
    for r in range(num_blocks_vert):
        for c in range(num_blocks_horz):
            image[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = blocks[idx]
            idx += 1

    return image
