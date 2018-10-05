import imageio
import sys

if __name__ == '__main__':

    if len(sys.argv) == 2:
        _, filename = sys.argv

        img = imageio.imread(filename).astype(dtype='float32')

        print('DTYPE:', img.dtype)
        print('SHAPE:', img.shape)

    elif len(sys.argv) == 3:

        _, filename, type = sys.argv

        img = imageio.imread(filename).astype(dtype=type)

        sys.stdout.buffer.write(img.tobytes())

    else:
        print("Usage: dump_img.py filename [type]")


