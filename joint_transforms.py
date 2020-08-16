import numbers
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
# random.seed(2018)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask,depth):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, depth = t(img, mask,depth)
        return img, mask, depth


class RandomCrop(object):
    def __init__(self, size, padding=0):
        # isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。 numbers.Number表示数字类型
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask,depth):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            depth = ImageOps.expand(depth, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        dw, dh = depth.size
        if w == tw and h == th and dw == tw and dh == th:
            return img, mask, depth
        if w < tw or h < th or dw < tw or dh < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST),depth.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)), depth.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask,depth):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT),depth.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask,depth


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask,depth):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST),depth.rotate(rotate_degree, Image.NEAREST)


class Get_box(object):
    def __init__(self, points=None, pad=0, zero_pad=False):  # pad is the relax pixel
        self.points = points
        self.pad = pad
        self.zero_pad = zero_pad

    def __call__(self, mask):
        if self.points is not None:
            inds = np.flip(self.transpose(), axis=0)
        else:
            inds = np.where(mask > 0)

        if inds[0].shape[0] == 0:
            return None

        if self.zero_pad:
            x_min_bound = -np.inf
            y_min_bound = -np.inf
            x_max_bound = np.inf
            y_max_bound = np.inf
        else:
            x_min_bound = 0
            y_min_bound = 0
            x_max_bound = mask.shape[1] - 1
            y_max_bound = mask.shape[0] - 1

        x_min = max(inds[1].min() - self.pad, x_min_bound)
        y_min = max(inds[0].min() - self.pad, y_min_bound)
        x_max = min(inds[1].max() + self.pad, x_max_bound)
        y_max = min(inds[0].max() + self.pad, y_max_bound)

        return x_min, y_min, x_max, y_max


class Crop_from_bbox(object):
    def __init__(self, relax, zero_pad=False):
        self.bbox = Get_box(points=None, pad=relax, zero_pad=False)
        self.zero_pad = zero_pad

    def __call__(self, mask):
        bounds = (0, 0, mask.shape[1] - 1, mask.shape[0] - 1)
        # Valid bounding box locations as (x_min, y_min, x_max, y_max)
        bbox_valid = (max(self.bbox[0], bounds[0]),
                      max(self.bbox[1], bounds[1]),
                      min(self.bbox[2], bounds[2]),
                      min(self.bbox[3], bounds[3]))

        if self.zero_pad:
            # Initialize crop size (first 2 dimensions)
            crop = np.zeros((self.bbox[3] - self.bbox[1] + 1, self.bbox[2] - self.bbox[0] + 1), dtype=mask.dtype)
            # Offsets for x and y
            offsets = (-self.bbox[0], -self.bbox[1])
        else:
            assert (self.bbox == bbox_valid)
            crop = np.zeros((bbox_valid[3] - bbox_valid[1] + 1, bbox_valid[2] - bbox_valid[0] + 1), dtype=mask.dtype)
            offsets = (-bbox_valid[0], -bbox_valid[1])

        # Simple per element addition in the tuple
        inds = tuple(map(sum, zip(bbox_valid, offsets + offsets)))

        mask = np.squeeze(mask)
        if mask.ndim == 2:
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1] = \
                mask[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1]
        else:
            crop = np.tile(crop[:, :, np.newaxis], [1, 1, 3])  # Add 3 RGB Channels
            crop[inds[1]:inds[3] + 1, inds[0]:inds[2] + 1, :] = \
                mask[bbox_valid[1]:bbox_valid[3] + 1, bbox_valid[0]:bbox_valid[2] + 1, :]
        return crop


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, img, mask):
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transforms.Resize(img, (new_h, new_w))
        mask = transforms.Resize(mask, (new_h, new_w))

        return img, mask

if __name__ == '__main__':
    mask = '/home/zun/Videos/WORK6/data/DUTS-Training/DUTS-TR-Image/3096.png'
    relax = 50
    zero_pad = False
    crop = Rescale(320)
    print(crop)