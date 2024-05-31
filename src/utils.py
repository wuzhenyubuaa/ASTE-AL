import torch
import numpy as np
from PIL import Image
from time import sleep


def query_strategy(prob, query_strategy):
    # prob: b x n_classes x h x w
    if query_strategy == "least_confidence":
        query = 1.0 - prob.max(dim=1)[0]  # b x h x w

    elif query_strategy == "mvmc":
        query = prob.topk(k=2, dim=1).values  # b x k x h x w
        query = (query[:, 0, :, :] - query[:, 1, :, :]).abs()  # b x h x w

    elif query_strategy == "entropy":
        query = (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    elif query_strategy == "random":
        b, _, h, w = prob.shape
        query = torch.rand((b, h, w))

    else:
        raise ValueError
    return query


class Visualiser:
    def __init__(self, dataset_name='voc'):
        if dataset_name == "cv":
            global palette_cv
            self.palette = palette_cv

        elif dataset_name == "cs":
            global palette_cs
            self.palette = palette_cs

        elif dataset_name == "voc":
            global palette_voc
            self.palette = palette_voc

    def _preprocess(self, tensor, seg=False, downsample=1):
        if len(tensor.shape) == 2:
            h, w = tensor.shape
        elif len(tensor.shape) == 3:
            c, h, w = tensor.shape
        else:
            raise ValueError(f"{tensor.shape}")

        if seg:
            tensor_flatten = tensor.flatten()
            grid = torch.zeros([h * w, 3], dtype=torch.uint8)
            for i in range(len(tensor_flatten)):
                grid[i] = torch.tensor(self.palette[tensor_flatten[i].item()], dtype=torch.uint8)
            tensor = grid.view(h, w, 3)

        else:
            tensor -= tensor.min()
            tensor = tensor / (tensor.max() + 1e-7)
            tensor *= 255

            if len(tensor.shape) == 3:
                c,h,w = tensor.shape
                if c ==3:
                    tensor = tensor.permute(1, 2, 0)
                else:
                    tensor=tensor[0]

        arr = np.clip(tensor.numpy(), 0, 255).astype(np.uint8)
        return Image.fromarray(arr).resize((w // downsample, h // downsample))

    @staticmethod
    def _make_grid(list_imgs):
        width = 0
        height = list_imgs[0].height
        for img in list_imgs:
            width += img.width

        grid = Image.new("RGB", (width, height))
        x_offset = 0
        for img in list_imgs:
            grid.paste(img, (x_offset, 0))
            x_offset += img.width
        return grid

    def __call__(self, dict_tensors, fp='', show=False):
        list_imgs = list()

        list_imgs.append(self._preprocess(dict_tensors['input']))

        list_imgs.append(self._preprocess(dict_tensors['target']))
        list_imgs.append(self._preprocess(dict_tensors['pred']))
        list_imgs.append(self._preprocess(dict_tensors['confidence']))
        list_imgs.append(self._preprocess(dict_tensors['margin']))
        list_imgs.append(self._preprocess(dict_tensors['entropy']))

        img = self._make_grid(list_imgs)

        if fp:
            img.save(fp)

        if show:
            img.show()
            sleep(60)
        return


palette_cv = {
    0: (128, 128, 128),
    1: (128, 0, 0),
    2: (192, 192, 128),
    3: (128, 64, 128),
    4: (0, 0, 192),
    5: (128, 128, 0),
    6: (192, 128, 128),
    7: (64, 64, 128),
    8: (64, 0, 128),
    9: (64, 64, 0),
    10: (0, 128, 192),
    11: (0, 0, 0)
}

palette_cs = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (107, 142, 35),
    9: (152, 251, 152),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230),
    18: (119, 11, 32),
    19: (0, 0, 0)
}

palette_voc = {
    0: [0, 0, 0],
    1: [128, 0, 0],
    2: [0, 128, 0],
    3: [128, 128, 0],
    4: [0, 0, 128],
    5: [128, 0, 128],
    6: [0, 128, 128],
    7: [128, 128, 128],
    8: [64, 0, 0],
    9: [192, 0, 0],
    10: [64, 128, 0],
    11: [192, 128, 0],
    12: [64, 0, 128],
    13: [192, 0, 128],
    14: [64, 128, 128],
    15: [192, 128, 128],
    16: [0, 64, 0],
    17: [128, 64, 0],
    18: [0, 192, 0],
    19: [128, 192, 0],
    20: [0, 64, 128],
    255: [255, 255, 255]
}

dict_cv_label_category = {
    0: "sky",
    1: "building",
    2: "pole",
    3: "road",
    4: "pavement",
    5: "tree",
    6: "sign symbol",
    7: "fence",
    8: "car",
    9: "pedestrian",
    10: "bicyclist",
    11: "void"
}