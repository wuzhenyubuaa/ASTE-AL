import pickle as pkl

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import cv2
import gc


class UncertaintySampler:
    def __init__(self, query_strategy):
        self.query_strategy = query_strategy

    @staticmethod
    def _entropy(prob):
        return (-prob * torch.log(prob)).sum(dim=1)  # b x h x w

    @staticmethod
    def _least_confidence(prob):
        return 1.0 - prob.max(dim=1)[0]  # b x h x w

    @staticmethod
    def _mvmc(prob):
        top2 = prob.topk(k=2, dim=1).values  # b x k x h x w
        return (top2[:, 0, :, :] - top2[:, 1, :, :]).abs()  # b x h x w

    @staticmethod
    def _random(prob):
        b, _, h, w = prob.shape
        return torch.rand((b, h, w))

    def __call__(self, prob):
        return getattr(self, f"_{self.query_strategy}")(prob)

class QuerySelector:
    def __init__(self, args, dataloader_query, device=torch.device("cuda:0")):
        self.dataloader_query = dataloader_query
        self.args = args
        self.device = device
        self.uncertainty_sampler = UncertaintySampler(args.query_strategy)
        self.superpixel_label_seeds = self.dataloader_query.dataset.superpixel

    """ Stage 3: Diversity Sampling  """
    def _select_queries(self, uc_map, name):

        h, w = uc_map.shape[-2:]
        uc_map = uc_map.flatten()
        k = int(h * w * self.args.top_n_percent) if self.args.top_n_percent > 0. else self.args.n_pixels_per_img
        ind_queries = uc_map.topk(k=k, dim=0, largest=self.args.query_strategy in ["entropy", "least_confidence"]).indices.cpu().numpy()

        if self.args.top_n_percent > 0.:
            ind_queries = np.random.choice(ind_queries, self.args.n_pixels_per_img, False)
        query = np.zeros((h * w), dtype=np.bool)

        if self.args.use_superpixel:  # 使用superpixel 更新

            label_flatten = self.superpixel_label_seeds[name[0]]
            for id in ind_queries:
                index_sup = np.where(label_flatten == label_flatten[id])
                query[index_sup[0]] = True
        else:
            query[ind_queries] = True
        query = query.reshape((h, w))
        return query

    def __call__(self, nth_query, model):

        model.eval()
        print(f'choosing pixels by {self.args.query_strategy}')
        list_queries, n_pixels, = list(), 0
        list_names =[]

        with torch.no_grad():
            for batch_ind, (image, gt, queries, name) in tqdm(enumerate(self.dataloader_query)):

                image, gt = image.permute(0, 3, 1, 2).cuda().float(), gt.cuda().float()
                mask = queries.squeeze(dim=0)
                list_names.append(name[0])

                out1u, out2u, out2r, out3r, out4r, out5r = model(image)
                prob = F.softmax(out2u, dim=1)
                uc_map = self.uncertainty_sampler(prob).squeeze(dim=0)
                # exclude pixels that are already annotated, belong to the void category
                uc_map[mask] = 0.0 if self.args.query_strategy in ["entropy", "least_confidence"] else 1.0

                # select queries
                query = self._select_queries(uc_map, name)
                list_queries.append(query)
                n_pixels += query.sum()
                # print(name)
                gc.collect()

        assert len(list_queries) > 0, f"no queries are chosen!"
        # Update labels for query dataloader. Note that this does not update labels for training dataloader.
        self.dataloader_query.dataset.update_query(list_queries, nth_query, list_names)
        return list_queries