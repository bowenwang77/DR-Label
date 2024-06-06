# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Sequence, Union

import pickle
from functools import lru_cache
from sklearn.preprocessing import normalize
import lmdb

import numpy as np
import torch
from torch import Tensor
import os
# from fairseq.data import (
#     FairseqDataset,
#     BaseWrapperDataset,
#     NestedDictionaryDataset,
#     data_utils,
# )
# from fairseq.tasks import FairseqTask, register_task

# from ..data.dataset import EpochShuffleDataset


def clean_outliers(data_path):

    env = lmdb.open(
        str(data_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    # import pdb
    # pdb.set_trace()
    save_path = os.path.join(*data_path.split('/')[:-3],data_path.split('/')[-3]+"_cleaned",data_path.split('/')[-2]) 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    db = lmdb.open(
        str(save_path+"/data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,  
    )
    len: int = env.stat()["entries"]
    print("begin processing file", data_path, "total instance", len)
    outliers_cnt = 0
    for idx in range(len):
        data = pickle.loads(env.begin().get(f"{idx}".encode()))
        pos=torch.as_tensor(data["pos"]).float()
        pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float()
        cell=torch.as_tensor(data["cell"]).float().view(3, 3)
        atoms=torch.as_tensor(data["atomic_numbers"]).long()
        tags=torch.as_tensor(data["tags"]).long()
        relaxed_energy=data["y_relaxed"]
        sid = data["sid"]
        cell_offsets = torch.tensor(
                [
                    [-1, -1, 0],
                    [-1, 0, 0],
                    [-1, 1, 0],
                    [0, -1, 0],
                    [0, 1, 0],
                    [1, -1, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                ],
            ).float()
        n_cells = cell_offsets.size(0)
        offsets = torch.matmul(cell_offsets, cell).view(n_cells, 1, 3)
        expand_pos_relaxed = (
            pos_relaxed.unsqueeze(0).expand(n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
        ).view(-1, 3)
        deltapos_norm_repeat = (pos-pos_relaxed).norm(dim=1).repeat(n_cells)
        deltapos_norm_expand = ((pos.unsqueeze(0).expand(n_cells,-1,-1)).reshape(-1,3)-expand_pos_relaxed).norm(dim=-1)
        boundary_passing_expand = (deltapos_norm_repeat-deltapos_norm_expand)>0
        deltapos_norm_expand[~boundary_passing_expand]=999
        error_atom_idx = boundary_passing_expand.reshape(n_cells,-1).any(dim=0)
        replacing_pos_idx = deltapos_norm_expand.reshape(n_cells,-1).argmin(dim=0)
        # expand_pos.reshape(n_cells,-1,3)[:,error_atom_idx].shape #[n_cell, num_or_error_node, 3]
        # replacing_pos_idx = replacing_pos_idx[error_atom_idx]
        to_replace_pos = (expand_pos_relaxed.reshape(n_cells,-1,3)[replacing_pos_idx])[error_atom_idx][:,error_atom_idx][torch.eye(error_atom_idx.sum()).bool()]
        pos_relaxed[error_atom_idx] = to_replace_pos
        data["pos_relaxed"]=pos_relaxed
        # expand_pos_relaxed = (
        #     pos_relaxed.unsqueeze(0).expand(n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
        # ).view(-1, 3)
        outliers_cnt += error_atom_idx.sum()
        if idx%10==0:
            print("done_process:",idx, "outliers removed:", outliers_cnt)
        txn=db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(data, protocol=-1))
        txn.commit()
        # txn.sync()
    print("done_process:",idx, "outliers removed:", outliers_cnt)
    print("file_saved_in:",save_path+"data.lmdb")
    db.close()

    pass


if __name__ == "__main__":
    base_dir_list = [
        "./data_example/toy_example"]
    # Iterate over all subdirectories in the base directory
    for base_dir in base_dir_list:
        for subdir in os.listdir(base_dir):
            data_path = os.path.join(base_dir, subdir, "data.lmdb")
            clean_outliers(data_path)

