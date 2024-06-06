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
from fairseq.data import (
    FairseqDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from ..data.dataset import EpochShuffleDataset

class LMDBDataset:
    def __init__(self, db_path):
        # db_path = db_path[:-5]+"_cleaned.lmdb"
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        # self.env = lmdb.Environment(
        #     db_path,
        #     map_size=(1024 ** 3) * 256,
        #     subdir=False,
        #     readonly=True,
        #     readahead=True,
        #     meminit=False,
        # )
        self.env = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            pos_relaxed=torch.as_tensor(data["pos_relaxed"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            relaxed_energy=data["y_relaxed"],  # python float
            sid = data["sid"],
        )

class LMDBDataset_test:
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        # self.env = lmdb.Environment(
        #     db_path,
        #     map_size=(1024 ** 3) * 256,
        #     subdir=False,
        #     readonly=True,
        #     readahead=True,
        #     meminit=False,
        # )
        self.env = lmdb.open(
            str(db_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return dict(
            pos=torch.as_tensor(data["pos"]).float(),
            cell=torch.as_tensor(data["cell"]).float().view(3, 3),
            atoms=torch.as_tensor(data["atomic_numbers"]).long(),
            tags=torch.as_tensor(data["tags"]).long(),
            sid = torch.as_tensor(data["sid"]).long(),
        )


class PBCDataset:
    def __init__(self, dataset: LMDBDataset,remove_outliers: bool,):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
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
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True
        self.remove_outliers = remove_outliers

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        pos_relaxed = data["pos_relaxed"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
        ).view(-1, 3)
        src_pos = pos[tags > 1] if self.filter_by_tag else pos

        ##Correct outliers
        if self.remove_outliers:
            deltapos_norm_repeat = (pos-pos_relaxed).norm(dim=1).repeat(self.n_cells)
            deltapos_norm_expand = ((pos.unsqueeze(0).expand(self.n_cells,-1,-1)).reshape(-1,3)-expand_pos_relaxed).norm(dim=-1)
            boundary_passing_expand = (deltapos_norm_repeat-deltapos_norm_expand)>0
            deltapos_norm_expand[~boundary_passing_expand]=999
            error_atom_idx = boundary_passing_expand.reshape(self.n_cells,-1).any(dim=0)
            replacing_pos_idx = deltapos_norm_expand.reshape(self.n_cells,-1).argmin(dim=0)
            # expand_pos.reshape(self.n_cells,-1,3)[:,error_atom_idx].shape #[self.n_cell, num_or_error_node, 3]
            # replacing_pos_idx = replacing_pos_idx[error_atom_idx]
            to_replace_pos = (expand_pos_relaxed.reshape(self.n_cells,-1,3)[replacing_pos_idx])[error_atom_idx][:,error_atom_idx][torch.eye(error_atom_idx.sum()).bool()]
            pos_relaxed[error_atom_idx] = to_replace_pos
            expand_pos_relaxed = (
                pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
            ).view(-1, 3)

        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        deltapos_full=torch.cat(
            [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
        )
        atoms_full = torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])
        pos_full = torch.cat([pos, used_expand_pos], dim=0)        
        tags_full = torch.cat([tags, used_expand_tags])
        real_mask_full = torch.cat(
                        [
                            torch.ones_like(tags, dtype=torch.bool),
                            torch.zeros_like(used_expand_tags, dtype=torch.bool),
                        ]
                    )

        #remove outliers:
        # if self.remove_outliers:
        #     deltapos_norm = deltapos_full.norm(dim =1)
        #     deltapos_nonzero = deltapos_norm!=0
        #     deltapos_mean = deltapos_norm[deltapos_nonzero].mean(dim=0)
        #     deltapos_std = deltapos_norm[deltapos_nonzero].std(dim=0)
        #     sigma = ((deltapos_norm-deltapos_mean)/deltapos_std).abs()
        #     large_delta_pos = torch.logical_or(torch.logical_and(sigma>2.5, deltapos_norm>4), deltapos_norm>7)

        #     atoms_full = atoms_full[~large_delta_pos]
        #     pos_full = pos_full[~large_delta_pos]
        #     tags_full = tags_full[~large_delta_pos]
        #     real_mask_full = real_mask_full[~large_delta_pos]
        #     deltapos_full = deltapos_full[~large_delta_pos]

        # print(sigma.max(),deltapos_norm.max())
        # if large_delta_pos.sum()>0:
        #     print("Outliers detected, removing nodes:",len(deltapos_norm[large_delta_pos]),
        #         "atomic number:",torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])[large_delta_pos] )

        return dict(
            pos=pos_full,
            atoms=atoms_full,
            tags=tags_full,
            real_mask=real_mask_full,
            deltapos=deltapos_full,
            relaxed_energy=data["relaxed_energy"],
            sid = data["sid"],
            cell = cell,
        )

class PBCDataset_NoisyNodes:
    def __init__(self, dataset: LMDBDataset, noise_scale: float, 
    noise_type: str, noisy_node_rate: float, 
    noise_deltapos_normed: bool, noise_in_traj: bool,
    noisy_label: bool, noisy_label_downscale: float,
    remove_outliers: bool,):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
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
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True ##If true, only consider expanded nodes that have dist<cutoff with center adsorbate
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        self.noisy_node_rate = noisy_node_rate
        self.noise_deltapos_normed = noise_deltapos_normed
        self.noise_in_traj = noise_in_traj
        self.noisy_label = noisy_label
        self.remove_outliers = remove_outliers
        self.noisy_label_downscale = noisy_label_downscale
        # if self.noise_type == "trunc_normal":
            # self.noise_f = lambda num_mask: np.clip(
            #     np.random.randn(num_mask, 3) * self.noise_scale,
            #     a_min=-self.noise_scale * 2.0,
            #     a_max=self.noise_scale * 2.0,
            # )
        if self.noise_type == "normal":
            # self.noise_f = lambda num_mask: np.random.randn(num_mask, 3) * self.noise_scale
            self.noise_f = lambda num_mask: normalize(np.random.randn(num_mask, 3),axis=1,norm="l2") * np.random.randn(num_mask,1) * self.noise_scale
        elif self.noise_type == "uniform":
            # self.noise_f = lambda num_mask: np.random.uniform(
            #     low=-self.noise_scale, high=self.noise_scale, size=(num_mask, 3)
            # )
            self.noise_f = lambda num_mask: normalize(np.random.randn(num_mask, 3),axis=1,norm="l2") * np.random.rand(num_mask,1) * self.noise_scale
        else:
            self.noise_f = lambda num_mask: 0.0

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        pos_relaxed = data["pos_relaxed"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]
        # Visualize
        # import pymatgen
        # lattice = np.array(cell)
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("./visualize/vasp/POSCAR"+str(idx))
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos_relaxed),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("./visualize/vasp/POSCAR_relaxed"+str(idx))

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        expand_pos_relaxed = (
            pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
        ).view(-1, 3)
        src_pos = pos[tags > 1] if self.filter_by_tag else pos
        ##Correct outliers
        if self.remove_outliers:
            deltapos_norm_repeat = (pos-pos_relaxed).norm(dim=1).repeat(self.n_cells)
            deltapos_norm_expand = ((pos.unsqueeze(0).expand(self.n_cells,-1,-1)).reshape(-1,3)-expand_pos_relaxed).norm(dim=-1)
            boundary_passing_expand = (deltapos_norm_repeat-deltapos_norm_expand)>0
            deltapos_norm_expand[~boundary_passing_expand]=999
            error_atom_idx = boundary_passing_expand.reshape(self.n_cells,-1).any(dim=0)
            replacing_pos_idx = deltapos_norm_expand.reshape(self.n_cells,-1).argmin(dim=0)
            # expand_pos.reshape(self.n_cells,-1,3)[:,error_atom_idx].shape #[self.n_cell, num_or_error_node, 3]
            # replacing_pos_idx = replacing_pos_idx[error_atom_idx]
            to_replace_pos = (expand_pos_relaxed.reshape(self.n_cells,-1,3)[replacing_pos_idx])[error_atom_idx][:,error_atom_idx][torch.eye(error_atom_idx.sum()).bool()]
            pos_relaxed[error_atom_idx] = to_replace_pos
            expand_pos_relaxed = (
                pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
            ).view(-1, 3)

        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]
        used_expand_pos_relaxed = expand_pos_relaxed[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        deltapos_full=torch.cat(
            [pos_relaxed - pos, used_expand_pos_relaxed - used_expand_pos], dim=0
        )
        atoms_full = torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])
        pos_full = torch.cat([pos, used_expand_pos], dim=0)        
        tags_full = torch.cat([tags, used_expand_tags])
        real_mask_full = torch.cat(
                        [
                            torch.ones_like(tags, dtype=torch.bool),
                            torch.zeros_like(used_expand_tags, dtype=torch.bool),
                        ]
                    )
        deltapos_norm = deltapos_full.norm(dim =1)
        deltapos_nonzero = deltapos_norm!=0

        #remove outliers:
        # if self.remove_outliers:
        #     deltapos_mean = deltapos_norm[deltapos_nonzero].mean(dim=0)
        #     deltapos_std = deltapos_norm[deltapos_nonzero].std(dim=0)
        #     sigma = ((deltapos_norm-deltapos_mean)/deltapos_std).abs()
        #     large_delta_pos = torch.logical_or(torch.logical_and(sigma>2.5, deltapos_norm>4), deltapos_norm>7)

        #     atoms_full = atoms_full[~large_delta_pos]
        #     pos_full = pos_full[~large_delta_pos]
        #     tags_full = tags_full[~large_delta_pos]
        #     real_mask_full = real_mask_full[~large_delta_pos]
        #     deltapos_full = deltapos_full[~large_delta_pos]
        # print(sigma.max(),deltapos_norm.max())
        # if large_delta_pos.sum()>0:
        #     print("Outliers detected, removing nodes:",len(deltapos_norm[large_delta_pos]),
        #         "atomic number:",torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])[large_delta_pos] )

        ####Noisy node
        # if self.remove_outliers:
        #     noisy_node_mask = torch.logical_and(torch.rand(deltapos_full.shape[0])<self.noisy_node_rate,deltapos_nonzero[~large_delta_pos])
        # else:
        noisy_node_mask = torch.logical_and(torch.rand(deltapos_full.shape[0])<self.noisy_node_rate,deltapos_nonzero)

        pos_full_noise = pos_full.clone()
        deltapos_full_noise = deltapos_full.clone()
        pos_full_noisy_label = pos_full.clone()+deltapos_full.clone()
        deltapos_full_noisy_label = torch.zeros_like(deltapos_full)
        if noisy_node_mask.sum()!=0: ##in case of error instances that have initial_pos==relaxed_pos
            node_noise = self.noise_f(noisy_node_mask.sum())
            node_noise = torch.from_numpy(node_noise).type_as(deltapos_full)
            if self.noise_deltapos_normed:
                    node_noise *= deltapos_norm[noisy_node_mask].unsqueeze(-1)
                    node_noise /= deltapos_norm[noisy_node_mask].unsqueeze(-1).mean()
            if self.noise_in_traj:
                node_noise += (deltapos_full*torch.rand(deltapos_full.shape[0]).unsqueeze(-1))[noisy_node_mask]
            pos_full_noise[noisy_node_mask] += node_noise
            deltapos_full_noise[noisy_node_mask] -= node_noise
            if self.noisy_label:
                label_noise = self.noise_f(noisy_node_mask.sum()) * self.noisy_label_downscale
                label_noise = torch.from_numpy(label_noise).type_as(deltapos_full)
                deltapos_full_noisy_label[noisy_node_mask] = label_noise
                pos_full_noisy_label[noisy_node_mask] -= label_noise
                # deltapos_full_noise[noisy_node_mask] += label_noise
        if not self.noisy_label:
            return dict(
                pos=pos_full,
                noisy_pos = pos_full_noise,
                atoms=atoms_full,
                tags=tags_full,
                real_mask=real_mask_full,
                deltapos=deltapos_full,
                noisy_deltapos = deltapos_full_noise,
                relaxed_energy=data["relaxed_energy"],
                sid = data["sid"],
                cell = cell,
            )
        else:
            return dict(
                pos=pos_full,
                noisy_pos = pos_full_noise,
                noisy_label_pos = pos_full_noisy_label,
                atoms=atoms_full,
                tags=tags_full,
                real_mask=real_mask_full,
                deltapos=deltapos_full,
                noisy_deltapos = deltapos_full_noise,
                noisy_label_deltapos = deltapos_full_noisy_label,
                relaxed_energy=data["relaxed_energy"],
                sid = data["sid"],
                cell = cell,
            )

class PBCDataset_test:
    def __init__(self, dataset: LMDBDataset_test):
        self.dataset = dataset
        self.cell_offsets = torch.tensor(
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
        self.n_cells = self.cell_offsets.size(0)
        self.cutoff = 8
        self.filter_by_tag = True
        # self.remove_outliers = remove_outliers

    def __len__(self):
        return len(self.dataset)

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        data = self.dataset[idx]

        pos = data["pos"]
        cell = data["cell"]
        atoms = data["atoms"]
        tags = data["tags"]

        offsets = torch.matmul(self.cell_offsets, cell).view(self.n_cells, 1, 3)
        expand_pos = (pos.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets).view(
            -1, 3
        )
        src_pos = pos[tags > 1] if self.filter_by_tag else pos
        ##Correct outliers
        # if True:
            # deltapos_norm_repeat = (pos-pos_relaxed).norm(dim=1).repeat(self.n_cells)
            # deltapos_norm_expand = ((pos.unsqueeze(0).expand(self.n_cells,-1,-1)).reshape(-1,3)-expand_pos_relaxed).norm(dim=-1)
            # boundary_passing_expand = (deltapos_norm_repeat-deltapos_norm_expand)>0
            # deltapos_norm_expand[~boundary_passing_expand]=999
            # error_atom_idx = boundary_passing_expand.reshape(self.n_cells,-1).any(dim=0)
            # replacing_pos_idx = deltapos_norm_expand.reshape(self.n_cells,-1).argmin(dim=0)
            # # expand_pos.reshape(self.n_cells,-1,3)[:,error_atom_idx].shape #[self.n_cell, num_or_error_node, 3]
            # # replacing_pos_idx = replacing_pos_idx[error_atom_idx]
            # to_replace_pos = (expand_pos_relaxed.reshape(self.n_cells,-1,3)[replacing_pos_idx])[error_atom_idx][:,error_atom_idx][torch.eye(error_atom_idx.sum()).bool()]
            # pos_relaxed[error_atom_idx] = to_replace_pos
            # expand_pos_relaxed = (
            #     pos_relaxed.unsqueeze(0).expand(self.n_cells, -1, -1) + offsets ###This is so wrong! Previously the source code use pos(initial)
            # ).view(-1, 3)
        dist: Tensor = (src_pos.unsqueeze(1) - expand_pos.unsqueeze(0)).norm(dim=-1)
        used_mask = (dist < self.cutoff).any(dim=0) & tags.ne(2).repeat(
            self.n_cells
        )  # not copy ads
        used_expand_pos = expand_pos[used_mask]

        used_expand_tags = tags.repeat(self.n_cells)[
            used_mask
        ]  # original implementation use zeros, need to test
        atoms_full = torch.cat([atoms, atoms.repeat(self.n_cells)[used_mask]])
        pos_full = torch.cat([pos, used_expand_pos], dim=0)        
        tags_full = torch.cat([tags, used_expand_tags])
        real_mask_full = torch.cat(
                        [
                            torch.ones_like(tags, dtype=torch.bool),
                            torch.zeros_like(used_expand_tags, dtype=torch.bool),
                        ]
                    )


        return dict(
            pos=pos_full,
            atoms=atoms_full,
            tags=tags_full,
            real_mask=real_mask_full,
            sid = data["sid"],
            cell = data["cell"]
        )

def pad_1d(samples: Sequence[Tensor], fill=0, multiplier=8):
    max_len = max(x.size(0) for x in samples)
    max_len = (max_len + multiplier - 1) // multiplier * multiplier
    n_samples = len(samples)
    out = torch.full(
        (n_samples, max_len, *samples[0].shape[1:]), fill, dtype=samples[0].dtype
    )
    for i in range(n_samples):
        x_len = samples[i].size(0)
        out[i][:x_len] = samples[i]
    return out


class AtomDataset(FairseqDataset):
    def __init__(self, dataset, keyword):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.atom_list = [
            1,
            5,
            6,
            7,
            8,
            11,
            13,
            14,
            15,
            16,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            55,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
        ]
        # fill others as unk
        unk_idx = len(self.atom_list) + 1
        self.atom_mapper = torch.full((128,), unk_idx)
        for idx, atom in enumerate(self.atom_list):
            self.atom_mapper[atom] = idx + 1  # reserve 0 for paddin

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        atoms: Tensor = self.dataset[index][self.keyword]
        return self.atom_mapper[atoms]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        return pad_1d(samples)


class KeywordDataset(FairseqDataset):
    def __init__(self, dataset, keyword, is_scalar=False, pad_fill=0):
        super().__init__()
        self.dataset = dataset
        self.keyword = keyword
        self.is_scalar = is_scalar
        self.pad_fill = pad_fill

    @lru_cache(maxsize=16)
    def __getitem__(self, index):
        return self.dataset[index][self.keyword]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):
        if self.is_scalar:
            return torch.tensor(samples)
        return pad_1d(samples, fill=self.pad_fill)


@register_task("is2re")
class IS2RETask(FairseqTask):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument("data", metavar="FILE", help="directory for data")

    @property
    def target_dictionary(self):
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        assert split in [
            "train",
            "val_id",
            "val_ood_ads",
            "val_ood_cat",
            "val_ood_both",
            "test_id",
            "test_ood_ads",
            "test_ood_cat",
            "test_ood_both",
        ], "invalid split: {}!".format(split)
        print(" > Loading {} ...".format(split))
        if self.cfg.data.split("/")[-1] == "SAA":
            split_idx =split.split("_")[0]+str(self.cfg.SAA_idx)
            db_path = str(Path(self.cfg.data) / split /  split_idx / "data.lmdb")
        elif self.cfg.full_dataset and split=="train":
            db_path = str(Path(self.cfg.data) / "train" / "data.lmdb")
        else:
            db_path = str(Path(self.cfg.data) / split / "data.lmdb")
        testing = split.split("_")[0]=="test" and self.cfg.data.split("/")[-1]!="SAA"
        # testing = True
        if not testing:
            lmdb_dataset = LMDBDataset(db_path)
        else:
            lmdb_dataset = LMDBDataset_test(db_path)

        use_noisy_nodes = self.cfg.noisy_nodes and split == "train"
        noisy_label = self.cfg.noisy_label and use_noisy_nodes
        if not testing:
            if use_noisy_nodes:
                pbc_dataset = PBCDataset_NoisyNodes(lmdb_dataset,self.cfg.noise_scale,
                        self.cfg.noise_type, self.cfg.noisy_nodes_rate,  
                        self.cfg.noise_deltapos_normed, self.cfg.noise_in_traj, 
                        self.cfg.noisy_label, self.cfg.noisy_label_downscale,
                        self.cfg.remove_outliers)
            else:
                pbc_dataset = PBCDataset(lmdb_dataset,self.cfg.remove_outliers)
        else:
            pbc_dataset = PBCDataset_test(lmdb_dataset)

        atoms = AtomDataset(pbc_dataset, "atoms")
        tags = KeywordDataset(pbc_dataset, "tags")
        real_mask = KeywordDataset(pbc_dataset, "real_mask")

        pos = KeywordDataset(pbc_dataset, "pos")
        cell = KeywordDataset(pbc_dataset, "cell")
        if not testing:
            relaxed_energy = KeywordDataset(pbc_dataset, "relaxed_energy", is_scalar=True)
            deltapos = KeywordDataset(pbc_dataset, "deltapos")
        sid = KeywordDataset(pbc_dataset, "sid", is_scalar=True)

        if not testing:
            if use_noisy_nodes:
                if noisy_label:
                    noisy_pos = KeywordDataset(pbc_dataset, "noisy_pos")
                    noisy_deltapos = KeywordDataset(pbc_dataset, "noisy_deltapos")
                    noisy_label_deltapos = KeywordDataset(pbc_dataset,"noisy_label_deltapos")
                    noisy_label_pos = KeywordDataset(pbc_dataset,"noisy_label_pos")
                    dataset = NestedDictionaryDataset(
                        {
                            "net_input": {
                                "pos": pos,
                                "noisy_pos": noisy_pos,
                                "noisy_label_pos": noisy_label_pos,
                                "atoms": atoms,
                                "tags": tags,
                                "real_mask": real_mask,
                                "sid": sid,
                                "cell": cell,
                            },
                            "targets": {
                                "relaxed_energy": relaxed_energy,
                                "deltapos": deltapos,
                                "noisy_deltapos": noisy_deltapos,
                                "noisy_label_deltapos": noisy_label_deltapos,
                            },
                        },
                        sizes=[np.zeros(len(atoms))],
                    )
                else:
                    noisy_pos = KeywordDataset(pbc_dataset, "noisy_pos")
                    noisy_deltapos = KeywordDataset(pbc_dataset, "noisy_deltapos")
                    dataset = NestedDictionaryDataset(
                        {
                            "net_input": {
                                "pos": pos,
                                "noisy_pos": noisy_pos,
                                "atoms": atoms,
                                "tags": tags,
                                "real_mask": real_mask,
                                "sid": sid,
                                "cell": cell,
                            },
                            "targets": {
                                "relaxed_energy": relaxed_energy,
                                "deltapos": deltapos,
                                "noisy_deltapos": noisy_deltapos,
                            },
                        },
                        sizes=[np.zeros(len(atoms))],
                    )
            else:
                dataset = NestedDictionaryDataset(
                    {
                        "net_input": {
                            "pos": pos,
                            "atoms": atoms,
                            "tags": tags,
                            "real_mask": real_mask,
                            "sid": sid,
                            "cell": cell,
                        },
                        "targets": {
                            "relaxed_energy": relaxed_energy,
                            "deltapos": deltapos,
                        },
                    },
                    sizes=[np.zeros(len(atoms))],
                )
        else:
            dataset = NestedDictionaryDataset(
                {
                    "net_input": {
                        "pos": pos,
                        "atoms": atoms,
                        "tags": tags,
                        "real_mask": real_mask,
                        "sid": sid,
                        "cell": cell,
                    },
                },
                sizes=[np.zeros(len(atoms))],
            )
        if split == "train":
            dataset = EpochShuffleDataset(
                dataset,
                num_samples=len(atoms),
                seed=self.cfg.seed,
            )

        print("| Loaded {} with {} samples".format(split, len(dataset)))
        self.datasets[split] = dataset
