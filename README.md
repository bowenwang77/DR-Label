# DR-Label: Label Deconstruction and Reconstruction of GNN Models for Catalysis Systems

This repository hosts the official implementation of DRFormer, as presented in our paper "DR-Label: Label Deconstruction and Reconstruction of GNN Models for Catalysis Systems" for AAAI 2024. DRFormer integrates the DRLabel strategy and the Graphormer model with intermediate positional updates and noisy nodes. You can use this code to reproduce the performance of our DRFormer model as described in our paper.

![](./figure/DRFormer_Architecture.png)

Our codebase is developed based on the Graphormer framework. For foundational code and setup instructions, please consult the [Graphormer GitHub repository](https://github.com/microsoft/Graphormer).

## Setup

First, create a directory for the project:

```bash
mkdir ~/DRFormer
```

Then, clone this repository under your directory:

```bash
cd ~/DRFormer
git clone https://github.com/your-username/graphormer.git
cd graphormer
```

## Environment Setup

We provide two ways to set up the environment:

### Option 1: Conda Environment

Set up the conda environment by following the detailed instructions provided on the Graphormer GitHub page.

### Option 2: Compressed Conda Environment

Alternatively, you can download a pre-configured conda environment using the following link:

[Download Environment](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155156871_link_cuhk_edu_hk/Eaj5wfpTb_JFsFB-t6BsyXgBO1zeqx1wGwDDpvBocuE_GQ?e=CtQkOV)

After downloading, use the following commands to set up the environment:

```bash
mkdir -p my_env
tar -xzvf [directory/to/]DRLabel_ADMET_20240131.tar -C my_env
source my_env/bin/activate
```

This environment setup has been tested and is successful on NVIDIA GPUs like TitanX, A40, A100, and V100. For NVIDIA 4090, follow the error messages to update the PyTorch Geometric (PyG) packages as needed.

After setting up the environment, make the following replacements in the environment directory:

- Replace `[your/conda/env/directory]/my_env/lib/python3.9/site-packages/fairseq_cli/train.py` with `fairseq_mods/fairseq_cli/train.py`.
- Replace `[your/conda/env/directory]/my_env/lib/python3.9/site-packages/fairseq/trainer.py` with `fairseq_mods/fairseq/trainer.py`.

## Data Preparation

You can use our toy dataset to validate the entire workflow.

First, run:

```bash
python /root/code/DRFormer_opensource/graphormer/graphormer/modules/outliers_cleaning.py
```

This will generate the `./data_example/toy_example_cleaned` directory.

Then, run:

```bash
fairseq-train --user-dir ./graphormer data_example/toy_example_cleaned --valid-subset val_id --best-checkpoint-metric loss --num-workers 0 --task is2re --criterion mae_deltapos --arch IEFormer_ep_pp_deq --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 5 --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 10000 --total-num-update 1000000 --batch-size 2 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 4 --seed 1 --wandb-project ocp22_geoformer --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1000000 --log-interval 10 --log-format simple --save-interval 2 --validate-interval 2 --keep-interval-updates 10 --save-dir ./bw_checkpoint/DRFormer --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 25 --use-fit-sphere --use-shift-proj --edge-loss-weight 50 --sphere-pass-origin --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.3 --noise-type normal --noise-in-traj --noisy-node-weight 1 --no-node-mask --full-dataset --explicit-pos --pos-update-freq 6 --noisy-label --noisy-label-downscale 1.0 --fix-atoms
```

If you see that the model is successfully training, you can proceed to deploy your own model or try our model on your own dataset by modifying the argument `data_example/toy_example_cleaned` with `[your/own/dataset]`.

## OC20 Dataset

To use the OC20 IS2RE dataset, follow these steps:

1. Download and uncompress the OC20 IS2RE data from the link provided on the OC20 official website: https://github.com/Open-Catalyst-Project

   Download link: https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz

2. Uncompress `is2res_train_val_test_lmdbs.tar.gz`, which will create the following directory structure:

   ```
   [your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all/train
   [your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all/val_id
   [your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all/val_ood_ads
   ...
   ```

3. Clean up the periodic boundary passing atoms calibration by running the following code:

   ```bash
   ./graphormer/modules/outliers_cleaning.py
   ```

   Note: Modify the directory path in `outliers_clean.py` to match your directory structure (`[your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all`). This will generate the `[your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all_cleaned` directory.

## Training

To start training, run the following command:

```bash
fairseq-train --user-dir ./graphormer [your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all_cleaned --valid-subset val_id,val_ood_ads,val_ood_both,val_ood_cat --best-checkpoint-metric loss --num-workers 0 --task is2re --criterion mae_deltapos --arch IEFormer_ep_pp_deq --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 5 --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 10000 --total-num-update 1000000 --batch-size 2 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 4 --seed 1 --wandb-project ocp22_geoformer --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1000000 --log-interval 10 --log-format simple --save-interval 2 --validate-interval 2 --keep-interval-updates 10 --save-dir ./bw_checkpoint/DRFormer --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 25 --use-fit-sphere --use-shift-proj --edge-loss-weight 50 --sphere-pass-origin --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.3 --noise-type normal --noise-in-traj --noisy-node-weight 1 --no-node-mask --full-dataset --explicit-pos --pos-update-freq 6 --noisy-label --noisy-label-downscale 1.0 --fix-atoms
```

Note: This code is designed to run on a machine with 8 A100 GPUs. The batch size is set to 2, and the `update_freq` is set to 4, resulting in a total of 8 * 2 * 4 = 64 instances involved in each update. If you have fewer GPUs available, adjust the `update_freq` accordingly to ensure that the number of instances per update remains 64 for consistent results.

To evaluate a pretrained model, download our checkpoint from the following link:

[Download Checkpoint](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155156871_link_cuhk_edu_hk/EXFiViGtZONHpcWXaQKmS6IBGD7h8Jrj7McDawvyMIm62Q?e=fmmvzq)

Then, run:

```bash
fairseq-train --user-dir ./graphormer [your/directory/to/]is2res_train_val_test_lmdbs/data/is2re/all_cleaned --valid-subset val_id,val_ood_ads,val_ood_both,val_ood_cat --best-checkpoint-metric loss --num-workers 0 --task is2re --criterion mae_deltapos --arch IEFormer_ep_pp_deq --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 5 --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 10000 --total-num-update 1000000 --batch-size 4 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 --wandb-project ocp22_geoformer --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1000000 --log-interval 10 --log-format simple --save-interval 2 --validate-interval 2 --keep-interval-updates 10 --save-dir ./bw_checkpoint/DRFormer_eval --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 25 --use-fit-sphere --use-shift-proj --edge-loss-weight 50 --sphere-pass-origin --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.3 --noise-type normal --noise-in-traj --noisy-node-weight 1 --no-node-mask --full-dataset --explicit-pos --pos-update-freq 6 --noisy-label --noisy-label-downscale 1.0 --fix-atoms --distributed-world-size 1 --device-id 1 --restore-file [your/directory/to]/DRFormer_checkpoint_last.pt
```

### Results

Results on OC20:
![](./figure/Results_table.png)

Visualization of intermediate geometries:
![](./figure/Improved_Visualization.png)

## Finetuning

To finetune our model on your own dataset, load our checkpoint and adjust the parameters:

```bash
fairseq-train --user-dir ./graphormer [your/own/dataset] --valid-subset val_id --best-checkpoint-metric loss --num-workers 0 --task is2re --criterion mae_deltapos --arch IEFormer_ep_pp_deq --optimizer adam --adam-betas 0.9,0.98 --adam-eps 1e-6 --clip-norm 5 --lr-scheduler polynomial_decay --lr 3e-4 --warmup-updates 10000 --total-num-update 1000000 --batch-size 4 --dropout 0.0 --attention-dropout 0.1 --weight-decay 0.001 --update-freq 1 --seed 1 --wandb-project ocp22_geoformer --embed-dim 768 --ffn-embed-dim 768 --attention-heads 48 --max-update 1000000 --log-interval 10 --log-format simple --save-interval 2 --validate-interval 2 --keep-interval-updates 10 --save-dir ./bw_checkpoint/[you_own_dataset] --layers 12 --blocks 4 --required-batch-size-multiple 1 --node-loss-weight 25 --use-fit-sphere --use-shift-proj --edge-loss-weight 50 --sphere-pass-origin --noisy-nodes --noisy-nodes-rate 1.0 --noise-scale 0.3 --noise-type normal --noise-in-traj --noisy-node-weight 1 --no-node-mask --full-dataset --explicit-pos --pos-update-freq 6 --noisy-label --noisy-label-downscale 1.0 --fix-atoms --distributed-world-size 1 --device-id 1 --restore-file [your/directory/to]/DRFormer_checkpoint_last.pt --reset-dataloader --reset-lr-scheduler --reset-optimizer --reset-meters 
```

## Acknowledgements

DRFormer is an advancement built upon significant prior work, particularly the study "Do Transformers Really Perform Badly for Graph Representation?". We extend our gratitude to the authors and contributors of the underlying research that has facilitated the development of DRFormer.


## Common Issues

**Error Message**: `OSError: libcusparse.so.11: cannot open shared object file: No such file or directory`

**Description**: This error typically occurs when the `libcusparse.so.11` library file is missing from your system. This library is essential for certain functionalities and its absence can prevent the application from running properly.

**Solutions**:

1. **Manual Installation of Library File**:
   - Locate the `libcusparse.so.11` file online or from another source.
   - Copy the file to your environment's library directory: `[your/conda/env/directory]/my_env/lib`.
   - Ensure that you have the necessary permissions to add files to this directory.

2. **Reinstallation of PyG Library**:
   - Sometimes, simply reinstalling the PyG library can resolve this issue as it might reinstall all its dependencies, including `libcusparse.so.11`.
   - To reinstall PyG, run the following command in your environment:
     ```bash
     pip install --force-reinstall pyg
     ```

## Citation

If you find our code useful, please consider citing our paper. Below is the BibTeX entry for citing our work in academic papers:

```bibtex
@inproceedings{wang2024dr,
  title={DR-Label: Label Deconstruction and Reconstruction of GNN Models for Catalysis Systems},
  author={Wang, Bowen and Liang, Chen and Wang, Jiaze and Qiu, Jiezhong and Liu, Furui and Hao, Shaogang and Li, Dong and Chen, Guangyong and Zou, Xiaolong and Heng, Pheng Ann},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15456--15465},
  year={2024}
}
```