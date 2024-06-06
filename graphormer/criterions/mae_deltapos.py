# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Callable, Mapping, Sequence, Tuple
from numpy import mod
import torch
from torch import Tensor
import torch.nn.functional as F


from fairseq import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from ..modules.visualize import visualize_trace, visualize_trace_poscar

@register_criterion("mae_deltapos")
class IS2RECriterion(FairseqCriterion):
    e_thresh = 0.02
    e_thresh_100 = 0.1
    e_mean = -1.4729953244844094
    e_std = 2.2707848125378405
    #node norm unclean center
    d_mean = [0.0020, -0.0072, 0.0797]
    d_std = [0.4524, 0.4885, 0.5783]    
    # print("node norm changed")
    def __init__(self, task, cfg):
        super().__init__(task)
        self.node_loss_weight = cfg.node_loss_weight
        if cfg.use_shift_proj:
            self.edge_loss_weight = cfg.edge_loss_weight
            self.min_edge_loss_weight = cfg.min_edge_loss_weight
            self.edge_loss_weight_range = max(
                0, self.edge_loss_weight - self.min_edge_loss_weight
            )

        self.min_node_loss_weight = cfg.min_node_loss_weight
        self.max_update = cfg.max_update
        self.node_loss_weight_range = max(
            0, self.node_loss_weight - self.min_node_loss_weight
        )

        self.jac_loss_weight = cfg.jac_loss_weight
        self.compute_jac_loss = cfg.compute_jac_loss
        self.deq_mode = cfg.deq_mode
        self.use_shift_proj = cfg.use_shift_proj
        self.no_node_mask = cfg.no_node_mask
        self.get_inter_pos_trace = cfg.get_inter_pos_trace
        self.visualize = cfg.visualize
        self.noisy_node_weight = cfg.noisy_node_weight
        self.explicit_pos = cfg.explicit_pos
        self.l2_node_loss = cfg.l2_node_loss
        self.fix_atoms = cfg.fix_atoms
        if self.no_node_mask:
            #node norm cleaned full
            IS2RECriterion.d_mean = [0.0004, -0.0015, 0.0167]
            IS2RECriterion.d_std = [0.2367, 0.2566, 0.3020]    
            print("node norm changed")

        if cfg.data.split("/")[-1] == "SAA":
            IS2RECriterion.e_mean = -0.8380
            IS2RECriterion.e_std = 0.7953

            IS2RECriterion.d_mean =[-0.0015,  0.0086,  0.0524]
            IS2RECriterion.d_std = [0.1891, 0.2368, 0.1969] 
            if self.no_node_mask:
                IS2RECriterion.d_mean = [-0.0011,  0.0056,  0.0246]
                IS2RECriterion.d_std = [0.0975, 0.1209, 0.1234] 
                print("node norm changed")

            IS2RECriterion.edge_mean = -0.0013
            IS2RECriterion.edge_std = 0.6475

        if cfg.use_unnormed_node_label:
            IS2RECriterion.d_mean = [0.0,0.0,0.0]
            IS2RECriterion.d_std = [1.0,1.0,1.0]
        print("node norm", IS2RECriterion.d_mean,IS2RECriterion.d_std)


    def forward(
        self,
        model: Callable[..., Tuple[Tensor, Tensor, Tensor]],
        sample: Mapping[str, Mapping[str, Tensor]],
        reduce=True,
    ):
        if 'targets' in sample.keys():

            update_num = model.num_updates
            assert update_num >= 0
            node_loss_weight = (
                self.node_loss_weight
                - self.node_loss_weight_range * update_num / self.max_update
            )
            if self.use_shift_proj:
                edge_loss_weight = (
                    self.edge_loss_weight
                    - self.edge_loss_weight_range * update_num / self.max_update
                )

            use_noisy_node = False
            cell=sample["net_input"]['cell'][:,:3,:]
            sid =sample["net_input"]['sid']
            if "noisy_pos" in sample["net_input"].keys():
                use_noisy_node = True
                noisy_node_weight = self.noisy_node_weight
                if "noisy_label_pos" in sample["net_input"].keys():
                    num_real_instances = sample["targets"]["relaxed_energy"].shape[0]
                    sample["net_input"]['pos']=torch.concat([sample["net_input"]['pos'],sample["net_input"]['noisy_pos'],sample["net_input"]['noisy_label_pos']], dim =0)
                    sample["net_input"]['atoms']=sample["net_input"]['atoms'].repeat(3,1)
                    sample["net_input"]['tags']=sample["net_input"]['tags'].repeat(3,1)
                    sample["net_input"]['real_mask']=sample["net_input"]['real_mask'].repeat(3,1)
                    sample["targets"]["relaxed_energy"]=sample["targets"]["relaxed_energy"].repeat(3)
                    sample["targets"]["deltapos"]=torch.concat([sample["targets"]["deltapos"],sample["targets"]["noisy_deltapos"],sample["targets"]["noisy_label_deltapos"]],dim = 0)
                    cell=sample["net_input"]['cell'].repeat(3,1,1)[:,:3,:]
                    sid = torch.concat([sample["net_input"]['sid'], sample["net_input"]['sid']+90000000, sample["net_input"]['sid']+990000000], dim=0)
                else:
                    num_real_instances = sample["targets"]["relaxed_energy"].shape[0]
                    sample["net_input"]['pos']=torch.concat([sample["net_input"]['pos'],sample["net_input"]['noisy_pos']], dim =0)
                    sample["net_input"]['atoms']=sample["net_input"]['atoms'].repeat(2,1)
                    sample["net_input"]['tags']=sample["net_input"]['tags'].repeat(2,1)
                    sample["net_input"]['real_mask']=sample["net_input"]['real_mask'].repeat(2,1)
                    sample["targets"]["relaxed_energy"]=sample["targets"]["relaxed_energy"].repeat(2)
                    sample["targets"]["deltapos"]=torch.concat([sample["targets"]["deltapos"],sample["targets"]["noisy_deltapos"]],dim = 0)
                    cell=sample["net_input"]['cell'].repeat(2,1,1)[:,:3,:]
                    sid = torch.concat([sample["net_input"]['sid'], sample["net_input"]['sid']+90000000], dim=0)
            sample["net_input"].pop('cell')
            sample['net_input'].pop('sid')
            sample["net_input"]['step']=update_num
            non_padding_mask = sample["net_input"]["atoms"].ne(0) 
            valid_nodes = non_padding_mask.sum()
            non_fix_atom_mask = sample["net_input"]['tags'].ne(0)

            output_vals = model(**sample["net_input"],)   

            output = output_vals['eng_output']
            node_output = output_vals['node_output']
            node_target_mask = output_vals['node_target_mask'] 
            edge_dirs = output_vals['edge_dirs']
            if self.get_inter_pos_trace:
                deltapos_trace = output_vals['deltapos_trace'] 
                temp_visualize = False
                temp_visualize_sid = 2044931
                if (sid == temp_visualize_sid).any():
                    temp_visualize=True
                if self.visualize or temp_visualize:
                    visualize_trace_poscar(sample, cell,sid, output_vals, self.e_mean, self.e_std, self.d_mean, self.d_std, directory="./visualize/SAA/new_tag_fix_training_clean")

            if self.compute_jac_loss:
                jac_loss = output_vals['jac_loss']
                f_deq_nstep = output_vals['f_deq_nstep']
                f_deq_residual = output_vals['f_deq_residual']
            if self.use_shift_proj:
                edge_output = output_vals['fit_scale_pred']
                edge_target_mask = output_vals['edge_target_mask']
            self.drop_edge_training=False
            if 'drop_edge_mask' in output_vals.keys():
                self.drop_edge_training=True
                drop_edge_mask = output_vals['drop_edge_mask']

            relaxed_energy = sample["targets"]["relaxed_energy"]
            relaxed_energy = relaxed_energy.float()
            relaxed_energy = (relaxed_energy - self.e_mean) / self.e_std
            sample_size = relaxed_energy.numel()
            loss = F.l1_loss(output.float().view(-1), relaxed_energy, reduction="none")
            if use_noisy_node:
                loss_ori = loss[:num_real_instances]
                loss_noise = loss[num_real_instances:]
                with torch.no_grad():
                    energy_within_threshold = (loss_ori.detach() * self.e_std < self.e_thresh).sum()
                    energy_within_threshold_100 = (loss_ori.detach() * self.e_std < self.e_thresh_100).sum()
                loss = loss_ori.sum() + self.noisy_node_weight * loss_noise.sum()
            else:
                with torch.no_grad():
                    energy_within_threshold = (loss.detach() * self.e_std < self.e_thresh).sum()
                    energy_within_threshold_100 = (loss.detach() * self.e_std < self.e_thresh_100).sum()
                loss = loss.sum()

            deltapos = sample["targets"]["deltapos"].float()
            deltapos = (deltapos - deltapos.new_tensor(self.d_mean)) / deltapos.new_tensor(
                self.d_std
            )

            deltapos_center = deltapos * node_target_mask
            node_output_center = node_output * node_target_mask
            target_cnt_center = node_target_mask.sum(dim=[1, 2])
            if self.l2_node_loss:
                node_loss_center = (node_output_center.float()-deltapos_center).norm(dim=-1).sum(dim=-1)/target_cnt_center
            else:
                node_loss_center = (
                    F.l1_loss(node_output_center.float(), deltapos_center, reduction="none")
                    .mean(dim=-1)
                    .sum(dim=-1)
                    / target_cnt_center
                )
            if use_noisy_node:
                node_loss_center_ori = node_loss_center[:num_real_instances]
                node_loss_center_noise = node_loss_center[num_real_instances:]
                node_loss_center = node_loss_center_ori.sum()+ self.noisy_node_weight * node_loss_center_noise.sum()
            else:
                node_loss_center = node_loss_center.sum()

            if self.no_node_mask:
                
                if self.fix_atoms:
                    boundary_node_mask = torch.logical_and(~node_target_mask.squeeze(),non_fix_atom_mask).unsqueeze(-1)
                else:
                    boundary_node_mask = non_padding_mask.unsqueeze(-1)&(~node_target_mask)
                target_cnt_boundary = boundary_node_mask.sum(dim=[1, 2])
                zero_boundary_node = (target_cnt_boundary==0).any()
                if zero_boundary_node:
                    node_loss_boundary = torch.zeros(node_target_mask.shape[0]).to(node_target_mask.device)
                    # print(target_cnt_boundary, "skipped")
                else:
                    deltapos_boundary = deltapos * boundary_node_mask
                    node_output_boundary = node_output * boundary_node_mask
                    # print(target_cnt_boundary, "proceed")
                    if self.l2_node_loss:
                        node_loss_boundary = (node_output_boundary.float()-deltapos_boundary).norm(dim=-1).sum(dim=-1)/target_cnt_boundary
                    else:   
                        node_loss_boundary = (
                            F.l1_loss(node_output_boundary.float(), deltapos_boundary, reduction="none")
                            .mean(dim=-1)
                            .sum(dim=-1)
                            / target_cnt_boundary
                        )

                if use_noisy_node:
                    node_loss_boundary_ori = node_loss_boundary[:num_real_instances]
                    node_loss_boundary_noise = node_loss_boundary[num_real_instances:]
                    node_loss_boundary = node_loss_boundary_ori.sum()+ self.noisy_node_weight * node_loss_boundary_noise.sum()
                else:
                    node_loss_boundary = node_loss_boundary.sum()

                node_loss = (node_loss_center+0.5*node_loss_boundary)

            else:
                node_loss = node_loss_center

            if self.use_shift_proj:
                # check number of edge types: edge_target_mask.detach().cpu().int().view(-1).bincount()
                if self.explicit_pos:
                    edge_label = ((sample["targets"]["deltapos"]-output_vals['final_delta_pos']).float().unsqueeze(-2)@edge_dirs.permute(0,1,3,2)) #Raw edge label
                else:
                    edge_label = (sample["targets"]["deltapos"].float().unsqueeze(-2)@edge_dirs.float().permute(0,1,3,2)) #Raw edge label
                edge_label = edge_label.squeeze(-2)
                thres_edge_type=3 #Mar. 21st Note: thres_edge_type to screen edges could be wrong! Edge loss should be calculated based on outedges of node.

                # center_edge_mask = edge_target_mask>thres_edge_type

                center_edge_mask = torch.zeros_like(edge_label)
                center_edge_mask = torch.logical_and(center_edge_mask.masked_fill(node_target_mask, 1).bool(), edge_target_mask!=0)
                if self.drop_edge_training:
                    center_edge_mask = torch.logical_and(center_edge_mask, ~drop_edge_mask)
                edge_label_center = edge_label*center_edge_mask
                edge_output_center = edge_output*center_edge_mask
                target_cnt_center = center_edge_mask.sum(dim=[1, 2])

                edge_loss_center = F.l1_loss(edge_output_center, edge_label_center,reduction = "none").sum(dim=-1).sum(dim=-1)/target_cnt_center
                if use_noisy_node:
                    edge_loss_center_ori = edge_loss_center[:num_real_instances]
                    edge_loss_center_noise = edge_loss_center[num_real_instances:]
                    edge_loss_center = edge_loss_center_ori.sum()+ self.noisy_node_weight * edge_loss_center_noise.sum()
                else:
                    edge_loss_center = edge_loss_center.sum()

                if self.no_node_mask:
                    # boundary_edge_mask = torch.logical_and(edge_target_mask>0, edge_target_mask<=thres_edge_type)
                    if zero_boundary_node:
                        edge_loss_boundary = torch.zeros(center_edge_mask.shape[0]).to(center_edge_mask.device)
                    else:
                        boundary_edge_mask = torch.zeros_like(edge_label)
                        boundary_edge_mask = torch.logical_and(boundary_edge_mask.masked_fill(boundary_node_mask,1).bool(), edge_target_mask!=0)
                        if self.drop_edge_training:
                            boundary_edge_mask = torch.logical_and(boundary_edge_mask, ~drop_edge_mask)
                        edge_label_boundary = edge_label*boundary_edge_mask
                        edge_output_boundary = edge_output*boundary_edge_mask
                        target_cnt_boundary = boundary_edge_mask.sum(dim=[1, 2])

                        edge_loss_boundary = F.l1_loss(edge_output_boundary,edge_label_boundary,reduction = "none").sum(dim=-1).sum(dim=-1)/target_cnt_boundary
                    
                    if use_noisy_node:
                        edge_loss_boundary_ori = edge_loss_boundary[:num_real_instances]
                        edge_loss_boundary_noise = edge_loss_boundary[num_real_instances:]
                        edge_loss_boundary = edge_loss_boundary_ori.sum()+ self.noisy_node_weight * edge_loss_boundary_noise.sum()
                    else:
                        edge_loss_boundary = edge_loss_boundary.sum()

                    edge_loss = edge_loss_center+0.5*edge_loss_boundary
                else:
                    edge_loss = edge_loss_center.sum()


            logging_output = {
                "loss": loss.detach(),
                "energy_within_threshold": energy_within_threshold,
                "energy_within_threshold_100": energy_within_threshold_100,
                "node_loss": node_loss.detach(),
                "sample_size": sample_size,
                "nsentences": sample_size,
                "num_nodes": valid_nodes.detach(),
                "node_loss_weight": node_loss_weight * sample_size,
            }

            total_loss = loss + node_loss_weight * node_loss

            if use_noisy_node:
                total_loss_ori=loss_ori.sum() + node_loss_weight * node_loss_center_ori.sum()
                if self.no_node_mask:
                    total_loss_ori += node_loss_weight * node_loss_boundary_ori.sum()
                if self.use_shift_proj:
                    total_loss_ori += edge_loss_weight * edge_loss_center_ori.sum()
                    if self.no_node_mask:
                        total_loss_ori += edge_loss_weight * edge_loss_boundary_ori.sum()

                total_loss_noise=loss_noise.sum() + node_loss_weight * node_loss_center_noise.sum()
                if self.no_node_mask:
                    total_loss_noise += node_loss_weight * node_loss_boundary_noise.sum()
                if self.use_shift_proj:
                    total_loss_noise += edge_loss_weight * edge_loss_center_noise.sum()
                    if self.no_node_mask:
                        total_loss_noise += edge_loss_weight * edge_loss_boundary_noise.sum()
                logging_output["total_loss_ori"] = total_loss_ori.detach()
                logging_output["total_loss_noise"] = total_loss_noise.detach()

            if self.deq_mode:
                logging_output['f_deq_shrink_ratio']= output_vals['f_deq_shrink_ratio']
            if self.compute_jac_loss:
                logging_output["jac_loss"] = jac_loss.detach()
                logging_output["f_deq_nstep"] = f_deq_nstep
                logging_output["f_deq_residual"] = f_deq_residual
                total_loss = total_loss + self.jac_loss_weight * jac_loss
            
            if self.use_shift_proj:
                logging_output["edge_loss_weight"]=edge_loss_weight * sample_size
                logging_output["edge_loss"] = edge_loss.detach()
                total_loss = total_loss + edge_loss_weight * edge_loss


            return total_loss, sample_size, logging_output

        else:
            update_num = -1
            sample_size = sample['net_input']['pos'].shape[0]
            sample["net_input"]['step']=update_num
            non_padding_mask = sample["net_input"]["atoms"].ne(0) ##also removes nodes that passes boundary. correlated codes in is2re.py
            valid_nodes = non_padding_mask.sum()
            sid = sample['net_input'].pop('sid')
            cell = sample["net_input"].pop('cell')

            output_vals = model(**sample["net_input"],)   
            energy = output_vals['eng_output']*self.e_std+self.e_mean
            return sid, energy, sample_size, None

    @staticmethod
    def reduce_metrics(logging_outputs: Sequence[Mapping]) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_sum_ori = sum(log.get("total_loss_ori", 0) for log in logging_outputs)
        loss_sum_noise = sum(log.get("total_loss_noise", 0) for log in logging_outputs)
        energy_within_threshold_sum = sum(
            log.get("energy_within_threshold", 0) for log in logging_outputs
        )
        energy_within_threshold_sum_100 = sum(
            log.get("energy_within_threshold_100", 0) for log in logging_outputs
        )
        node_loss_sum = sum(log.get("node_loss", 0) for log in logging_outputs)
        edge_loss_sum = sum(log.get("edge_loss",0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        jac_loss_sum = sum(log.get("jac_loss",0) for log in logging_outputs)
        mean_jac_loss = jac_loss_sum 
        f_deq_nstep_sum = sum(log.get("f_deq_nstep",0) for log in logging_outputs)
        f_deq_shrink_ratio_sum = sum(log.get("f_deq_shrink_ratio",0) for log in logging_outputs)
        mean_f_deq_nstep = f_deq_nstep_sum
        f_deq_residual_sum = sum(log.get("f_deq_residual",0) for log in logging_outputs)
        mean_f_deq_residual = (f_deq_residual_sum / sample_size)
        mean_loss = (loss_sum / sample_size) * IS2RECriterion.e_std
        mean_loss_ori = (loss_sum_ori / sample_size) * IS2RECriterion.e_std
        mean_loss_noise = (loss_sum_noise / sample_size) * IS2RECriterion.e_std
        energy_within_threshold = energy_within_threshold_sum / sample_size
        energy_within_threshold_100 = energy_within_threshold_sum_100 / sample_size
        mean_node_loss = (node_loss_sum / sample_size) * sum(IS2RECriterion.d_std) / 3.0
        mean_edge_loss = (edge_loss_sum / sample_size)
        mean_n_nodes = (
            sum([log.get("num_nodes", 0) for log in logging_outputs]) / sample_size
        )
        node_loss_weight = (
            sum([log.get("node_loss_weight", 0) for log in logging_outputs])
            / sample_size
        )
        edge_loss_weight = (
            sum([log.get("edge_loss_weight", 0) for log in logging_outputs])
            / sample_size
        )
        metrics.log_scalar("loss", mean_loss, sample_size, round=6)
        metrics.log_scalar("loss_ori", mean_loss_ori, sample_size, round=6)
        metrics.log_scalar("loss_noise", mean_loss_noise, sample_size, round=6)        
        metrics.log_scalar("ewth", energy_within_threshold, sample_size, round=6)
        metrics.log_scalar("ewth_100", energy_within_threshold_100, sample_size, round=6)
        metrics.log_scalar("node_loss", mean_node_loss, sample_size, round=6)
        metrics.log_scalar("edge_loss", mean_edge_loss, sample_size, round=6)
        metrics.log_scalar("nodes_per_graph", mean_n_nodes, sample_size, round=6)
        metrics.log_scalar("node_loss_weight", node_loss_weight, sample_size, round=6)
        metrics.log_scalar("edge_loss_weight", edge_loss_weight, sample_size, round=6)
        metrics.log_scalar("jac_loss", mean_jac_loss, sample_size, round = 6)
        metrics.log_scalar("f_deq_nstep", mean_f_deq_nstep, sample_size, round = 6)
        metrics.log_scalar("f_deq_shrink_ratio", f_deq_shrink_ratio_sum, sample_size, round = 6)
        metrics.log_scalar("f_deq_residual", mean_f_deq_residual, sample_size, round = 6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
