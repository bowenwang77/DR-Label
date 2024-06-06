import numpy as np
import torch

def result_average(file_list,save_dir):
    data_list=[]
    name = ""
    for file in file_list:
        data_list.append(np.load(file))
        name+=file.split("/")[-1][10:13]
    num_recs=len(data_list)
    save_data={}
    save_data['id_ids']=data_list[0]['id_ids']
    save_data['ood_ads_ids']=data_list[0]['ood_ads_ids']
    save_data['ood_both_ids']=data_list[0]['ood_both_ids']
    save_data['ood_cat_ids']=data_list[0]['ood_cat_ids']

    averaged_id=np.zeros_like(data_list[0]['id_energy'])
    averaged_ads=np.zeros_like(data_list[0]['ood_ads_energy'])
    averaged_both=np.zeros_like(data_list[0]['ood_both_energy'])
    averaged_cat=np.zeros_like(data_list[0]['ood_cat_energy'])
    for i in range(num_recs):
        averaged_id +=data_list[i]['id_energy']/num_recs
        averaged_ads +=data_list[i]['ood_ads_energy']/num_recs
        averaged_both +=data_list[i]['ood_both_energy']/num_recs
        averaged_cat +=data_list[i]['ood_cat_energy']/num_recs     

    save_data['id_energy']=averaged_id+1e-8
    save_data['ood_ads_energy']=averaged_ads+1e-8
    save_data['ood_both_energy']=averaged_both+1e-8
    save_data['ood_cat_energy']=averaged_cat+1e-8
    np.savez_compressed(
        save_dir+name,
        **save_data
    )
    print("submission file saved",save_dir+name)
    

file_list=[
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint54_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint58_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint60_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint62_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint66_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint68_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_ep_freq6/checkpoint70_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4/checkpoint_best_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4/checkpoint48_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4/checkpoint68_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/jiaze/bw_checkpoint/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/checkpoint108_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/jiaze/bw_checkpoint/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/checkpoint128_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/jiaze/bw_checkpoint/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full/checkpoint_last_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/jiaze/bw_checkpoint/pnf_noisy_IEF_cleanotlrs_epad_newnnorm_dw_full_cont/checkpoint6_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint68_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint48_submit.npz",
"/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint56_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint42_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint62_submit.npz",
# "/bowen/ocp22/ocp22/ocpmodels/models/graphormer_private/graphormer/bw_checkpoint/mine_cluster/pfformer_snoise_bat4_cont/pfformer_snoise_bat4_cont/checkpoint60_submit.npz",
]
save_dir=file_list[0][:-4]

for file in file_list: print(file)
print(save_dir)
result_average(file_list,save_dir)


