import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import os

def visualize_trace(pc_trace_delta,pos,label_delta,non_padding_mask,directory="visualize/default"):
    fig = plt.figure(figsize=(50,50))
    num_trace = pc_trace_delta.shape[0]
    num_frame = pc_trace_delta.shape[1]

    for trace_id in range(num_trace):
        mins = pos[trace_id][non_padding_mask[trace_id]].min(axis=0)
        maxs = pos[trace_id][non_padding_mask[trace_id]].max(axis=0)
        spans = maxs-mins
        trace_inst = pc_trace_delta[trace_id]
        for frame_id in range(num_frame):
            # pc = pc_trace_delta[trace_id,frame_id]+pos[trace_id]
            pc = pc_trace_delta[trace_id,frame_id][non_padding_mask[trace_id]]+pos[trace_id][non_padding_mask[trace_id]]
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            pc_l = label_delta[trace_id][non_padding_mask[trace_id]]+pos[trace_id][non_padding_mask[trace_id]]
            x_l = pc_l[:, 0]
            y_l = pc_l[:, 1]
            z_l = pc_l[:, 2]
            ax = fig.add_subplot(num_trace,num_frame, trace_id*num_frame+(frame_id+1), projection ='3d')
            ax.scatter(x,y,z,marker='o',c='b')
            ax.scatter(x_l,y_l,z_l,marker='x',c='r')
            ax.set_xlim([mins[0]-0.1*spans[0],maxs[0]+0.1*spans[0]])
            ax.set_ylim([mins[1]-0.1*spans[1],maxs[1]+0.1*spans[1]])
            ax.set_zlim([mins[2]-0.1*spans[2],maxs[2]+0.1*spans[2]])
            label_drift_sum = np.linalg.norm(label_delta[trace_id],axis=1).sum()
            pred_drift_sum = np.linalg.norm(label_delta[trace_id]-pc_trace_delta[trace_id,frame_id],axis=1).sum()
            print(trace_id,frame_id,pred_drift_sum/label_drift_sum)
            ax.set_title("Pos remains:"+str(pred_drift_sum/label_drift_sum))
    plt.savefig(directory)

    fig = plt.figure(figsize=(50,50))
    num_trace = pc_trace_delta.shape[0]
    num_frame = pc_trace_delta.shape[1]

    for trace_id in range(num_trace):
        trace_inst = pc_trace_delta[trace_id]
        all_delta_pos = np.concatenate([pc_trace_delta[trace_id].reshape(-1,3),label_delta[trace_id].reshape(-1,3)],axis=0)
        mins = all_delta_pos.min(axis=0)
        maxs = all_delta_pos.max(axis=0)
        spans = maxs-mins
        for frame_id in range(num_frame):
            # pc = pc_trace_delta[trace_id,frame_id]+pos[trace_id]
            pc = pc_trace_delta[trace_id,frame_id][non_padding_mask[trace_id]]
            x = pc[:, 0]
            y = pc[:, 1]
            z = pc[:, 2]
            pc_l = label_delta[trace_id][non_padding_mask[trace_id]]
            x_l = pc_l[:, 0]
            y_l = pc_l[:, 1]
            z_l = pc_l[:, 2]
            ax = fig.add_subplot(num_trace,num_frame, trace_id*num_frame+(frame_id+1), projection ='3d')
            ax.scatter(x,y,z,marker='o',c='b')
            ax.scatter(x_l,y_l,z_l,marker='x',c='r')
            ax.set_xlim([mins[0]-0.1*spans[0],maxs[0]+0.1*spans[0]])
            ax.set_ylim([mins[1]-0.1*spans[1],maxs[1]+0.1*spans[1]])
            ax.set_zlim([mins[2]-0.1*spans[2],maxs[2]+0.1*spans[2]])
            label_drift_sum = np.linalg.norm(label_delta[trace_id],axis=1).sum()
            pred_drift_sum = np.linalg.norm(label_delta[trace_id]-pc_trace_delta[trace_id,frame_id],axis=1).sum()
            print(trace_id,frame_id,pred_drift_sum/label_drift_sum)
            ax.set_title("Pos remains:"+str(pred_drift_sum/label_drift_sum))
    plt.savefig(directory+"only_drift")

def visualize_trace_poscar(sample, cell,sid, output_vals,e_mean, e_std, d_mean, d_std, directory="visualize/default"):
    import pymatgen
    import numpy as np
    atom_list = [
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
    symbols = ['H','He','Li','Be','B','C','N','O','F','Ne',
            'Na','Mg','Al','Si','P','S','Cl','Ar','K', 'Ca',
            'Sc', 'Ti', 'V','Cr', 'Mn', 'Fe', 'Co', 'Ni',
            'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
            'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
            'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
            'I', 'Xe','Cs', 'Ba','La', 'Ce', 'Pr', 'Nd', 'Pm',
            'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm',
            'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir',
            'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
            'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
            'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
            'Rf', 'Db', 'Sg', 'Bh','Hs', 'Mt', 'Ds', 'Rg', 'Cn',
            'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']


    atom_list=np.array(atom_list)
    node_output=output_vals['node_output']
    deltapos_trace = output_vals['deltapos_trace']
    num_trace= deltapos_trace.shape[0]
    num_frames = deltapos_trace.shape[1]
    directory_root = directory
    for idx in range(num_trace):
        lattice = np.array(cell[idx].cpu())
        lattice = pymatgen.core.lattice.Lattice(lattice)
        real_mask = sample["net_input"]['real_mask'][idx].cpu()
        tag=np.array(sample["net_input"]['tags'][idx].cpu())[real_mask]
        atom=np.array(sample["net_input"]['atoms'][idx].cpu())[real_mask]
        atom=atom_list[atom-1]
        atom_mark_fix = atom.copy()
        atom_mark_fix[tag==0]=1
        directory=directory_root+"/POSCAR"+str(sid[idx].item())
        if directory.split("/")[2]=="SAA": 
            directory = directory+"_"+symbols[atom[-3]-1]
        if not os.path.exists(directory):
            os.makedirs(directory)
        log=open(directory+"/log.txt","w+")

        position_init=np.array(sample["net_input"]['pos'][idx].cpu())[real_mask]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_init,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_init")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_init,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_init_mark_fix")        

        position_relax=position_init.copy()+np.array(sample["targets"]["deltapos"][idx].cpu())[real_mask]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_relax,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_relaxed")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_relax,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_relaxed_mark_fix")        

        for frame_idx in range(num_frames):
            delta_position_pred = np.array((deltapos_trace[idx,frame_idx]*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean)).detach().cpu())[real_mask]
            position_pred=position_init.copy()
            position_pred[tag!=0]+=delta_position_pred[tag!=0]
            ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
            Poscar = pymatgen.io.vasp.Poscar(ins)
            Poscar.write_file(directory+"/POSCAR_pred_interframe"+str(frame_idx))
            ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
            Poscar = pymatgen.io.vasp.Poscar(ins)
            Poscar.write_file(directory+"/POSCAR_pred_interframe_mark_fix"+str(frame_idx))        

            label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
            pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
            print(sid[idx],frame_idx," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(),)
            log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()),'\n'])

        delta_position_pred=np.array((node_output*node_output.new_tensor(d_std)+node_output.new_tensor(d_mean))[idx].detach().cpu())[real_mask]
        position_pred=position_init.copy()
        position_pred[tag!=0]+=delta_position_pred[tag!=0]
        ins = pymatgen.core.structure.IStructure(lattice,atom,position_pred,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_pred_final")
        ins = pymatgen.core.structure.IStructure(lattice,atom_mark_fix,position_pred,coords_are_cartesian=True)
        Poscar = pymatgen.io.vasp.Poscar(ins)
        Poscar.write_file(directory+"/POSCAR_pred_final_mark_fix")        

        label_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu(),axis=1)
        pred_remain_norm = np.linalg.norm(sample["targets"]["deltapos"][idx][real_mask][tag!=0].detach().cpu()-delta_position_pred[tag!=0],axis=1)
        print(sid[idx]," final pos pred. "," Pred remain MAE: ",pred_remain_norm.mean()," Label remain MAE: ",label_remain_norm.mean()," Relative remain ratio: ",pred_remain_norm.mean()/label_remain_norm.mean(),)
        log.writelines([str(sid[idx].item()),str(frame_idx)," Pred remain MAE: ",str(pred_remain_norm.mean())," Label remain MAE: ",str(label_remain_norm.mean())," Relative remain ratio: ",str(pred_remain_norm.mean()/label_remain_norm.mean()),'\n'])

        print("Energy label: ",sample['targets']['relaxed_energy'][idx].item(), " Energy prediction: ", output_vals['eng_output'][idx].item()*e_std+e_mean, " Energy Absolute Error: ", abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['eng_output'][idx].item()*e_std+e_mean)))
        log.writelines(["Energy label: ",str(sample['targets']['relaxed_energy'][idx].item()), " Energy prediction: ", str(output_vals['eng_output'][idx].item()*e_std+e_mean), " Energy Absolute Error: ", str(abs(sample['targets']['relaxed_energy'][idx].item()-(output_vals['eng_output'][idx].item()*e_std+e_mean)))])

        print("visualization file saved in: ", directory)
        log.writelines(["visualization file saved in: ", directory])
        log.close()

        # Visualize
        # import pymatgen
        # lattice = np.array(cell)
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("POSCAR")


        #vis backup
        # import pymatgen
        # import ase
        # import numpy as np

        # coord = np.array([0.1,0.2,0.3])
        # lattice = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ele = pymatgen.core.periodic_table.Element.from_Z(10)
        # site = pymatgen.core.sites.PeriodicSite(ele,coord,lattice)
        # sites = [site,site]
        # mol = pymatgen.core.structure.IStructure.from_sites(sites)
        # Poscar = pymatgen.io.vasp.Poscar(mol)
        # Poscar.write_file("POSCAR")


        ##Vis
        # import pymatgen
        # lattice = np.array(cell)
        # lattice = pymatgen.core.lattice.Lattice(lattice)
        # ins = pymatgen.core.structure.IStructure(lattice,np.array(atoms),np.array(pos),coords_are_cartesian=True)
        # Poscar = pymatgen.io.vasp.Poscar(ins)
        # Poscar.write_file("POSCAR")
        # sites = []
        # for i in range(len(pos)):
        #     coord = np.array(pos[i])
        #     ele = pymatgen.core.periodic_table.Element.from_Z(np.array(atoms[i]))
        #     site = pymatgen.core.sites.PeriodicSite(ele,coord,lattice)
        #     sites.append(site)
        #     pass
        # ins = pymatgen.core.structure.IStructure.from_sites(sites)

