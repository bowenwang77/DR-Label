import re

import os

def combine(file1, file2, output_dir):
    # Function to combine the two input files and write the result to the output directory
    # Your implementation here
    # Read in the contents of file1 and file2
    with open(file1, "r") as f1:
        file1 = f1.readlines()
    with open(file2, "r") as f2:
        file2 = f2.readlines()

    # Rule 1
    file2 = [re.sub("((([0-9\-\.]){8,9}\s){3})[H]", "\\1   F   F   F", line) for line in file2]
    file2 = [re.sub("((([0-9\-\.]){8,9}\s){3})[^H\s]{1,2}", "\\1   T   T   T", line) for line in file2]
    # Rule 2
    file2[0:8] = file1[0:8]

    # Rule 3
    index = file2.index("direct\n")
    file2.insert(index, "Selective dynamics\n")

    # Rule 4
    file2[index+1] = "Direct\n"

    # Write the combined file
    with open(output_dir, "w") as cf:
        cf.writelines(file2)

# Set the directory containing the subdirectories
base_dir = "/Users/bowenwang/ICML/visualization/SAA/pfformer_ep_FT_snoise0_751/full"
base_dir = "/Users/bowenwang/ICML/visualization/pfformer_ep_FT_snoise0_751_correct_tag_rerun"
base_dir = "/Users/bowenwang/ICML/visualization/new_tag_fix_training_clean"
# Iterate over all subdirectories in the base directory
for subdir in os.listdir(base_dir):
    if subdir=="combine_final" or subdir.split(".")[-1]=="zip" or subdir==".DS_Store":
        continue
    sub_path = os.path.join(base_dir, subdir)
    output_dir = os.path.join(base_dir,"combine_final",subdir)
    log_dir = os.path.join(base_dir,subdir,"log.txt")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Check if the subdirectory contains the necessary files
    if "POSCAR_pred_final" in os.listdir(sub_path) and "POSCAR_pred_final_mark_fix" in os.listdir(sub_path):
        # Call the combine function with the two files and the output directory
        combine(os.path.join(sub_path, "POSCAR_pred_final"), os.path.join(sub_path, "POSCAR_pred_final_mark_fix"), os.path.join(output_dir, "POSCAR_pred_combine"))

# Calculate abs error
total=0
cnt=0
for subdir in os.listdir(base_dir):
    if subdir=="combine_final" or subdir.split(".")[-1]=="zip" or subdir==".DS_Store":
        continue
    sub_path = os.path.join(base_dir, subdir)
    output_dir = os.path.join(base_dir,"combine_final",subdir)
    log_dir = os.path.join(base_dir,subdir,"log.txt")
    with open(log_dir, "r") as f1:
        file1 = f1.readlines()
    total+=float(file1[-1].split("visualization")[0].split("Energy Absolute Error: ")[-1])
    cnt+=1
    print(total, total/cnt)
