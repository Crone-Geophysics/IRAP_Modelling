from pathlib import Path
import shutil
import re
import time

input_folder = input("Enter input directory:")
if not input_folder:
    print(F"Using default folder")
    input_folder = Path(__file__).absolute().parent
else:
    input_folder = Path(input_folder).absolute()
print(f"Input folder: {input_folder}")

output_folder = input("Enter output directory:")
if not output_folder:
    print(F"Using default folder")
    output_folder = Path(__file__).absolute().parent
else:
    output_folder = Path(output_folder).absolute()
print(f"Output folder: {output_folder}")


def copy_dat_files():
    files = list(input_folder.rglob("*.dat"))
    files_found = '\n'.join([str(f.name) for f in files])
    print(f"Files found: {files_found}")
    
    print(f"Copying files to {output_folder.name}")
    for file in files:
        print(F"{file} to {output_folder.joinpath(file.name)}")
        shutil.copy(str(file), str(output_folder.joinpath(file.name)))


def format_file_names():
    print("Removing '3D_modelling_results_Crone_50ms_Model8_'")
 
    new_files = list(output_folder.glob("*.dat"))
    for file in new_files:
        if "3D_modelling_results_Crone_50ms_Model8_" in file.name:
            print(f"Renaming {file.name} to {re.sub('3D_modelling_results_Crone_50ms_Model8_', '', file.name)}")
            file.rename(output_folder.joinpath(re.sub("3D_modelling_results_Crone_50ms_Model8_", "", file.name)))
    
    print("Removing '_dBdt'")
    new_files = list(output_folder.glob("*.dat"))
    print(f"Files found to remove dBdt: {new_files}")
    for file in new_files:
        if "_dBdt" in file.name:
            print(f"Removing 'dBdt' from {file.name}")
            file.rename(output_folder.joinpath(re.sub("_dBdt", "", file.name)))


def rename_files():
    def get_model_num(file_name):
        match = re.search(r"c(\d+)", str(file_name))
        if match:
            print(F"Model number for {file_name}: {match.group(1)}")
            return match.group(1)
        else:
            print(f"No model number found in {file_name}")
            return None
     
    def get_conductance(file_name):
        match = re.search(r"(\d+)S", str(file_name.upper()))
        if match:
            print(F"Condutance for {file_name}: {match.group(1)}")
            return match.group(1)
        else:
            print(f"No conductance found in {file_name}")
            return None

    print("Renaming files to correct model name")
    for file in list(output_folder.glob("*.dat")):
        print(f"File: {file.name}")
        file_num = get_model_num(file.name)
        file_con = get_conductance(file.name)
        
        if file_con == '100' or file_con is None:
            model_name = model_names.get(file_num)
        else:
            model_name = f"{file_num}@{file_con}S"
        print(f"Model name for file {file.name}: {model_name}")
        
        file.rename(output_folder.joinpath(model_name + '.dat'))
        
model_names = {
"1":"1",
"2":"2",
"3":"3",
"4":"4",
"5":"5",
"6":"6",
"7":"1_2",
"8":"1+2",
"9":"1_4",
"10":"1+4",
"11":"2_3",
"12":"2+3",
"13":"2_4",
"14":"2+4",
"15":"2_5",
"16":"2+5",
"17":"4_5",
"18":"4+5",
"19":"5_3",
"20":"5+3",
"21":"3_6",
"22":"3+6",
"23":"1_4_5",
"24":"1+4+5",
"25":"5_3_6",
"26":"5+3+6",
"27":"1_2_3",
"28":"1+2+3",
"29":"1_2_3_6",
"30":"1+2+3+6",
"31":"(1+2)_(3+6)",
"32":"(1+2+3)_6",
"33":"1_4_5_3_6",
"34":"1+4+5+3+6",
"35":"(1+4+5)_(3+6)",
"36":"1@10S",
"37":"1@1000S",
"38":"4@1000S",
"39":"1@10S_4@100S",
"40":"1@10S_4@1000S",
"41":"1@100S_4@1000S",
"42":"1@10S+4@100S",
"43":"1@10S+4@1000S",
"44":"1@100S+4@1000S",
"45":"1@10S_2@100S",
"46":"1@10S+2@100S",
}

copy_dat_files()
time.sleep(1)
format_file_names()
time.sleep(1)
rename_files()
    
print("Process complete.")

