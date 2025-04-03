import pandas as pd

input_folder="/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Input/"
input_path1=input_folder+"Samp_orbits_tdum.res"
input_path2=input_folder+"Samp_orbits_tdvo_LT2000ppm.res"
input_path3=input_folder+"Samp_orbits_tdvp.res"
input_path4=input_folder+"Samp_orbits_tdvq_LT2000ppm.res"
input_path5=input_folder+"Samp_orbits_tdum.res"
input_path6=input_folder+"Samp_orbits_tdvo_LT2000ppm.res"
input_path7=input_folder+"Singarayer + Valdes 2010/Samp_orbits_tdab.res"

input_paths = [input_path1, input_path2, input_path3, input_path4, input_path5, input_path6, input_path7]

input_data = pd.DataFrame(columns=['co2','obliquity','esinw','ecosw','ice'])

for i in range(8):
    emu_input_data = pd.read_csv(input_paths[i], sep='\s+', header=0)
    input_data = pd.concat([input_data, emu_input_data[['co2', 'obliquity', 'esinw', 'ecosw', 'ice']]], ignore_index=True)

print("shape of emulator input data (5 variables):",input_data.shape)

input_data.to_csv('emul_in_X_Char.res', sep='\t', index=False)
