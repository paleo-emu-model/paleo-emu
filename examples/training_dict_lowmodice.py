# config_dict.py

config = {
    "input_dir": "/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Input",
    "output_dir": "/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Output/dTeq/LT2000",
    "save_res": "example_outputs/training_data_lowmodice_temp.res",
    "save_nc": "example_outputs/training_data_lowmodice_temp.nc",

    # 加入实验ID列表
    "exp_ids": ["tdum", "tdvo", "tdvp", "tdvq"],

    "res_path_prefix": [
        "", 
        "",
        "",
        ""
    ],

    # the suffix for res file 
    "res_file": ["Samp_orbits_tdum.res",
                   "Samp_orbits_tdvo_LT2000ppm.res", 
                   "Samp_orbits_tdvp.res", 
                   "Samp_orbits_tdvq_LT2000ppm.res"
                   ],

    "nc_path_prefix": [
        "", 
        "",
        "",
        ""
    ],

    # the suffix for nc file
    "nc_file": ["dTeq_temp_mm_1_5m_ann_tdum.nc", 
                  "dTeq_temp_mm_1_5m_ann_tdvo_LT2000ppm.nc", 
                  "dTeq_temp_mm_1_5m_ann_tdvp.nc", 
                  "dTeq_temp_mm_1_5m_ann_tdvq_LT2000ppm.nc"
                  ],

    # 每个实验ID对应的变量名后缀列表
    "postfix_dict": {
        "tdum": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"),
        "tdvo": list("cdghiknopqsuvwyCDEHM"),
        "tdvp": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"),
        "tdvq": list("cdghiknopqsuvwyCDEHM")
    }
}
