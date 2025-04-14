# config_dict.py

config = {
    "input_dir": "/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Input",
    "output_dir": "/Users/bo20541/Library/CloudStorage/OneDrive-UniversityofBristol/TONIC-Oligocene/Emulator_Charlie/Emulator/2015_Bristol_5D_v001/orig/Output/dTeq/LT2000",
    "save_res": "example_outputs/training_data_highmodice_temp.res",
    "save_nc": "example_outputs/training_data_highmodice_temp.nc",

    # 加入实验ID列表
    "exp_ids": ["tdum", 
                "tdvo", 
                "tdab28k", 
                "tdab80k", 
                "tdab120k"
                ],

    "res_path_prefix": [
        "", 
        "",
        "Singarayer + Valdes 2010/",
        "Singarayer + Valdes 2010/",
        "Singarayer + Valdes 2010/"
    ],

    # the suffix for res file 
    "res_file": ["Samp_orbits_tdum.res",
                 "Samp_orbits_tdvo_LT2000ppm.res", 
                 "Samp_orbits_tdab28k.res", 
                 "Samp_orbits_tdab80k.res", 
                 "Samp_orbits_tdab120k.res"
                ],

    "nc_path_prefix": [
        "", "",
        "../../Singarayer + Valdes 2010/",
        "../../Singarayer + Valdes 2010/",
        "../../Singarayer + Valdes 2010/"
    ],

    # the suffix for nc file
    "nc_file": ["dTeq_temp_mm_1_5m_ann_tdum.nc", 
                "dTeq_temp_mm_1_5m_ann_tdvo_LT2000ppm.nc", 
                "dT500_temp_mm_1_5m_ann_tdab28k_anomaly.nc", 
                "dT500_temp_mm_1_5m_ann_tdab80k_anomaly.nc", 
                "dT500_temp_mm_1_5m_ann_tdab120k_anomaly.nc"
            ],

    # 每个实验ID对应的变量名后缀列表
    "postfix_dict": {
        "tdum": list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"),
        "tdvo": list("cdghiknopqsuvwyCDEHM"),
        "tdab28k": list("abcdefghijklmnopqrstuvwxyz"),
        "tdab80k": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        "tdab120k": list("0123456789")
    }
}
