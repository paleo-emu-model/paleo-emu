
from paleo_emu.export import save_prediction
from paleo_emu.training import run_training


def full_emulator_experiment(train_dict, emulator_name, output_dir="outputs", vae_config=None):
    """
    全自动尝试所有model+encoder组合，并保存结果。
    """
    # model_kernel_combinations = [
    #     ("GPR", "RBF"),
    #     ("GPR", "RBF_White"),
    #     ("GPR", "Matern_0.5_White"),
    #     ("GPR", "Matern_1.5"),
    #     ("GPR", "RationalQuadratic"),
    #     ("GPR", "Matern_2.5_White"),
    #     ("LGBM", None)  # LGBM不需要kernel
    # ]
    model_kernel_combinations = [
        ("GPR", "Matern_2.5_White"),
        ("LGBM", None)]
 
    encoders = [ "VAE", "PCA"]

    for encoder in encoders:
        for model_type, kernel in model_kernel_combinations:
            print("="*80)
            print(f"[INFO] Training model: {model_type} | Kernel: {kernel} | Encoder: {encoder}")

            emulator = run_training(
                train_dict[emulator_name],
                model_type=model_type,
                kernel=kernel if kernel else "RBF_White",  # 给LGBM随便传一个kernel（无效但占位）
                encoder=encoder,
                vae_config=vae_config,
                return_pred=True
            )

            # 打印得分
            print(f"[RESULT] {model_type} + {encoder} --> Test R² Score: {emulator['gpr_r2_score']:.4f}")

            # 保存预测和真实
            pred_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ypred.nc"
            true_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ytrue.nc"

            save_prediction(emulator["Y_pred_out"], output_dir, pred_filename)
            save_prediction(emulator["Y_True_out"], output_dir, true_filename)
