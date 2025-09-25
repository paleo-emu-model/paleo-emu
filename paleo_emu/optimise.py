
from paleo_emu.export import save_prediction
from paleo_emu.training import run_training
from paleo_emu.load import load_training_data
import numpy as np
from scipy.optimize import minimize


def full_emulator_experiment(train_dict, emulator_name, output_dir="outputs", vae_config=None):
    """
    Run a full emulator experiment with all model and encoder combinations.
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
                kernel=kernel if kernel else "RBF_White",  # invalid but placeholder
                encoder=encoder,
                vae_config=vae_config,
                return_pred=True
            )

            print(f"[RESULT] {model_type} + {encoder} --> Test R² Score: {emulator['gpr_r2_score']:.4f}")

            # Save predictions and truth
            pred_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ypred.nc"
            true_filename = f"{emulator_name}_{model_type}_{kernel if kernel else 'None'}_{encoder}_Ytrue.nc"

            save_prediction(emulator["Y_pred_out"], output_dir, pred_filename)
            save_prediction(emulator["Y_True_out"], output_dir, true_filename)


def optimize_hyperparameters(train_dict):
    """
    Given training data (X, y), optimize the GP hyperparameters
    (length scales δ and nugget ν) by maximizing the penalized log-likelihood.
    - Work in log-space to ensure positivity
    - Use L-BFGS-B optimizer
    - Return the optimal δ and ν
    """
    X_train, Y_train, _, _, _, _ = load_training_data(train_dict)
    # standardize Y_train
    Y_mean = np.mean(Y_train, axis=0)
    Y_std = np.std(Y_train, axis=0)
    Y_train = (Y_train - Y_mean) / Y_std
    # standardize X_train
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_train = (X_train - X_mean) / X_std
    n_samples, n_features = X_train.shape
    n_outputs = Y_train.shape[1]
    print(f"[INFO] Loaded training data: X_train shape={X_train.shape}, Y_train shape={Y_train.shape}")
    # Initial guess for length scales and nugget
    initial_length_scales = np.log(np.ones(n_features))  # log of 1.0
    initial_nugget = np.log(1e-3)  # log of small nugget
    initial_signal_variance = np.log(100.0) # log of signal variance
    initial_params = np.concatenate([initial_length_scales, [initial_nugget, initial_signal_variance]])  # shape (n_features + 2,)
    print(f"[INFO] Initial hyperparameters (log-space): {initial_params}")
    print(f"[INFO] initial length scales: {np.exp(initial_length_scales)}, initial nugget: {np.exp(initial_nugget)}, initial signal variance: {np.exp(initial_signal_variance)}")
    def cov_mat(lambda_, X1, X2):
        m = X1.shape[1]
        theta = lambda_[0:m]       # 第 1:m 行（注意 Python 的索引从 0 开始）
        nk = len(theta)
        nx, ny = X1.shape[0], X2.shape[0] 
        # Initialize RR
        RR = np.zeros((nx, ny, nk))
        
        # Convert X1 and X2 to numpy arrays if they are not already
        if not isinstance(X1, np.ndarray):
            X1 = X1.to_numpy()
        if not isinstance(X2, np.ndarray):
            X2 = X2.to_numpy()
            
        # Compute scaled differences
        for k in range(nk):
            RR[:, :, k] = np.subtract.outer(X1[:, k], X2[:, k]) / theta[k]

        # Compute sum of squared differences and apply covariance function
        # the value of R(i,j) is the similarity between X(i) and X(j)
        R = np.exp(-np.sum(RR ** 2, axis=2))
        return R

    def neg_penalized_log_likelihood(params):
        weights = np.exp(params[:-2]).reshape(-1)  # δ (weights for linear kernel)
        nugget = np.exp(params[-2])  # ν
        signal_variance = np.exp(params[-1])  # σ²

        # 使用自定义核
        lambda_ = np.concatenate([weights, [nugget, signal_variance]])
        K = cov_mat(lambda_, X_train, X_train)

        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf

        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
        log_det_K = 2.0 * np.sum(np.log(np.diag(L)))

        lambda_penalty = 1.0  # Define a lambda parameter for the penalty
        lambda_penalty2 = 10.0  # Stronger penalty for nugget
        lambda_penalty3 = 10.0  # Penalty for signal variance
        nll = 0.5 * np.sum(Y_train * alpha) + 0.5 * log_det_K + 0.5 * n_samples * n_outputs * np.log(2 * np.pi)
        penalty = lambda_penalty * np.sum((params[:-2] - np.log(1.0)) ** 2)  \
              + lambda_penalty2 * (params[-2] - np.log(1e-1)) ** 2 \
              + lambda_penalty3 * (params[-1] - np.log(2.0)) ** 2
        # print(f"[DEBUG] params: {params}, nll: {nll}")
        # print(f"[DEBUG] mean and variance of K: {np.mean(K)}, {np.var(K)}")
        return nll + penalty
    
    bounds = [(np.log(1e-2), np.log(1e3))] * n_features \
    + [(np.log(1e-7), np.log(1e2)), (np.log(1e-2), np.log(1e4))]  # bounds for weights, nugget, and signal variance
    result = minimize(neg_penalized_log_likelihood, initial_params, method='L-BFGS-B', bounds=bounds)

    if not result.success:
        print(f"[WARNING] Hyperparameter optimization did not converge: {result.message}")

    optimal_params = result.x
    optimal_weights = np.exp(optimal_params[:-2])
    optimal_nugget = np.exp(optimal_params[-2])
    optimal_signal_variance = np.exp(optimal_params[-1])

    print(f"[INFO] Optimized weights: {optimal_weights}")
    print(f"[INFO] Optimized nugget: {optimal_nugget}")
    print(f"[INFO] Optimized signal variance: {optimal_signal_variance}")

    return optimal_weights, optimal_nugget, optimal_signal_variance
