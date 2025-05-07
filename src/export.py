

# ===== 模块 2：特征提取模块（PCA / VAE） =====
def save_training_log(epoch_losses, seed, latent_dim, epochs, learning_rate, batch_size, kl_weight, log_dir="training/logs"):
    """
    保存VAE训练日志，包括：
    - loss曲线图
    - 超参数+最终loss的CSV记录
    """

    # --- 创建logs目录 ---
    os.makedirs(log_dir, exist_ok=True)

    # --- 统一格式化信息 ---
    info_str = f"seed{seed}_latent{latent_dim}_ep{epochs}_lr{learning_rate}_bs{batch_size}_kl{kl_weight}"

    # --- 保存loss曲线 ---
    loss_curve_filename = os.path.join(log_dir, f"loss_curve_{info_str}.png")

    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(epoch_losses)+1), epoch_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"VAE Loss Curve ({info_str})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_curve_filename, dpi=300)
    plt.close()

    print(f"[INFO] Loss curve saved to: {loss_curve_filename}")

    # --- 保存超参数和最终loss到CSV ---
    log_file = os.path.join(log_dir, "vae_hyperparameter_log.csv")

    log_entry = {
        "seed": seed,
        "latent_dim": latent_dim,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "kl_weight": kl_weight,
        "final_loss": epoch_losses[-1]  # 最后一个epoch的loss
    }

    if not os.path.exists(log_file):
        df = pd.DataFrame([log_entry])
        df.to_csv(log_file, index=False)
    else:
        df = pd.read_csv(log_file)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
        df.to_csv(log_file, index=False)

    print(f"[INFO] Hyperparameter log updated: {log_file}")