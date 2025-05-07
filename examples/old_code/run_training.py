def run_training():
    emulator = "highmod_ice"
    needs_preprocessing = False

    if needs_preprocessing:
        from examples.config.training_dict_raw import dict_raw
        from examples.processing_training_data_orig import preprocess_data
        preprocess_data(dict_raw[emulator])

    from examples.config.training_dict_formatted import training_dict as train_dict
    from examples.processing_training_data import process_training

    processed_data = process_training(train_dict[emulator])

if __name__ == "__main__":
    run_training()