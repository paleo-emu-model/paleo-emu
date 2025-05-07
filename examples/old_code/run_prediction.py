def run_prediction(forcing = "rcp85.1", emulator = "lowmod_ice"):
    

    from examples.config.prediction_dict import forcing_dict
    from examples.config.training_dict_formatted import training_dict
    from examples.Prediction import process_prediction

    processed_data = process_prediction(forcing_dict[forcing], training_dict[emulator])

if __name__ == "__main__":
    run_prediction(forcing="rcp35.1")