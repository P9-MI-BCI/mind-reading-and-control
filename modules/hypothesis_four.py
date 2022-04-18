from data_preprocessing.init_dataset import get_dataset_paths


# Hypothesis four aims to test if training cross subject with subject specific calibration can improve the accuracy, and
# achieve a higher accuracy than the other hypothesis.
def run(config):
    """
    Calibration will improve the accuracy of the model when predicting on a new subject.
    """
    subject_id = int(input("Choose subject to predict on 0-9\n"))
    config.transfer_learning = True
    config.rest_classification = True

    training_dataset_path, online_dataset_path, dwell_dataset_path = get_dataset_paths(subject_id, config)


    pass