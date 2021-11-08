from classes.Simulation import Simulation
from data_preprocessing.data_distribution import create_uniform_distribution
from data_preprocessing.mrcp_detection import mrcp_detection_for_online_use, load_index_list, pair_index_list
from data_preprocessing.optimize_windows import remove_worst_windows, prune_poor_quality_samples
from data_preprocessing.train_test_split import train_test_split_data
from data_preprocessing.trigger_points import trigger_time_table
from data_training.KNN.knn_prediction import knn_classifier
from data_training.LDA.lda_prediction import lda_classifier
from data_training.SVM.svm_prediction import svm_classifier
from data_training.online_emulation import simulate_online, evaluate_online_predictions
from data_training.scikit_classifiers import load_scikit_classifiers
from data_visualization.average_channels import average_channel, plot_average_channels
from utility.pdf_creation import save_results_to_pdf
from utility.save_and_load import save_train_test_split, load_train_test_split
from utility.logger import get_logger


def online(script_params, config, dataset):
    # Create table containing information when trigger points were shown/removed
    trigger_table = trigger_time_table(dataset.TriggerPoint, dataset.time_start_device1)

    simulation = Simulation(config)
    simulation.mount_dataset(dataset)
    simulation.evaluation_metrics()
    simulation.simulate(real_time=True)

    if script_params['run_mrcp_detection']:
        windows, trigger_table = mrcp_detection_for_online_use(data=dataset, tp_table=trigger_table, config=config)

        prune_poor_quality_samples(windows, trigger_table, config, remove=9, method=remove_worst_windows)
        avg_windows = average_channel(windows)
        plot_average_channels(avg_windows, save_fig=False, overwrite=True)

        uniform_data = create_uniform_distribution(windows)
        train_data, test_data = train_test_split_data(uniform_data, split_per=20)

        save_train_test_split(train_data, test_data, dir_name='online_EEG')

    if script_params['run_classification']:
        train_data, test_data = load_train_test_split(dir_name='online_EEG')

        feature = 'features'
        knn_score = knn_classifier(train_data, test_data, features=feature)
        svm_score = svm_classifier(train_data, test_data, features=feature)
        lda_score = lda_classifier(train_data, test_data, features=feature)

        results = {
            'KNN_results': knn_score,
            'SVM_results': svm_score,
            'LDA_results': lda_score
        }

        # Writes the test and train window plots + classifier score tables to pdf file
        save_results_to_pdf(train_data, test_data, results, file_name='result_overview.pdf')

    if script_params['run_online_emulation']:
        models = load_scikit_classifiers('knn')
        index = load_index_list()
        pair_indexes = pair_index_list(index)

        get_logger().info('Starting Online Predictions.')
        windows_on, predictions = simulate_online(dataset, config, models, features='features', continuous=True)
        get_logger().info('Finished Online Predictions.')

        score = evaluate_online_predictions(windows_on, predictions, pair_indexes)
        print(score)
