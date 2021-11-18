from data_preprocessing.data_distribution import create_uniform_distribution
from data_preprocessing.data_shift import shift_data
from data_preprocessing.mrcp_detection import mrcp_detection
from data_preprocessing.optimize_windows import prune_poor_quality_samples, remove_windows_with_blink, \
    remove_worst_windows
from data_preprocessing.train_test_split import train_test_split_data
from data_preprocessing.trigger_points import trigger_time_table
from data_training.KNN.knn_prediction import knn_classifier_loocv
from data_training.LDA.lda_prediction import lda_classifier_loocv
from data_training.SVM.svm_prediction import svm_classifier_loocv
from data_visualization.average_channels import average_channel, plot_average_channels
from data_visualization.visualize_windows import visualize_labeled_windows, visualize_windows, \
    visualize_window_all_channels
from utility.pdf_creation import save_results_to_pdf_2
from utility.save_and_load import save_train_test_split, load_train_test_split
from utility.logger import get_logger


def offline(script_params, config, dataset):

    # Shift Data to remove startup
    dataset = shift_data(freq=config.start_time, dataset=dataset)

    # Create table containing information when trigger points were shown/removed
    trigger_table = trigger_time_table(dataset.TriggerPoint, dataset.time_start_device1)

    if script_params.run_mrcp_detection:
        # Perform MRCP Detection and update trigger_table with EMG timestamps
        windows, trigger_table = mrcp_detection(data=dataset, tp_table=trigger_table, config=config)

        avg_channels = average_channel(windows)
        plot_average_channels(avg_channels, config)

        uniform_data = create_uniform_distribution(windows)
        train_data, test_data = train_test_split_data(uniform_data, split_per=20)

        save_train_test_split(train_data, test_data, dir_name=f'online_test')

    if script_params.run_classification:
        train_data, test_data = load_train_test_split(dir_name=f'online_test')

        # in LOOCV everything is test and train data
        train_data.extend(test_data)

        feature = 'feature_vec'
        get_logger().info('LOOCV with KNN. ')
        knn_score, model = knn_classifier_loocv(train_data, features=feature, prediction='w')
        get_logger().info('LOOCV with SVM. ')
        svm_score, model = svm_classifier_loocv(train_data, features=feature, prediction='w')
        get_logger().info('LOOCV with LDA. ')
        lda_score, model = lda_classifier_loocv(train_data, features=feature, prediction='w')

        results = {
            'KNN_results': knn_score,
            'SVM_results': svm_score,
            'LDA_results': lda_score
        }

        # Writes the test and train window plots + classifier score tables to pdf file
        save_results_to_pdf_2(train_data, results, file_name='yxtest_eeg_overview.pdf', save_fig=False)
