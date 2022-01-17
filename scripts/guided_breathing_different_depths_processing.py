import os
import pandas as pd
from portable_eit_01.utils.utils import load_data, select_reference_and_denoise_raw_data, reconstruct_eit_images, \
    setup_eit, spatio_temporal_filtering, compute_clustering_threshold, get_lung_clusters, mask_maps_with_clusters, \
    compute_global_waveform_and_general_functional_maps, compute_regional_indicators, \
    normalize_indicators_and_maps_by_subject, visualize_results, save_pickle, save_temp_dir, results_folder


paradigm = 'guided_breathing_different_depths'
raw_images_path = os.path.join(save_temp_dir(paradigm), 'raw_images.pkl')
filtered_images_path = os.path.join(save_temp_dir(paradigm), 'filtered_images.pkl')
general_outputs_path = os.path.join(save_temp_dir(paradigm), 'general_outputs.pkl')
clusters_and_rois_path = os.path.join(save_temp_dir(paradigm), 'clusters_and_rois.pkl')
general_regional_indicators_path = os.path.join(save_temp_dir(paradigm), 'general_regional_indicators.pkl')


if __name__ == "__main__":
    data_info_csv = os.path.join(os.path.dirname(os.getcwd()), 'data', paradigm, 'data_info.csv')
    data_info_df = pd.read_csv(data_info_csv)

    for data_row in data_info_df.iterrows():
        subject = data_row[1]['subject']
        rawdata_filepath = os.path.join(os.path.dirname(data_info_csv), subject, data_row[1]['data_file'])
        breathing_depth = data_row[1]['breathing_depth']
        system = data_row[1]['system']
        result_filepath = os.path.join(results_folder(paradigm), data_row[1]['data_file'].replace('.pkl', '_result.pkl'))

        print(subject, '|', breathing_depth, '|', data_row[1]['data_file'])
        if not os.path.isfile(rawdata_filepath) or os.path.isfile(result_filepath):
            continue

        raw_data = load_data(rawdata_filepath, system)
        denoised_raw_data = select_reference_and_denoise_raw_data(raw_data, system)
        eit = setup_eit(system)
        raw_images = reconstruct_eit_images(denoised_raw_data, eit, save_temp_path=raw_images_path)
        filtered_images = spatio_temporal_filtering(raw_images_path, temporal_filtering_method='butterworth', save_temp_path=filtered_images_path)
        general_outputs_dict = compute_global_waveform_and_general_functional_maps(filtered_images_path, amplitude_percentile=50, paradigm=paradigm, save_temp_path=general_outputs_path)
        clustering_threshold = compute_clustering_threshold(general_outputs_dict)
        clusters_and_rois = get_lung_clusters(general_outputs_path, clustering_threshold, save_temp_path=clusters_and_rois_path)
        clustered_general_functional_maps = mask_maps_with_clusters(general_outputs_dict, clusters_and_rois)
        general_regional_indicators = compute_regional_indicators(filtered_images_path, general_outputs_path, clusters_and_rois_path, save_temp_path=general_regional_indicators_path)

        result_dict = {
            'subject': subject, 'breathing_depth': breathing_depth,
            'clustered_general_functional_maps': clustered_general_functional_maps,
            'general_regional_indicators': general_regional_indicators
        }
        save_pickle(result_filepath, result_dict)

    normalize_indicators_and_maps_by_subject(data_info_df, paradigm=paradigm)
    visualize_results(data_info_df, paradigm=paradigm, subject='average')
