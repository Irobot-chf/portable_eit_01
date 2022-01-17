import os
import pandas as pd

from portable_eit_01.utils.utils import select_reference_and_denoise_raw_data, setup_eit, \
    reconstruct_eit_images, compute_clustering_threshold, load_data, save_pickle, save_temp_dir, subject_id_key, \
    raw_eit_data_filename_key, effort_level_key, results_folder, visualize_results, \
    compute_group_clustering_threshold, normalize_indicators_and_maps_by_subject, spatio_temporal_filtering, \
    compute_global_forced_breathing_indicators, compute_forced_breathing_functional_maps, \
    compute_global_waveform_and_general_functional_maps

paradigm = 'forced_breathing'
data_info_csv = os.path.join(os.path.dirname(os.getcwd()), 'data', 'forced_breathing', 'data_info.csv')
raw_data_dir = os.path.dirname(data_info_csv)
result_save_folder = results_folder(paradigm)

save_temp_path_raw_images = os.path.join(save_temp_dir(paradigm), 'raw_images.pkl')
save_temp_path_filtered_images = os.path.join(save_temp_dir(paradigm), 'filtered_images.pkl')
save_temp_path_general_outputs = os.path.join(save_temp_dir(paradigm), 'general_outputs.pkl')
save_temp_path_global_forced_breathing_indicators = os.path.join(save_temp_dir(paradigm), 'global_forced_breathing_indicators.pkl')
save_temp_path_forced_breathing_indicators_maps = os.path.join(save_temp_dir(paradigm), 'forced_breathing_indicators_maps.pkl')


if __name__ == '__main__':
    system = 1
    eit = setup_eit(system)
    data_info_df = pd.read_csv(data_info_csv)

    for i in data_info_df.index:
        raw_eit_data_filename = data_info_df[raw_eit_data_filename_key][i]
        subject_id = data_info_df[subject_id_key][i]
        effort_level = data_info_df[effort_level_key][i]
        print(subject_id, '|', effort_level, '|', raw_eit_data_filename)
        rawdata_filepath = os.path.join(raw_data_dir, subject_id, raw_eit_data_filename)
        result_save_path = os.path.join(result_save_folder, '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0]))
        if not os.path.isfile(rawdata_filepath) or os.path.isfile(result_save_path):
            continue
        raw_data = load_data(rawdata_filepath, system=system)
        denoised_raw_data = select_reference_and_denoise_raw_data(raw_data, system=system)
        raw_images = reconstruct_eit_images(denoised_raw_data, eit, save_temp_path=save_temp_path_raw_images)
        filtered_images = spatio_temporal_filtering(save_temp_path_raw_images, temporal_filtering_method='moving_average',
                                                    save_temp_path=save_temp_path_filtered_images)
        general_outputs_dict = compute_global_waveform_and_general_functional_maps(save_temp_path_filtered_images,
                                                                                   amplitude_percentile=100, paradigm=paradigm, save_temp_path=save_temp_path_general_outputs)
        global_forced_breathing_indicators = compute_global_forced_breathing_indicators(save_temp_path_general_outputs, save_temp_path=save_temp_path_global_forced_breathing_indicators)
        forced_breathing_indicators_maps = compute_forced_breathing_functional_maps(save_temp_path_filtered_images,
                                                                                    save_temp_path_global_forced_breathing_indicators,
                                                                                    save_temp_path_general_outputs,
                                                                                    save_temp_path_forced_breathing_indicators_maps)
        clustering_threshold = compute_clustering_threshold(general_outputs_dict)

        output_dict = {'general_outputs_dict': general_outputs_dict,
                       'global_forced_breathing_indicators': global_forced_breathing_indicators,
                       'forced_breathing_indicators_maps': forced_breathing_indicators_maps,
                       'clustering_threshold': clustering_threshold}
        save_pickle(result_save_path, output_dict)

    subjects_forced_breathing_group_threshold = compute_group_clustering_threshold(data_info_df, paradigm)
    normalize_indicators_and_maps_by_subject(data_info_df, paradigm=paradigm)
    visualize_results(data_info_df, paradigm=paradigm, subject='average')
