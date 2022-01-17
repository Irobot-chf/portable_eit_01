import os
import sys
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.metrics import mean_absolute_percentage_error

from sklearn.model_selection import LeavePGroupsOut, train_test_split
from random import randint, seed

pyeit_path = '..\\..\\pyEIT'
if pyeit_path not in sys.path:
    sys.path.append(pyeit_path)

import pyeit.eit.jac as jac
from pyeit.eit.utils import eit_scan_lines


save_temp_dir = lambda paradigm: os.path.join(os.path.dirname(os.getcwd()), 'cache', 'temp', paradigm)
forced_breathing_indicators = ['FVC', 'FEV1', 'FEV1/FVC', 'PEF', 'FEF2575']
label_name = {'FVC': 'MVE (L)', 'FEV1': 'EV1 (L)', 'FEV1/FVC': 'EV1/MVE', 'PEF': 'MEF (L/s)', 'FEF2575': 'EF25-75% (L/s)'}
normalized_label_name = {'FVC': 'MVE', 'FEV1': 'EV1', 'FEV1/FVC': 'EV1/MVE', 'PEF': 'MEF', 'FEF2575': 'EF25-75%'}
effort_level_label_name = {1: 'Full inhale,\nfast exhale', 2: 'Full inhale,\nslow exhale',
                           3: 'Mid inhale,\nfast exhale', 4: 'Mid inhale,\nslow exhale'}
gain_missed = {1: 1, 2: 6}

results_folder = lambda paradigm: os.path.join(os.path.dirname(os.getcwd()), 'cache', 'results', paradigm)
note = 'Note that these results are from partial data set\n'

subject_spirometry_id_key = 'spirometry_id'
subject_id_key = 'participant_id'
raw_eit_data_filename_key = 'raw-data filename'
chest_circumference_key = 'chest circumference'
weight_key = 'weight'
height_key = 'height'
effort_level_key = 'Effort_Level'


def load_pickle(pickle_filepath):
    pikd = open(pickle_filepath, "rb")
    data = pickle.load(pikd)
    pikd.close()
    return data


def save_pickle(pickle_filepath, data):
    if pickle_filepath is not None:
        if not os.path.isdir(os.path.dirname(pickle_filepath)):
            os.makedirs(os.path.dirname(pickle_filepath))
        with open(pickle_filepath, 'wb') as handle:
            pickle.dump(data, handle)


def copy_or_load(dict_or_path):
    if isinstance(dict_or_path, str):
        dict_or_path_use = load_pickle(dict_or_path)
    else:
        dict_or_path_use = dict_or_path.copy()
    return dict_or_path_use


def load_data(rawdata_filepath, system=1):

    rawdata_dict = load_pickle(rawdata_filepath)
    data = np.array(rawdata_dict['data'])
    time = np.array(rawdata_dict['time'])
    if np.max(data) < 1:
        # (If the voltage was measured in Volts convert to mVolts)
        data = data * 1000
    if type(time[0]) in [np.int64, int, np.int32, np.int_, np.int]:
        # (If the time was measured in mSeconds convert to Seconds)
        time = time / 1000
    data = data * gain_missed[system]
    return {'data': data, 'time': time-time[0]}


def select_reference_and_denoise_raw_data(raw_data, system=1):
    reference_data_frame = np.mean(raw_data['data'], axis=0)
    max_threshold = 150 if system == 2 else 1000
    min_threshold = 0.6 if system == 2 else 0
    raw_data_copy = raw_data['data'].copy()
    reference_data_frame_copy = reference_data_frame.copy()
    bad_indices_series = np.where(np.logical_or((raw_data_copy > max_threshold), (raw_data_copy < min_threshold)))
    if len(bad_indices_series[0]) > 0:
        for ele in np.transpose(bad_indices_series):
            raw_data_copy[ele[0]][ele[1]] = reference_data_frame[ele[1]]
    reference_data_bad_indices = np.where(np.logical_or((reference_data_frame_copy > max_threshold), (reference_data_frame_copy < min_threshold)))
    denoised_reference_data_frame = np.mean(raw_data_copy, axis=0) if len(reference_data_bad_indices[0]) > 0 else reference_data_frame_copy
    return {'data': raw_data_copy, 'time': raw_data['time'], 'reference_data': denoised_reference_data_frame}


def setup_eit(system=1):
    eit_save_path = os.path.join(os.path.dirname(os.getcwd()), 'cache', 'eit_setup_system_{}.pkl'.format(system))

    if os.path.isfile(eit_save_path):
        eit = load_pickle(eit_save_path)
    else:
        '''Setup for inverse problem'''
        mesh = load_pickle(os.path.join(os.path.dirname(os.getcwd()), 'data', 'eit_mesh.pickle'))
        mesh_obj = mesh['mesh_obj']
        el_pos = mesh['el_pos'][::-1] if system == 2 else mesh['el_pos']

        ex_mat = eit_scan_lines(ne=16, dist=1)
        eit = jac.JAC(mesh_obj, el_pos, ex_mat, step=1, perm=1.0, parser='fmmu')
        eit.setup(p=0.35, lamb=0.005, method='kotre')
        save_pickle(eit_save_path, eit)
    return eit


def reconstruct_eit_images(denoised_raw_data, eit, save_temp_path=None):
    data_series = denoised_raw_data['data']
    reference_data_frame = denoised_raw_data['reference_data']
    img_tri = np.array([eit.solve(data_series[idx], reference_data_frame, normalize=False) for idx in range(data_series.shape[0])])
    raw_images = {'img_tri': img_tri, 'time': denoised_raw_data['time'], 'mesh': eit.mesh}
    save_pickle(save_temp_path, raw_images)
    return raw_images


def compute_clustering_threshold(general_outputs_dict):
    correlation_amplitude_product = np.multiply(general_outputs_dict['amplitude_map'], general_outputs_dict['correlation_map'])
    return np.percentile(correlation_amplitude_product[~general_outputs_dict['mesh']['mask']], 65)


def compute_group_clustering_threshold(data_info_df, paradigm):
    result_save_folder = results_folder(paradigm)
    subjects_forced_breathing_group_threshold = {}
    for subjects_spirometry_id in data_info_df[subject_spirometry_id_key].unique():
        data_info_sub_df = data_info_df[data_info_df[subject_spirometry_id_key] == subjects_spirometry_id]
        effort_level_for_group_threshold = max(data_info_sub_df['Effort_Level'].unique())
        data_info_sub_df_effort = data_info_sub_df[
            data_info_sub_df[effort_level_key] == effort_level_for_group_threshold]
        raw_eit_data_filename_list = data_info_sub_df_effort[raw_eit_data_filename_key].to_list()

        group_clustering_threshold_list = []
        for raw_eit_data_filename in raw_eit_data_filename_list:
            processing_results = load_pickle(os.path.join(result_save_folder,
                                            '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0])))
            group_clustering_threshold_list.append(processing_results['clustering_threshold'])
        subjects_forced_breathing_group_threshold[subjects_spirometry_id] = np.mean(group_clustering_threshold_list)
    save_pickle(os.path.join(result_save_folder, 'forced_breathing_group_threshold.pkl'),
                subjects_forced_breathing_group_threshold)
    return subjects_forced_breathing_group_threshold


def mask_maps_with_clusters(general_outputs_dict, clusters_and_rois):
    clustered_general_functional_maps = {'mesh': general_outputs_dict['mesh']}
    for general_output_dict_key in general_outputs_dict.keys():
        if 'map' not in general_output_dict_key:
            continue
        masked_map = general_outputs_dict[general_output_dict_key].copy()
        masked_map[~clusters_and_rois['lung_clusters']] = 0
        masked_map[general_outputs_dict['mesh']['mask']] = np.nan
        clustered_general_functional_maps[general_output_dict_key] = masked_map
    return clustered_general_functional_maps


def draw_image(image, mesh, ax=None, vmin=None, vmax=None, plt_show=False, title=None, labelsize=8, show_text=True):
    if not ax:
        fig, ax = plt.subplots(1)
    mask = mesh['mask']
    img_show = image.copy()
    img_show[mask] = np.nan
    Nxy = 64
    xs = mesh['xy'][:, 0].reshape((Nxy, Nxy))[0, :]
    ys = mesh['xy'][:, 1].reshape((Nxy, Nxy))[:, 0]
    im = ax.pcolor(xs, ys, img_show.reshape((Nxy, Nxy)), edgecolors=None, linewidth=0, alpha=0.8,
                   antialiased=True, zorder=-1, cmap='jet', snap=True, shading='auto', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_axis_off()
    if show_text:
        ax.set_title(title)
        cb = plt.colorbar(im, ax=ax)
        cb.ax.tick_params(labelsize=labelsize)
    if plt_show:
        plt.show()


def normalize_indicators_and_maps_by_subject(data_info_df, paradigm):
    if paradigm == 'forced_breathing':
        result_save_folder = results_folder(paradigm)
        save_temp_path_general_outputs_i = os.path.join(save_temp_dir(paradigm), 'general_outputs_i.pkl')
        """===========================Load group_threshold====================="""
        subjects_forced_breathing_group_threshold = load_pickle(
            os.path.join(result_save_folder, 'forced_breathing_group_threshold.pkl'))
        """===========================Image segmentation====================="""
        for i in data_info_df.index:
            raw_eit_data_filename = data_info_df[raw_eit_data_filename_key][i]
            result_save_path = os.path.join(result_save_folder,
                                            '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0]))
            lung_clusters_save_path = result_save_path.replace('_result.pkl', '_clusters.pkl')
            if os.path.isfile(lung_clusters_save_path):
                continue
            processing_results = load_pickle(result_save_path)
            general_outputs_dict = processing_results['general_outputs_dict']
            subjects_spirometry_id = data_info_df[subject_spirometry_id_key][i]

            save_pickle(save_temp_path_general_outputs_i, general_outputs_dict)
            get_lung_clusters(save_temp_path_general_outputs_i,
                              clustering_threshold=subjects_forced_breathing_group_threshold[subjects_spirometry_id],
                              save_temp_path=lung_clusters_save_path)

        """===========================Normalization====================="""
        subjects_spirometry_group_normalization_factor = {}
        subjects_spirometry_id_list = data_info_df[subject_spirometry_id_key].unique()
        for subjects_spirometry_id in subjects_spirometry_id_list:
            data_info_sub_df = data_info_df[data_info_df[subject_spirometry_id_key] == subjects_spirometry_id]

            normalization_factor_list = {indicator: [] for indicator in forced_breathing_indicators}
            raw_eit_data_filename_list = data_info_sub_df[raw_eit_data_filename_key].to_list()

            for raw_eit_data_filename in raw_eit_data_filename_list:
                result_save_path = os.path.join(result_save_folder,
                                                '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0]))
                processing_results = load_pickle(result_save_path)
                lung_clusters_save_path = result_save_path.replace('_result.pkl', '_clusters.pkl')
                lung_clusters_and_rois = load_pickle(lung_clusters_save_path)
                forced_breathing_indicators_maps = processing_results['forced_breathing_indicators_maps']
                for indicator in forced_breathing_indicators:
                    if indicator != 'FEV1/FVC':
                        normalization_factor_list[indicator].append(np.nanmax(
                            forced_breathing_indicators_maps[indicator][lung_clusters_and_rois['lung_clusters']]))
                    else:
                        normalization_factor_list[indicator].append(1)
            subjects_spirometry_group_normalization_factor[subjects_spirometry_id] = {
                indicator: np.nanmax(normalization_factor_list[indicator]) for indicator in forced_breathing_indicators}
            print(subjects_spirometry_id, subjects_spirometry_group_normalization_factor[subjects_spirometry_id])

        forced_breathing_group_normalization_factor_path = os.path.join(result_save_folder,
                                                                           'forced_breathing_group_normalization_factors.pkl')
        save_pickle(forced_breathing_group_normalization_factor_path,
                    subjects_spirometry_group_normalization_factor)

    elif paradigm == 'guided_breathing_different_depths':
        for subject in list(data_info_df['subject'].unique()):
            subject_data_info_df = data_info_df[data_info_df['subject'] == subject]
            normalization_factors = {}
            for data_row in subject_data_info_df.iterrows():
                data_file = data_row[1]['data_file']
                result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
                data_file_results = load_pickle(result_filepath)
                for indicator_key in ['amplitude_map', 'regional_waveform', 'freq_spectrum']:
                    if indicator_key == 'amplitude_map':
                        indicator = np.abs(data_file_results['clustered_general_functional_maps'][indicator_key])
                        if indicator_key not in normalization_factors.keys():
                            normalization_factors[indicator_key] = np.nanmax(indicator)
                        else:
                            normalization_factors[indicator_key] = max(normalization_factors[indicator_key], np.nanmax(indicator))
                    else:
                        indicator_left = np.abs(data_file_results['general_regional_indicators'][indicator_key]['left'])
                        indicator_right = np.abs(data_file_results['general_regional_indicators'][indicator_key]['right'])
                        if indicator_key not in normalization_factors.keys():
                            normalization_factors[indicator_key] = {'left': np.nanmax(indicator_left), 'right': np.nanmax(indicator_right)}
                        else:
                            normalization_factors[indicator_key]['left'] = max(normalization_factors[indicator_key]['left'], np.nanmax(indicator_left))
                            normalization_factors[indicator_key]['right'] = max(normalization_factors[indicator_key]['right'], np.nanmax(indicator_right))
            for data_row in subject_data_info_df.iterrows():
                data_file = data_row[1]['data_file']
                result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
                data_file_results = load_pickle(result_filepath)
                for indicator_key in ['amplitude_map', 'regional_waveform', 'freq_spectrum']:
                    if indicator_key == 'amplitude_map':
                        data_file_results['clustered_general_functional_maps']['normalization_factor_{}'.format(indicator_key)] = normalization_factors[indicator_key]
                    else:
                        data_file_results['general_regional_indicators']['normalization_factor_{}'.format(indicator_key)] = {
                            'left': normalization_factors[indicator_key]['left'],
                            'right': normalization_factors[indicator_key]['right']
                        }
                save_pickle(result_filepath, data_file_results)

    elif paradigm == 'guided_breathing_COVID19':
        for data_row in data_info_df.iterrows():
            data_file = data_row[1]['data_file']
            result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
            data_file_results = load_pickle(result_filepath)

            amplitude_map = data_file_results['clustered_general_functional_maps']['amplitude_map']
            data_file_results['clustered_general_functional_maps']['normalization_factor_amplitude_map'] = np.nanmax(amplitude_map)
            save_pickle(result_filepath, data_file_results)

        for subject in list(data_info_df['subject'].unique()):
            subject_data_info_df = data_info_df[data_info_df['subject'] == subject]
            normalization_factors = {}
            for data_row in subject_data_info_df.iterrows():
                data_file = data_row[1]['data_file']
                result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
                data_file_results = load_pickle(result_filepath)
                for indicator_key in ['cv']:
                    if indicator_key == 'amplitude_map':
                        indicator = np.abs(data_file_results['clustered_general_functional_maps'][indicator_key])
                        if indicator_key not in normalization_factors.keys():
                            normalization_factors[indicator_key] = np.nanmax(indicator)
                        else:
                            normalization_factors[indicator_key] = max(normalization_factors[indicator_key],
                                                                       np.nanmax(indicator))
                    elif indicator_key == 'cv':
                        if indicator_key not in normalization_factors.keys():
                            normalization_factors[indicator_key] = [
                                data_file_results['general_regional_indicators']['SNR']]
                        else:
                            normalization_factors[indicator_key].append(
                                data_file_results['general_regional_indicators']['SNR'])
                    else:
                        indicator_left = np.abs(data_file_results['general_regional_indicators'][indicator_key]['left'])
                        indicator_right = np.abs(
                            data_file_results['general_regional_indicators'][indicator_key]['right'])
                        if indicator_key not in normalization_factors.keys():
                            normalization_factors[indicator_key] = {'left': np.nanmax(indicator_left),
                                                                    'right': np.nanmax(indicator_right)}
                        else:
                            normalization_factors[indicator_key]['left'] = max(
                                normalization_factors[indicator_key]['left'], np.nanmax(indicator_left))
                            normalization_factors[indicator_key]['right'] = max(
                                normalization_factors[indicator_key]['right'], np.nanmax(indicator_right))
            for data_row in subject_data_info_df.iterrows():
                data_file = data_row[1]['data_file']
                result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
                data_file_results = load_pickle(result_filepath)
                for indicator_key in ['cv']:
                    if indicator_key == 'amplitude_map':
                        data_file_results['clustered_general_functional_maps'][
                            'normalization_factor_{}'.format(indicator_key)] = normalization_factors[indicator_key]
                    elif indicator_key == 'cv':
                        data_file_results['general_regional_indicators'][
                            'normalization_factor_{}'.format(indicator_key)] = np.mean(
                            np.array(normalization_factors[indicator_key]))
                    else:
                        data_file_results['general_regional_indicators'][
                            'normalization_factor_{}'.format(indicator_key)] = {
                            'left': normalization_factors[indicator_key]['left'],
                            'right': normalization_factors[indicator_key]['right']
                        }
                save_pickle(result_filepath, data_file_results)

    else:
        raise Exception("Paradigm {} not defined".format(paradigm))


def visualize_results(data_info_df, paradigm, subject='average'):
    if paradigm == 'forced_breathing':
        result_save_folder = results_folder(paradigm)

        """===========================Predict forced_breathing indicators====================="""
        global_forced_breathing_indicators_list = {indicator: [] for indicator in forced_breathing_indicators}
        for i in data_info_df.index:
            raw_eit_data_filename = data_info_df[raw_eit_data_filename_key][i]
            result_save_path = os.path.join(result_save_folder,
                                            '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0]))
            processing_results = load_pickle(result_save_path)
            global_forced_breathing_indicators = processing_results['global_forced_breathing_indicators']
            for indicator in forced_breathing_indicators:
                global_forced_breathing_indicators_list[indicator].append(global_forced_breathing_indicators[indicator])
        predicted_indicators = predict_spirometry_indicators_from_forced_breathing_indicators(
            global_forced_breathing_indicators_list,
            data_info_df[chest_circumference_key],
            data_info_df[weight_key], data_info_df[height_key])

        """===========================Scatter plots of indicators====================="""
        train_idx, test_idx = custom_train_test_split(data_info_df)
        fig, axs = plt.subplots(1, len(forced_breathing_indicators), figsize=(20, 6))
        fig.suptitle(note)
        for i, indicator in enumerate(forced_breathing_indicators):
            if indicator == 'FEV1/FVC':
                normalization = 100
            elif indicator == 'PEF':
                normalization = 60
            else:
                normalization = 1
            xy_min = -100 / normalization if indicator == 'PEF' else 0
            xy_max = 1.1 * np.max(np.divide(data_info_df['{}_x'.format(indicator)], normalization))
            title = label_name[indicator]
            for idx, color, set in zip([train_idx, test_idx], ['b', 'r'], ['Train', 'Test']):
                slope, intercept, r_value, pv, se = stats.linregress(np.array(predicted_indicators[indicator])[idx],
                                                                     np.array(data_info_df['{}_x'.format(indicator)])[
                                                                         idx])
                mape = mean_absolute_percentage_error(np.array(data_info_df['{}_x'.format(indicator)])[idx],
                                                      np.array(predicted_indicators[indicator])[idx])
                title = title + '\n' + r'{}: $\rho={}$ | MAE%={} | ${}$'.format(set, np.round(r_value, 2),
                                                                                round(mape, 2), format_pval(pv))
                axs[i].scatter(np.divide(np.array(predicted_indicators[indicator])[idx], normalization),
                               np.divide(np.array(data_info_df['{}_x'.format(indicator)])[idx], normalization), c=color)
            axs[i].plot([xy_min, xy_max], [xy_min, xy_max], ':', color='grey', linewidth=2)
            axs[i].set_title(title, fontsize=8)
            axs[i].set_xlim(xy_min, xy_max)
            axs[i].set_ylim(xy_min, xy_max)
        """===========================Load normalization factors====================="""
        forced_breathing_group_normalization_factor_path = os.path.join(result_save_folder, 'forced_breathing_group_normalization_factors.pkl')
        subjects_spirometry_group_normalization_factor = load_pickle(forced_breathing_group_normalization_factor_path)
        """===========================Compute average maps====================="""
        effort_level_list = data_info_df[effort_level_key].unique()
        avg_indicator_maps_levels = {}
        for effort_level in effort_level_list:
            data_info_sub_df = data_info_df[data_info_df[effort_level_key] == effort_level]
            avg_indicator_maps = {indicator: np.zeros(64 ** 2) for indicator in forced_breathing_indicators}
            n_data = {indicator: 0 for indicator in forced_breathing_indicators}
            for i in data_info_sub_df.index:
                raw_eit_data_filename = data_info_sub_df[raw_eit_data_filename_key][i]
                subjects_spirometry_id = data_info_sub_df[subject_spirometry_id_key][i]
                result_save_path = os.path.join(result_save_folder,
                                                '{}_result.pkl'.format(raw_eit_data_filename.split('.')[0]))
                lung_clusters_save_path = result_save_path.replace('_result.pkl', '_clusters.pkl')
                processing_results = load_pickle(result_save_path)
                lung_clusters_and_rois = load_pickle(lung_clusters_save_path)
                forced_breathing_indicators_maps = processing_results['forced_breathing_indicators_maps']
                for indicator in forced_breathing_indicators:
                    indicator_map = forced_breathing_indicators_maps[indicator]
                    if np.sum(np.isnan(indicator_map)) > 0:
                        continue
                    else:
                        n_data[indicator] +=1
                    lung_clusters = lung_clusters_and_rois['lung_clusters']
                    avg_indicator_maps[indicator][lung_clusters] += (indicator_map[lung_clusters] /
                                                                     subjects_spirometry_group_normalization_factor[
                                                                         subjects_spirometry_id][indicator])
            for indicator in forced_breathing_indicators:
                if indicator != 'FEV1/FVC':
                    avg_indicator_maps[indicator][avg_indicator_maps[indicator] < 0.1] = 0
            print(n_data)
            avg_indicator_maps_levels[str(effort_level)] = {indicator: avg_indicator_maps[indicator]/n_data[indicator] for indicator in forced_breathing_indicators}
        """===========================Draw images====================="""
        show_text = True
        for indicator in forced_breathing_indicators:
            vmin = 0
            vmax = 1 if indicator == 'FEV1/FVC' else 0.7

            fig, axs = plt.subplots(2, 2)
            if show_text:
                fig.suptitle(note + normalized_label_name[indicator])
            for effort_level in effort_level_list:
                avg_indicator_map = avg_indicator_maps_levels[str(effort_level)][indicator]
                mesh = processing_results['general_outputs_dict']['mesh']
                draw_image(avg_indicator_map, mesh, ax=axs[int((effort_level - 1) / 2), (effort_level - 1) % 2],
                           vmin=vmin, vmax=vmax, title=effort_level_label_name[effort_level], show_text=show_text)

    elif paradigm == 'guided_breathing_different_depths':
        activated_voxels_left = []
        activated_voxels_right = []
        total_amplitude_left = []
        total_amplitude_right = []
        breathing_depth = []

        time_unified = list(np.arange(0, 60, 1/10))
        times = []
        freqs = []
        regional_waveforms_left = []
        regional_waveforms_right = []
        freq_spectra_left = []
        freq_spectra_right = []
        breathing_depth_long = []

        amplitude_images = []

        for data_row in data_info_df.iterrows():
            data_file = data_row[1]['data_file']
            result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
            data_file_results = load_pickle(result_filepath)
            if subject != data_file_results['subject'] and subject != 'average':
                continue
            activated_voxels_left.append(data_file_results['general_regional_indicators']['activated_voxels']['left'])
            activated_voxels_right.append(data_file_results['general_regional_indicators']['activated_voxels']['right'])
            total_amplitude_left.append(data_file_results['general_regional_indicators']['total_amplitude']['left'])
            total_amplitude_right.append(data_file_results['general_regional_indicators']['total_amplitude']['right'])
            breathing_depth.append(data_file_results['breathing_depth'])
            times = times + time_unified
            freqs = freqs + list(data_file_results['general_regional_indicators']['freq'] * 60)
            regional_waveforms_left = regional_waveforms_left + list(data_file_results['general_regional_indicators']['regional_waveform']['left']/data_file_results['general_regional_indicators']['normalization_factor_regional_waveform']['left'])
            regional_waveforms_right = regional_waveforms_right + list(data_file_results['general_regional_indicators']['regional_waveform']['right']/data_file_results['general_regional_indicators']['normalization_factor_regional_waveform']['right'])
            freq_spectra_left = freq_spectra_left + list(data_file_results['general_regional_indicators']['freq_spectrum']['left']/data_file_results['general_regional_indicators']['normalization_factor_freq_spectrum']['left'])
            freq_spectra_right = freq_spectra_right + list(data_file_results['general_regional_indicators']['freq_spectrum']['right']/data_file_results['general_regional_indicators']['normalization_factor_freq_spectrum']['right'])
            breathing_depth_long = breathing_depth_long + list(np.repeat(data_file_results['breathing_depth'], len(time_unified)))
            amplitude_images = amplitude_images + list(
                data_file_results['clustered_general_functional_maps']['amplitude_map'] /
                data_file_results['clustered_general_functional_maps']['normalization_factor_amplitude_map']
            )

        amplitude_images = np.array(amplitude_images).reshape(-1, 4096)
        group = np.array(breathing_depth)
        categories = np.unique(group)
        fig, axs = plt.subplots(1, categories.size)
        for idx_category, category in enumerate(list(categories)):
            img_show = np.mean(amplitude_images[group == category], axis=0)
            draw_image(img_show, data_file_results['clustered_general_functional_maps']['mesh'], ax=axs[idx_category], vmin=0, vmax=1, plt_show=False, title=category)
        fig.suptitle(note+'Average amplitude image (a.u.)')

        barplot_df = pd.DataFrame(dict(
            activated_voxels_left=activated_voxels_left, activated_voxels_right=activated_voxels_right,
            total_amplitude_left=total_amplitude_left, total_amplitude_right=total_amplitude_right,
            breathing_depth=breathing_depth
        ))
        melted_df = pd.melt(barplot_df, id_vars=['breathing_depth'], value_vars=['activated_voxels_left', 'activated_voxels_right'], var_name="ROI")
        plt.figure()
        ax = sns.barplot(x="breathing_depth", y="value", hue="ROI", data=melted_df)
        ax.set(xlabel='Breathing depth', ylabel='Activated voxels')
        ax.set_title(note + 'Activated voxels')
        melted_df = pd.melt(barplot_df, id_vars=['breathing_depth'], value_vars=['total_amplitude_left', 'total_amplitude_right'], var_name="ROI")
        plt.figure()
        ax = sns.barplot(x="breathing_depth", y="value", hue="ROI", data=melted_df)
        ax.set(xlabel='Breathing depth', ylabel='Total amplitude (a.u.)')
        ax.set_title(note + 'Total amplitude (a.u.)')

        lineplot_df = pd.DataFrame(dict(
            regional_waveforms_left=regional_waveforms_left, regional_waveforms_right=regional_waveforms_right,
            freq_spectra_left=freq_spectra_left, freq_spectra_right=freq_spectra_right,
            breathing_depth=breathing_depth_long, time=times, freq=freqs
        ))
        melted_df = pd.melt(lineplot_df, id_vars=['breathing_depth', 'time'], value_vars=['regional_waveforms_left', 'regional_waveforms_right'], var_name="ROI")
        axplot = sns.relplot(data=melted_df, x="time", y="value", hue="ROI", col="breathing_depth", kind="line")
        axplot.set(xlabel='Time (s)', ylabel='Amplitude (a.u.)')
        axplot.fig.subplots_adjust(top=0.85)
        axplot.fig.suptitle(note+'Regional waveforms')
        melted_df = pd.melt(lineplot_df, id_vars=['breathing_depth', 'freq'], value_vars=['freq_spectra_left', 'freq_spectra_right'], var_name="ROI")
        axplot = sns.relplot(data=melted_df, x="freq", y="value", hue="ROI", col="breathing_depth", kind="line")
        axplot.set(xlabel='Breaths per minute (Bpm)', ylabel='Power (a.u.)')
        axplot.set(xlim=(0, 30))
        axplot.fig.subplots_adjust(top=0.85)
        axplot.fig.suptitle(note+'Frequency spectra')

    elif paradigm == 'guided_breathing_COVID19':
        group = []
        activated_voxels_left = []
        activated_voxels_right = []
        total_amplitude_left = []
        total_amplitude_right = []

        subjects = []
        days = []
        cvs_left = []
        cvs_right = []
        cvs_aleft = []
        cvs_aright = []
        cvs_pleft = []
        cvs_pright = []

        amplitude_images = []

        for data_row in data_info_df.iterrows():
            data_file = data_row[1]['data_file']
            result_filepath = os.path.join(results_folder(paradigm), data_file.replace('.pkl', '_result.pkl'))
            data_file_results = load_pickle(result_filepath)
            if subject != data_file_results['subject'] and subject != 'average':
                continue

            group_name = data_file_results['group_name']
            group.append(data_file_results[group_name])
            activated_voxels_left.append(data_file_results['general_regional_indicators']['activated_voxels']['left'])
            activated_voxels_right.append(data_file_results['general_regional_indicators']['activated_voxels']['right'])
            total_amplitude_left.append(data_file_results['general_regional_indicators']['total_amplitude']['left'])
            total_amplitude_right.append(data_file_results['general_regional_indicators']['total_amplitude']['right'])

            subjects.append(data_row[1]['subject'])
            days.append(data_row[1]['day'])
            cvs_left.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['left']/data_file_results['general_regional_indicators']['normalization_factor_cv'])
            cvs_right.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['right']/data_file_results['general_regional_indicators']['normalization_factor_cv'])
            cvs_aleft.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['anterior_left']/data_file_results['general_regional_indicators']['normalization_factor_cv'])
            cvs_aright.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['anterior_right']/data_file_results['general_regional_indicators']['normalization_factor_cv'])
            cvs_pleft.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['posterior_left']/data_file_results['general_regional_indicators']['normalization_factor_cv'])
            cvs_pright.append(data_file_results['general_regional_indicators']['coefficient_of_variation']['posterior_right']/data_file_results['general_regional_indicators']['normalization_factor_cv'])

            amplitude_images = amplitude_images + list(
                data_file_results['clustered_general_functional_maps']['amplitude_map'] /
                data_file_results['clustered_general_functional_maps']['normalization_factor_amplitude_map']
            )

        amplitude_images = np.array(amplitude_images).reshape(-1, 4096)
        group = np.array(group)
        categories = np.unique(group)
        fig, axs = plt.subplots(1, categories.size)
        for idx_category, category in enumerate(list(categories)):
            img_show = np.mean(amplitude_images[group == category], axis=0)
            try:
                draw_image(img_show, data_file_results['clustered_general_functional_maps']['mesh'], ax=axs[idx_category],
                           vmin=0, vmax=1, plt_show=False, title=category)
            except:
                draw_image(img_show, data_file_results['clustered_general_functional_maps']['mesh'], ax=axs,
                           vmin=0, vmax=1, plt_show=False, title=category)
        fig.suptitle(note + 'Average amplitude image (a.u.)')

        barplot_df = pd.DataFrame({
            'activated_voxels_left': activated_voxels_left, 'activated_voxels_right': activated_voxels_right,
            'total_amplitude_left': total_amplitude_left, 'total_amplitude_right': total_amplitude_right,
            'cv_left': cvs_left, 'cv_right': cvs_right, group_name: group
        })
        melted_df = pd.melt(barplot_df, id_vars=[group_name], value_vars=['activated_voxels_left', 'activated_voxels_right'])
        plt.figure()
        ax = sns.barplot(x=group_name, y="value", hue="variable", data=melted_df)
        ax.set(xlabel=group_name, ylabel='Activated voxels')
        ax.set_title(note+'Activated voxels')
        melted_df = pd.melt(barplot_df, id_vars=[group_name], value_vars=['total_amplitude_left', 'total_amplitude_right'])
        plt.figure()
        ax = sns.barplot(x=group_name, y="value", hue="variable", data=melted_df)
        ax.set(xlabel=group_name, ylabel='Total amplitude (a.u.)')
        ax.set_title(note + 'Total amplitude (a.u.)')
        melted_df = pd.melt(barplot_df, id_vars=[group_name], value_vars=['cv_left', 'cv_right'])
        plt.figure()
        ax = sns.barplot(x=group_name, y="value", hue="variable", data=melted_df)
        ax.set(xlabel=group_name, ylabel='Coefficient of Variation')
        ax.set_title(note + 'Coefficient of Variation')

        lmplot_df = pd.DataFrame({
            'cv_left': cvs_left, 'cv_right': cvs_right, 'cv_aleft': cvs_aleft, 'cv_aright': cvs_aright,
            'cv_pleft': cvs_pleft, 'cv_pright': cvs_pright, 'subject': subjects, 'day': days, 'group': group
        })
        lmplot_df = lmplot_df.groupby(['day', 'subject', 'group'], as_index=False).mean()
        col_order = sorted(list(lmplot_df['subject'].unique()))
        region_names = {
            'cv_left': 'left', 'cv_right': 'right', 'cv_aleft': 'anterior left',
            'cv_aright': 'anterior right',  'cv_pleft': 'posterior left', 'cv_pright': 'posterior right'
        }
        for cv_region in ['cv_left', 'cv_right', 'cv_aleft', 'cv_aright', 'cv_pleft', 'cv_pright']:
            axplot = sns.lmplot(x="day", y=cv_region, hue="group", col="subject", data=lmplot_df, line_kws={'label': "Linear Reg"}, legend=True, col_order=col_order, ci=None)
            for col_idx, col in enumerate(col_order):
                subject_df = lmplot_df[lmplot_df['subject'] == col]
                slope, intercept, r_value, pv, se = stats.linregress(subject_df['day'], subject_df[cv_region])
                ax = axplot.axes[0, col_idx]
                ax.legend()
                leg = ax.get_legend()
                L_labels = leg.get_texts()
                L_labels[0].set_text('r = {}'.format(np.round(r_value,2)))
                L_labels[1].set_text(format_pval(pv))
            axplot.set(ylim=(0, 0.06))
            axplot.fig.subplots_adjust(top=0.85)
            axplot.fig.suptitle(note+'C.V. ({}) vs. Time'.format(region_names[cv_region]))

    else:
        raise Exception("Paradigm {} not defined".format(paradigm))

    plt.show()


def predict_spirometry_indicators_from_forced_breathing_indicators(eit_indicators, chest_circumference, weight, height):
    # These coefs are ordered as follows: eit, chest_circumference (cm), weight/height (kg/cm), intercept
    coefs = {'FEV1':        [2.30726726e-04, 3.62066834e-02, 1.21170744e+01, -7.057008495867905],
             'FVC':         [1.71965817e-04, 3.68838560e-02, 1.71532549e+01, -8.036810745184955],
             'PEF':         [2.85954280e-02, 0,              2.50845032e+03, -947.9662312833666],
             'FEV1/FVC':    [98.84049,       0,              0,              -6.481973877152896],
             'FEF2575':     [2.73824682e-04, 4.97148637e-02, 5.74347085,     -6.04889737785458]}
    predicted_indicators = {}
    for indic_key in forced_breathing_indicators:
        predicted_indicators[indic_key] = np.array(eit_indicators[indic_key])*coefs[indic_key][0] \
                                          + chest_circumference*coefs[indic_key][1] \
                                          + np.divide(weight, height)*coefs[indic_key][2] \
                                          + coefs[indic_key][3]
    return predicted_indicators


def custom_train_test_split(df, random_state=42, verbose=False):
    seed(random_state)

    lpgo = LeavePGroupsOut(n_groups=2)
    train_test_two_groups = [(train, test) for train, test in lpgo.split(df, groups=df['participant_id'])]
    i_set = (randint(0, len(train_test_two_groups)) - 11) % len(train_test_two_groups)

    train_phase1_idx, test_phase1_idx = train_test_two_groups[i_set]
    train_phase2_idx, test_phase2_idx = train_test_split(train_phase1_idx, test_size=0.1, random_state=random_state)
    train_idx = train_phase2_idx
    test_idx = np.hstack([test_phase1_idx.reshape(-1), test_phase2_idx.reshape(-1)])
    if verbose:
        print('test/train (leave 2 groups out) ratio      ', test_phase1_idx.shape[0] / train_phase1_idx.shape[0])
        print('test/train (random split of train) ratio   ', test_phase2_idx.shape[0] / train_phase2_idx.shape[0])
        print('test/train (overall) ratio                 ', test_idx.shape[0] / train_idx.shape[0])
    return train_idx, test_idx


def format_pval(p):
    if p < 0.0001:
        ps = 'p < 0.0001'
    elif p < 0.01:
        ps = 'p < 0.01'
    elif p < 0.05:
        ps = 'p < 0.05'
    else:
        ps = 'n.s.'
    return ps


def get_lung_clusters(general_outputs, clustering_threshold, save_temp_path):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'get_lung_clusters.exe')
    function_command = r"{} --general_outputs {} --clustering_threshold {} --save_temp_path {}".format(exe_path, general_outputs, clustering_threshold, save_temp_path)
    os.system(function_command)
    outputs_dict = load_pickle(save_temp_path)
    return outputs_dict


def spatio_temporal_filtering(raw_images, temporal_filtering_method, save_temp_path):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'spatio_temporal_filtering.exe')
    function_command = r"{} --raw_images {} --temporal_filtering_method {} --save_temp_path {}".format(exe_path, raw_images, temporal_filtering_method, save_temp_path)
    os.system(function_command)
    outputs_dict = load_pickle(save_temp_path)
    return outputs_dict


def compute_global_waveform_and_general_functional_maps(filtered_images, amplitude_percentile=100, paradigm=None, save_temp_path=None):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'compute_general_outputs.exe')
    function_command = r"{} --filtered_images {} --amplitude_percentile {} --paradigm {} --save_temp_path {}".format(exe_path, filtered_images, amplitude_percentile, paradigm, save_temp_path)
    os.system(function_command)
    outputs_dict = load_pickle(save_temp_path)
    return outputs_dict


def compute_global_forced_breathing_indicators(general_outputs_dict, save_temp_path=None):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'compute_gfbi.exe')
    function_command = r"{} --general_outputs_dict {} --save_temp_path {}".format(exe_path, general_outputs_dict, save_temp_path)
    os.system(function_command)
    outputs_dict = load_pickle(save_temp_path)
    return outputs_dict


def compute_forced_breathing_functional_maps(filtered_images, global_forced_breathing_indicators, general_outputs_dict, save_temp_path=None):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'compute_fbfm.exe')
    function_command = r"{} --filtered_images {} --global_forced_breathing_indicators {} --general_outputs_dict {} --save_temp_path {}".format(exe_path, filtered_images, global_forced_breathing_indicators, general_outputs_dict, save_temp_path)
    os.system(function_command)
    outputs_dict = load_pickle(save_temp_path)
    return outputs_dict


def compute_regional_indicators(filtered_images, general_outputs, clusters_and_rois, save_temp_path=None):
    exe_path = os.path.join(os.path.dirname(os.getcwd()), 'exe', 'compute_regional_indicators.exe')
    function_command = r"{} --filtered_images {} --general_outputs {} --clusters_and_rois {} --save_temp_path {}".format(exe_path, filtered_images, general_outputs, clusters_and_rois, save_temp_path)
    os.system(function_command)
    general_regional_indicators = load_pickle(save_temp_path)
    return general_regional_indicators


if __name__ == '__main__':
    print()