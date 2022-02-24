from functools import partial
from do_inference import Detector, LightningClassifierInferer, do_inference
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
from sklearn import metrics
import sys
import numpy as np
import matplotlib.pyplot as plt


def partial_roc_auc_score(y_true, y_pred):
    min_spec = 0.9
    try:
        return metrics.roc_auc_score(y_true, y_pred, max_fpr=(1 - min_spec))
    except:
        return 0.5


def roc_auc_score(y_trues, y_preds):
    try:
        return metrics.roc_auc_score(y_trues, y_preds)
    except:
        return 0.5


def screening_sens_at_spec(y_true, y_pred):
    try:
        eps = sys.float_info.epsilon
        at_spec = 0.95
        fpr, tpr, threshes = metrics.roc_curve(
            y_true, y_pred, drop_intermediate=False)
        spec = 1 - fpr

        operating_points_with_good_spec = spec >= (at_spec - eps)
        max_tpr = tpr[operating_points_with_good_spec][-1]

        operating_point = np.argwhere(
            operating_points_with_good_spec).squeeze()[-1]
        operating_tpr = tpr[operating_point]

        assert max_tpr == operating_tpr or (np.isnan(max_tpr) and np.isnan(
            operating_tpr)), f'{max_tpr} != {operating_tpr}'
        assert max_tpr == max(tpr[operating_points_with_good_spec]) or (np.isnan(max_tpr) and max(tpr[operating_points_with_good_spec])), \
            f'{max_tpr} == {max(tpr[operating_points_with_good_spec])}'

        return max_tpr
    except:
        return 0


path_to_data = '/home/firas/Desktop/work/fundus/data/AIROGS/unzipped/'
path_to_eval_folder = '/home/firas/Desktop/work/fundus/evaluation_outputs/submission/version_1_1/'


def create_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    detector = Detector()
    classifier = LightningClassifierInferer()
    labels = pd.read_csv(os.path.join(path_to_data, 'train_labels.csv'), index_col='challenge_id')

    create_dir(path_to_eval_folder)
    results = {'name': [], 'label': [], 'rg_likelihood': [], 'rg_binary': [], 'ungradability_score': [], 'ungradability_binary': [], 'rg_variance': []}
    for img in tqdm([file for file in os.listdir(path_to_data) if file.endswith('.jpg')][:2000]):
        sample_img = np.asarray(Image.open(os.path.join(path_to_data, img)))
        label = labels.loc[img.split('.')[0]]['class']
        label = 1 if label == 'RG' else 0
        rg_likelihood, rg_binary, ungradability_score, ungradability_binary, rg_variance = do_inference(
            input_image_array=sample_img,
            Detector=detector,
            Classifier=classifier,
            )
        results['name'].append(img)
        results['label'].append(label)
        results['rg_likelihood'].append(rg_likelihood)
        results['rg_binary'].append(rg_binary)
        results['ungradability_score'].append(ungradability_score)
        results['ungradability_binary'].append(ungradability_binary)
        results['rg_variance'].append(rg_variance)
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(path_to_eval_folder, 'results.csv'), index=False)
    total_auc = roc_auc_score(df_results['label'].to_list(), df_results['rg_likelihood'].to_list())
    partial_auc = partial_roc_auc_score(df_results['label'].to_list(), df_results['rg_likelihood'].to_list())
    sens_at_spec = screening_sens_at_spec(df_results['label'].to_list(), df_results['rg_likelihood'].to_list())
    
    # Get Youden's J index
    fpr, tpr, thresholds = metrics.roc_curve(df_results['label'].to_list(), df_results['rg_likelihood'].to_list())
    optimal_idx = np.argmax(tpr - fpr)  
    optimal_threshold = thresholds[optimal_idx]

    with open(os.path.join(path_to_eval_folder, 'metric_results.txt'), 'w+') as f:
        f.write(f'Total AUC: {total_auc}')
        f.write('\n')
        f.write(f'Partial AUC: {partial_auc}')
        f.write('\n')
        f.write(f'Sensitivity at Specificity: {sens_at_spec}')
        f.write('\n')
        f.write(f'Optimal Threshold: {optimal_threshold}')
        f.write('\n')

    print('Total AUC: ', total_auc)
    print('Partial AUC: ', partial_auc)
    print('Sensitivity at Specificity: ', sens_at_spec)
    print('Optimal Threshold: ', optimal_threshold)

    occurence_frequency = []
    for lower_bound, upper_bound in tqdm(zip(list(np.arange(0, 1.0, 0.1)), list(np.arange(0.1, 1.1, 0.1)))):
        bounded_results = df_results[df_results['ungradability_score'].between(lower_bound, upper_bound)]
        occurence_frequency.append(len(bounded_results))
        bounded_results_head = bounded_results.head(100)
        path = os.path.join(path_to_eval_folder, f'{round(lower_bound, 2)}-{round(upper_bound, 2)}')
        os.mkdir(path)
        for img_name, ungradability_score in zip(bounded_results_head['name'], bounded_results_head['ungradability_score']):
            img = Image.open(os.path.join(path_to_data, img_name))
            img.save(os.path.join(path, str(ungradability_score) + img_name))
    range_ = list(np.arange(0.1, 1.1, 0.1))
    plt.bar(range_, occurence_frequency, width=0.05)
    plt.savefig(os.path.join(path_to_eval_folder, 'occurence_histogram.png'))

