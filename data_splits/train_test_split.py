import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from data_splits.utils import link_patient_id_to_outcome


def generate_splits(features_path:str, continuous_outcome_path:str, final_outcome_path:str, outcome:str,
                    output_dir: str = '',
                    test_size: float = 0.2, seed=42, n_splits=5,
                    verbose: bool = True
                    ) -> None:
    if output_dir == '':
        output_dir = os.path.join(os.path.dirname(features_path), 'splits')
        os.makedirs(output_dir, exist_ok=True)

    # save all arguments given to function as json
    args = locals()
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    data = pd.read_csv(features_path)
    continuous_outcomes = pd.read_csv(continuous_outcome_path)
    final_outcomes = pd.read_csv(final_outcome_path)

    final_outcomes['patient_id'] = final_outcomes['case_admission_id'].apply(lambda x: x.split('_')[0])
    data['patient_id'] = data['case_admission_id'].apply(lambda x: x.split('_')[0])
    continuous_outcomes['patient_id'] = continuous_outcomes['case_admission_id'].apply(lambda x: x.split('_')[0])

    """
    SPLITTING DATA
    Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
    would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
    """
    # Reduce every patient to a single outcome (to avoid duplicates)
    selected_final_outcome_df = final_outcomes[['case_admission_id', 'patient_id', outcome]]
    selected_final_outcome_df = selected_final_outcome_df.melt(id_vars=['case_admission_id', 'patient_id'], var_name='outcome_label', value_name='outcome')
    all_pids_with_outcome = link_patient_id_to_outcome(selected_final_outcome_df, outcome)
    all_pids_with_outcome.dropna(subset=['outcome'], inplace=True)

    pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                    all_pids_with_outcome.outcome.tolist(),
                                                                    stratify=all_pids_with_outcome.outcome.tolist(),
                                                                    test_size=test_size,
                                                                    random_state=seed)

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.csv'), index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.csv'), index=False)

    # define K fold
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
        fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
        fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]
        split_df =  pd.DataFrame([fold_train_pidx, fold_val_pidx]).T
        split_df.columns= ['train', 'val']
        split_df.to_csv(
            os.path.join(output_dir, 'pid_cv_split_{}.csv'.format(len(splits))), index=False)
        splits.append({'train': fold_train_pidx, 'val': fold_val_pidx})

    # Split csv files and save results
    test_features_df = data[data.patient_id.isin(pid_test)]
    test_continuous_outcomes_df = continuous_outcomes[continuous_outcomes.patient_id.isin(pid_test)]
    test_final_outcomes_df = final_outcomes[final_outcomes.patient_id.isin(pid_test)]
    test_features_df.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
    test_continuous_outcomes_df.to_csv(os.path.join(output_dir, 'test_continuous_outcomes.csv'), index=False)
    test_final_outcomes_df.to_csv(os.path.join(output_dir, 'test_final_outcomes.csv'), index=False)

    for i, split in enumerate(splits):
        train_features_df = data[data.patient_id.isin(split['train'])]
        train_continuous_outcomes_df = continuous_outcomes[continuous_outcomes.patient_id.isin(split['train'])]
        train_final_outcomes_df = final_outcomes[final_outcomes.patient_id.isin(split['train'])]
        val_features_df = data[data.patient_id.isin(split['val'])]
        val_continuous_outcomes_df = continuous_outcomes[continuous_outcomes.patient_id.isin(split['val'])]
        val_final_outcomes_df = final_outcomes[final_outcomes.patient_id.isin(split['val'])]
        train_features_df.to_csv(os.path.join(output_dir, 'train_features_split_{}.csv'.format(i)), index=False)
        train_continuous_outcomes_df.to_csv(os.path.join(output_dir, 'train_continuous_outcomes_split_{}.csv'.format(i)), index=False)
        train_final_outcomes_df.to_csv(os.path.join(output_dir, 'train_final_outcomes_split_{}.csv'.format(i)), index=False)
        val_features_df.to_csv(os.path.join(output_dir, 'val_features_split_{}.csv'.format(i)), index=False)
        val_continuous_outcomes_df.to_csv(os.path.join(output_dir, 'val_continuous_outcomes_split_{}.csv'.format(i)), index=False)
        val_final_outcomes_df.to_csv(os.path.join(output_dir, 'val_final_outcomes_split_{}.csv'.format(i)), index=False)

    if verbose:
        # print distribution of outcome for every split and test
        print('Distribution of outcome in test set: {}'.format(
            test_final_outcomes_df[outcome].value_counts(normalize=True)))
        for i, split in enumerate(splits):
            print('Distribution of outcome in train split {}: {}'.format(i, final_outcomes[final_outcomes.patient_id.isin(split['train'])][outcome].value_counts(normalize=True)))
            print('Distribution of outcome in val split {}: {}'.format(i, final_outcomes[final_outcomes.patient_id.isin(split['val'])][outcome].value_counts(normalize=True)))
            print('\n')
            print('Distribution treatment strategy in train split {}: {}'.format(i, data[(data.patient_id.isin(split['train']) & (data.sample_label == 'anti_hypertensive_strategy'))]['value'].value_counts(normalize=True)))
            print('Distribution treatment strategy in val split {}: {}'.format(i, data[(data.patient_id.isin(split['val']) & (data.sample_label == 'anti_hypertensive_strategy'))]['value'].value_counts(normalize=True)))
            print('\n')
            print('\n')

    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features_path', type=str, required=True)
    parser.add_argument('-co', '--continuous_outcome_path', type=str, required=True)
    parser.add_argument('-fo', '--final_outcome_path', type=str, required=True)
    parser.add_argument('-o', '--outcome', type=str, required=True)
    parser.add_argument('-od', '--output_dir', type=str, required=False, default='')
    parser.add_argument('-t', '--test_size', type=float, required=False, default=0.2)
    parser.add_argument('-s', '--seed', type=int, required=False, default=42)
    parser.add_argument('-n', '--n_splits', type=int, required=False, default=5)
    parser.add_argument('-v', '--verbose', type=bool, required=False, default=True)
    args = parser.parse_args()

    generate_splits(features_path=args.features_path,
                    continuous_outcome_path=args.continuous_outcome_path,
                    final_outcome_path=args.final_outcome_path,
                    outcome=args.outcome,
                    output_dir=args.output_dir,
                    test_size=args.test_size,
                    seed=args.seed,
                    n_splits=args.n_splits,
                    verbose=args.verbose)
