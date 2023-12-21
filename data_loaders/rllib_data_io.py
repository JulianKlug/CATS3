from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.offline import JsonWriter
from gym.spaces import Box, Discrete, Dict
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from data_loaders.rllib_SampleBatchBuilder import SampleBatchBuilder
from utils import ensure_dir


def rllib_gsu_dataset_creation(gsu_features_path: str, gsu_final_outcomes_path: str,
                               output_path: str = '',
                               target_outcome:str = '3M mRS 0-2',
                               target_treatment:str = 'anti_hypertensive_strategy',
                               truncate_at_penultimate_timestep:bool = True,
                               save_index_columns:bool = False,
                               verbose:bool = True
                            ) -> None:
    """
    Creates an RLLib-compatible dataset from the given GSU features and outcomes data.

    Args:
        gsu_features_path: The file path to the GSU features data.
        gsu_final_outcomes_path: The file path to the GSU final outcomes data.
        output_path: The output path to save the RLLib dataset. If not provided, a default path will be used.
        target_outcome: The target outcome variable to use for rewards.
        target_treatment: The target treatment variable to use for actions.
        truncate_at_penultimate_timestep: Whether to truncate the data at the penultimate timestep. This is necessary
                                            because the last timestep does not have a next timestep
        save_index_columns: Whether to save the column indices in the dataset.
        verbose: Whether to display progress information.

    Returns:
        None

    Raises:
        None
    """

    if output_path == '':
        output_path = os.path.join(os.path.dirname(gsu_features_path), f'{os.path.basename(gsu_features_path).split(".")[0]}_rllib_dataset')
    ensure_dir(output_path)

    # Load data
    features_df = pd.read_csv(gsu_features_path)
    outcomes_df = pd.read_csv(gsu_final_outcomes_path)

    # Preprocess data
    features_df.drop(columns=['impute_missing_as'], inplace=True)
    pivoted_features_df = features_df.pivot(index=['case_admission_id', 'relative_sample_date_hourly_cat'],
                                            columns='sample_label', values='value')
    # get rid of multiindex
    pivoted_features_df = pivoted_features_df.rename_axis(None, axis=1).reset_index()
    # separate out treatment features
    treatment_df = pivoted_features_df[
        ['case_admission_id', 'relative_sample_date_hourly_cat', target_treatment]]
    pivoted_features_df.drop(columns=[target_treatment], inplace=True)

    all_cids = pivoted_features_df.case_admission_id.unique()

    # Initialize RLLib objects
    batch_builder = SampleBatchBuilder()
    writer = JsonWriter(output_path)

    if save_index_columns:
        # saving indices (case_admission_id, relative_sample_date_hourly_cat)
        # this needs a definition of a vaster observation space
        # Needed for inference with pre-defined treatment policies (as cid and ts are needed to get the correct treatment)
        # As preprocesor does not accept Text spaces, we need to convert them to two integers
        cid0 = pivoted_features_df[pivoted_features_df.columns[0]].apply(lambda x: x.split('_')[0]).astype(int)
        cid1 = pivoted_features_df[pivoted_features_df.columns[0]].apply(lambda x: x.split('_')[1]).astype(int)
        features_with_index_columns_df = pd.concat([cid0, cid1, pivoted_features_df[pivoted_features_df.columns[1:]]],
                                                   axis=1).astype(np.float64)
        features_with_index_columns_df.columns = ['cid0', 'cid1'] + list(pivoted_features_df.columns[1:])
        pivoted_features_df = features_with_index_columns_df

        column_start_index = 0
    else:
        column_start_index = 2

    n_episodes = features_df.case_admission_id.nunique()
    n_features = len(pivoted_features_df.columns) - column_start_index

    # Initialize preprocessor (needed for compatibility with RLLib algorithms)
    obs_space = Box(low=pivoted_features_df[pivoted_features_df.columns[column_start_index:]].min().min(),
                    high=pivoted_features_df[pivoted_features_df.columns[column_start_index:]].max().max(), shape=(n_features,),
                    dtype=np.float64)
    prep = get_preprocessor(obs_space)(obs_space)

    if verbose:
        print("The preprocessor is", prep)

    # Iterate over episodes and timesteps
    for eps_id in tqdm(range(n_episodes)):
        cid = all_cids[eps_id]
        if not save_index_columns:
            cid_data_df = pivoted_features_df[pivoted_features_df.case_admission_id == cid]
        else:
            cid_data_df = pivoted_features_df[
                (pivoted_features_df.cid0.astype(int) == int(cid.split('_')[0]))
                & (pivoted_features_df.cid1.astype(int) == int(cid.split('_')[1]))]

        # Truncating at penultimate timestep to avoid missing data for last timestep (new obs must be defined for ts + 1)
        if truncate_at_penultimate_timestep:
            last_timestep = cid_data_df.relative_sample_date_hourly_cat.max() - 1
        else:
            last_timestep = cid_data_df.relative_sample_date_hourly_cat.max()

        for ts in range(int(last_timestep + 1)):
            obs = cid_data_df[cid_data_df.relative_sample_date_hourly_cat == ts]
            obs = obs[obs.columns[column_start_index:]].values[0]
            obs = prep.transform(obs)

            new_obs = cid_data_df[cid_data_df.relative_sample_date_hourly_cat == ts + 1]
            if not truncate_at_penultimate_timestep and ts == last_timestep:
                # if not truncating at penultimate timestep, then the last timestep does not have a next timestep
                # so we just use the same obs as the new_obs
                new_obs = obs
            else:
                new_obs = new_obs[new_obs.columns[column_start_index:]].values[0]
            new_obs = prep.transform(new_obs)

            action = int(treatment_df[(treatment_df.case_admission_id == cid) & (
                        treatment_df.relative_sample_date_hourly_cat == ts)][target_treatment].values[0])

            if ts == 0:
                prev_action = action
            else:
                prev_action = int(treatment_df[(treatment_df.case_admission_id == cid) & (
                            treatment_df.relative_sample_date_hourly_cat == ts - 1)][
                                      target_treatment].values[0])

            if ts == last_timestep:
                terminated = True
                reward = outcomes_df[outcomes_df.case_admission_id == cid][target_outcome].values[0]
            else:
                terminated = False
                reward = 0

            truncated = False
            prev_reward = 0
            info = {}

            if verbose:
                print(f'cid: {cid}, ts: {ts}, action: {action}, terminated: {terminated}, reward: {reward}')
                print(f'prev_action: {prev_action}')
                print('---')

            batch_builder.add_values(
                t=ts,
                eps_id=eps_id,
                agent_index=0,
                obs=obs,
                actions=action,
                action_prob=1.0,  # put the true action probability here
                action_logp=0.0,
                rewards=reward,
                prev_actions=prev_action,
                prev_rewards=prev_reward,
                terminateds=terminated,
                truncateds=truncated,
                infos=info,
                new_obs=new_obs,
            )

        writer.write(batch_builder.build_and_reset())

    if verbose:
        print(f'Dataset saved to {output_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("GSU RLLib Dataset Creation")
    parser.add_argument('-f', '--gsu_features_path', type=str, required=True, help='Path to the GSU features data.')
    parser.add_argument('-o', '--gsu_final_outcomes_path', type=str, required=True, help='Path to the GSU final outcomes data.')
    parser.add_argument('-p', '--output_path', type=str, default='', help='Path to save the RLLib dataset.')
    parser.add_argument('-t', '--target_outcome', type=str, default='3M mRS 0-2', help='Target outcome variable to use for rewards.')
    parser.add_argument('-r', '--target_treatment', type=str, default='anti_hypertensive_strategy', help='Target treatment variable to use for actions.')
    parser.add_argument('-u', '--do_not_truncate_at_penultimate_timestep', action='store_false', help='Whether to truncate the data at the penultimate timestep. This is necessary because the last timestep does not have a next timestep.')
    parser.add_argument('-s', '--save_index_columns', default=False, action='store_true', help='Whether to save the column indices in the dataset.')
    parser.add_argument('-v', '--verbose', type=bool, default=True, help='Whether to display progress information.')
    args = parser.parse_args()

    rllib_gsu_dataset_creation(gsu_features_path=args.gsu_features_path,
                               gsu_final_outcomes_path=args.gsu_final_outcomes_path,
                               output_path=args.output_path,
                               target_outcome=args.target_outcome,
                               target_treatment=args.target_treatment,
                               truncate_at_penultimate_timestep=args.do_not_truncate_at_penultimate_timestep,
                               save_index_columns=args.save_index_columns,
                               verbose=args.verbose)
