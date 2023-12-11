import pandas as pd
import os
#%%
data_path = '/Users/jk1/temp/treatment_effects/preprocessing/gsu_Extraction_20220815_prepro_25112023_213851/splits/train_features_split_0.csv'
continuous_outcomes_path = '/Users/jk1/temp/treatment_effects/preprocessing/gsu_Extraction_20220815_prepro_25112023_213851/splits/train_continuous_outcomes_split_0.csv'
outcomes_path = '/Users/jk1/temp/treatment_effects/preprocessing/gsu_Extraction_20220815_prepro_25112023_213851/splits/train_final_outcomes_split_0.csv'

#%%
features_df = pd.read_csv(data_path)
continuous_outcomes_df = pd.read_csv(continuous_outcomes_path)
outcomes_df = pd.read_csv(outcomes_path)
#%%
# Features data
features_df.drop(columns=['impute_missing_as'], inplace=True)

pivoted_features_df = features_df.pivot(index=['case_admission_id', 'relative_sample_date_hourly_cat'],
                                        columns='sample_label', values='value')

# get rid of multiindex
pivoted_features_df = pivoted_features_df.rename_axis(None, axis=1).reset_index()

# seperate out treatment features
treatment_df = pivoted_features_df[
    ['case_admission_id', 'relative_sample_date_hourly_cat', 'anti_hypertensive_strategy']]
pivoted_features_df.drop(columns=['anti_hypertensive_strategy'], inplace=True)


n_treatment_options = len(treatment_df.anti_hypertensive_strategy.unique())
n_treatment_options
#%%
# custom_data_path = '/Users/jk1/temp/ope_tests/custom_data_out/output-2023-12-10_12-32-12_worker-0_0.json'
# custom_data_path = '/Users/jk1/temp/ope_tests/custom_data_out/output-2023-12-10_17-06-53_worker-0_0.json'
custom_data_path = '/Users/jk1/temp/ope_tests/custom_data_out/output-2023-12-10_21-01-48_worker-0_0.json'

#%%
from ray.rllib.algorithms.cql import CQLConfig
from ray.rllib.algorithms.crr import CRRConfig
from gymnasium.spaces import Box, Discrete, Dict
import numpy as np


# Training on offline data
# Algorithms:
# Critic Regularized Regression (CRR)
# Conservative Q-Learning (CQL)

# CRR
config = CRRConfig()


# CQL
# config = CQLConfig().training(gamma=0.9, lr=0.01)
# config = config.resources(num_gpus=0)
# config = config.rollouts(num_rollout_workers=1)
# print(config.to_dict())


config = config.offline_data(
    input_ = custom_data_path
)

# import gymnasium as gym
# env = gym.make("CartPole-v1")

n_features = len(pivoted_features_df.columns) - 2
config = config.environment(
    observation_space = Dict({
	    'obs': Box(low=pivoted_features_df[pivoted_features_df.columns[2:]].min().min(), high=pivoted_features_df[pivoted_features_df.columns[2:]].max().max(), shape=(n_features,), dtype=np.float32)
	    # 'obs': env.observation_space,
    }),
    # action_space = Box(low=0, high=n_treatment_options-1, shape=(1,), dtype=np.int32)
    # action_space = Box(low=0, high=1, shape=(1,), dtype=np.int32)
    # action_space = env.action_space
    action_space = Discrete(n_treatment_options)
)
#%%
algo = config.build()
#%%
for _ in range(2):
    algo.train()

algo.save(os.path.join(os.path.dirname(custom_data_path), 'crr_model'))
