import json
import os
from datetime import datetime

from tempor import plugin_loader
from tempor.utils.serialization import save_to_file

from data_loaders.temporAI_dataloader import load_gsu_dataset
from utils import ensure_dir


def train_crn(splits_folder_path: str, epochs: int, output_dir: str, verbose: bool = False):
    """
    Train a Counterfactual Regression Network (CRN) model.

    Args:
        splits_folder_path (str): The path to the folder containing the train splits.
        epochs (int): The number of training epochs.
        output_dir (str): The directory to save the trained model.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        None

    Examples:
        train_crn('/path/to/splits', 100, '/path/to/output')
    """
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, f'crn_{timestamp}')
    ensure_dir(output_dir)

    params = {
        'epochs': epochs,
        'encoder_rnn_type': 'LSTM',
        'encoder_hidden_size': 100,
        'encoder_num_layers': 1,
        'encoder_bias': True,
        'encoder_dropout': 0.1,
        'encoder_bidirectional': False,
        'encoder_nonlinearity': None,
        'encoder_proj_size': None,

        'decoder_rnn_type': 'LSTM',
        'decoder_hidden_size': 100,
        'decoder_num_layers': 1,
        'decoder_bias': True,
        'decoder_dropout': 0.1,
        'decoder_bidirectional': False,
        'decoder_nonlinearity': None,
        'decoder_proj_size': None,

        'adapter_hidden_dims': [50],
        'adapter_out_activation': 'Tanh',
        'predictor_hidden_dims': [],
        'predictor_out_activation': None,
        'max_len': None,
        'optimizer_str': 'Adam',
        'optimizer_kwargs': {'lr': 0.001, 'weight_decay': 1e-05},
        'batch_size': 32,
        'padding_indicator': -999.0
    }

    # save all arguments given to function as json
    args = locals()
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args, f)

    split = 0
    data_path = os.path.join(splits_folder_path, 'train_features_split_{}.csv'.format(split))
    continuous_outcomes_path = os.path.join(splits_folder_path, 'train_continuous_outcomes_split_{}.csv'.format(split))
    gsu_dataset = load_gsu_dataset(data_path, continuous_outcomes_path)

    # TODO: remove this line
    gsu_dataset = gsu_dataset[:2]

    model = plugin_loader.get("treatments.temporal.regression.crn_regressor", epochs=params['epochs'],
                              encoder_rnn_type=params['encoder_rnn_type'],
                              encoder_hidden_size=params['encoder_hidden_size'],
                              encoder_num_layers=params['encoder_num_layers'], encoder_bias=params['encoder_bias'],
                              encoder_dropout=params['encoder_dropout'],
                              encoder_bidirectional=params['encoder_bidirectional'],
                              encoder_nonlinearity=params['encoder_nonlinearity'],
                              encoder_proj_size=params['encoder_proj_size'],
                              decoder_rnn_type=params['decoder_rnn_type'],
                              decoder_hidden_size=params['decoder_hidden_size'],
                              decoder_num_layers=params['decoder_num_layers'], decoder_bias=params['decoder_bias'],
                              decoder_dropout=params['decoder_dropout'],
                              decoder_bidirectional=params['decoder_bidirectional'],
                              decoder_nonlinearity=params['decoder_nonlinearity'],
                              decoder_proj_size=params['decoder_proj_size'],
                              adapter_hidden_dims=params['adapter_hidden_dims'],
                              adapter_out_activation=params['adapter_out_activation'],
                              predictor_hidden_dims=params['predictor_hidden_dims'],
                              predictor_out_activation=params['predictor_out_activation'], max_len=params['max_len'],
                              optimizer_str=params['optimizer_str'], optimizer_kwargs=params['optimizer_kwargs'],
                              batch_size=params['batch_size'], padding_indicator=params['padding_indicator']
                              )

    # Train the model:
    model.fit(gsu_dataset)

    save_to_file(os.path.join(output_dir, f'crn_model_{timestamp}_split_{split}.cpkl'), model)

    return None


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--splits_folder_path', type=str, required=True)
    parser.add_argument('-e', '--epochs', type=int, required=True)
    parser.add_argument('-o', '--output_dir', type=str, required=True)
    parser.add_argument('-v', '--verbose', type=bool, required=False, default=True)
    args = parser.parse_args()

    train_crn(splits_folder_path=args.splits_folder_path,
              epochs=args.epochs,
              output_dir=args.output_dir,
              verbose=args.verbose)
