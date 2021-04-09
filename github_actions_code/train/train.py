import os
import argparse
import itertools
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from azureml.core import Dataset, Run
run = Run.get_context()

# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('Y', axis=1).values
    y = df['Y'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)
    data = {"train": {"X": X_train, "y": y_train},
            "test": {"X": X_test, "y": y_test}}
    return data


# Train the model, return the model
def train_model(data, ridge_args):
    reg_model = Ridge(**ridge_args)
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    mse = mean_squared_error(preds, data["test"]["y"])
    metrics = {"mse": mse}
    return metrics



def main(args):
    # create the outputs folder
    os.makedirs('outputs', exist_ok=True)
    
    # Log arguments
    run.log('Model Name', np.str(args.model_name))
    run.log('Version', np.float(args.dataset_version))
    run.log('DataFile Path', np.str(args.data_file_path))
    run.log('DataSet Name', np.str(args.dataset_name))

    model_name = args.model_name
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name
    
    run = Run.get_context()
    
    if (dataset_name):
        if (data_file_path == 'none'):
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  
            columns_to_ignore = ['sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time', 'authors', 'journal', 'mag_id',
                     'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 'url', 's2_id' ]
            dataset = dataset.drop_columns(columns_to_ignore) 
        else:
            dataset = register_dataset(run.experiment.workspace,
                                       dataset_name,
                                       os.environ.get("DATASTORE_NAME"),
                                       data_file_path)
    else:
        e = ("No dataset provided")
        print(e)
        raise Exception(e)
        
    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets['training_data'] = dataset
    run.parent.tag("dataset_id", value=dataset.id)

    # Split the data into test/train
    df = dataset.to_pandas_dataframe()
    data = split_data(df)

    train_args = {"alpha": 0.5}
    # Train the model
    model = train_model(data, train_args)

    # Evaluate and log the metrics returned from the train function
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        run.log(k, v)
        run.parent.log(k, v)
   
  
    
    # files saved in the "outputs" folder are automatically uploaded into run history
    model_file_name = "COVID19Articles_model.pkl"
    joblib.dump(model, os.path.join('outputs', model_file_name))
    run.tag("run_type", value="train")
    print(f"tags now present for run: {run.tags}")

    run.complete()


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the Model",
        default="diabetes_model.pkl",
    )

    parser.add_argument(
        "--step_output",
        type=str,
        help=("output for passing data to next step")
    )

    parser.add_argument(
        "--dataset_version",
        type=str,
        help=("dataset version")
    )

    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified,\
               a new version of the dataset will be registered")
    )

    parser.add_argument(
        "--caller_run_id",
        type=str,
        help=("caller run id, for example ADF pipeline run id")
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help=("Dataset name. Dataset must be passed by name\
              to always get the desired dataset version\
              rather than the one used while the pipeline creation")
    )

        args = parser.parse_args()

    print("Argument [model_name]: %s" % args.model_name)
    print("Argument [step_output]: %s" % args.step_output)
    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)

    model_name = args.model_name
    step_output_path = args.step_output
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args=args)
