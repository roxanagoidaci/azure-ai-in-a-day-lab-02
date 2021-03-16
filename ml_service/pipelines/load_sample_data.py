from azureml.core import Workspace, Dataset


# Loads the COVID articles sample data from dataset COVID19Articles_Test.
def create_sample_data_csv(file_name: str = "COVID19Articles.csv",
                           for_scoring: bool = False):

    sample_data = Dataset.File.from_files('https://solliancepublicdata.blob.core.windows.net/ai-in-a-day/lab-02/COVID19Articles.csv')
    df = sample_data.to_pandas_dataframe()
    if for_scoring:
        df = df.drop(columns=['cluster'])
    df.to_csv(file_name, index=False)
