
from azureml.core import Workspace, Dataset

# Loads the COVID articles sample data from dataset COVID19Articles_Test.
def create_sample_data_csv(file_name: str = "COVID19Articles.csv",
                           for_scoring: bool = False):
  
    ws = Workspace.from_config()
    sample_data = Dataset.get_by_name(ws, 'COVID19Articles_Test')
    df = sample_data.to_pandas_dataframe()
    if for_scoring:
        df = df.drop(columns=['cluster'])
    df.to_csv(file_name, index=False)
