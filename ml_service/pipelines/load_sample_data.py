
import pandas as pd
from sklearn.datasets import load_diabetes


# Loads the COVID articles sample data from sklearn and produces a csv file that can
# be used by the build/train pipeline script.
def create_sample_data_csv(file_name: str = "metadata_clusters.csv",
                           for_scoring: bool = False):
    sample_data = load_COVID19articles()
    df = pd.DataFrame(
        data=sample_data.data,
        columns=sample_data.feature_names)
    if not for_scoring:
        df['cluster'] = sample_data.target
    # Hard code to diabetes so we fail fast if the project has been
    # bootstrapped.
    df.to_csv(file_name, index=False)
