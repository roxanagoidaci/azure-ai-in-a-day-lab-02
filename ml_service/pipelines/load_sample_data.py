
import pandas as pd


# Loads the COVID articles sample data from dataset COVID19Articles_Test.
def create_sample_data_csv(file_name: str = "metadata_clusters.csv",
                           for_scoring: bool = False):
    sample_data = Dataset.get_by_name(ws, 'COVID19Articles_Test')
    columns_to_ignore = ['sha', 'source_x', 'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract', 'publish_time', 'authors', 'journal', 'mag_id',
                     'who_covidence_id', 'arxiv_id', 'pdf_json_files', 'pmc_json_files', 'url', 's2_id' ]
    sample_data = sample_data.drop_columns(columns_to_ignore) 
    
    df = sample_data.to_pandas_dataframe()
    
    if for_scoring:
        df = df.drop(columns=['cluster'])

    df.to_csv(file_name, index=False)
