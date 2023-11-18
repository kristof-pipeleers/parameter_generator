import pandas as pd
import json
from ast import literal_eval 

def transform_to_csv(output_file_path):

    # Read JSON data from a file
    with open(output_file_path, "r") as json_file:
        json_data = json.load(json_file)


    # Create a DataFrame
    df = pd.DataFrame(json_data)

    # Create a dictionary to store the data
    data_dict = {}

    # Iterate through the DataFrame and populate the dictionary
    for index, row in df.iterrows():
        company_name = row['company_name']
        category = json.loads(row['answer']['arguments'])  # Use json.loads to parse JSON
        
        for key, value in category.items():
            if key not in data_dict:
                data_dict[key] = {}

            if company_name not in data_dict[key]:
                data_dict[key][company_name] = []

            data_dict[key][company_name].append(value)

    # Create a new DataFrame from the dictionary
    result_df = pd.DataFrame(data_dict)

    # Transpose the DataFrame to have the desired format
    result_df = result_df.T

    # Save the DataFrame to a CSV file
    result_df.to_csv('output.csv', quoting=1, index_label='parameters')