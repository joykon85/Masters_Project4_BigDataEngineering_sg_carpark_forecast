#from kfp.dsl import component, pipeline, Input, Output, Condition, OutputPath
#from kfp.dsl import Artifact, Dataset
from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Output, Condition, OutputPath
from google_cloud_pipeline_components.types import artifact_types
#########PIPELINE LOAD AND CHECK ########################################################################################
#@component(packages_to_install=["google-cloud-bigquery", "pandas", "scikit-learn"])
#def load_check_data(bigquery_dataset_input: str, my_project_id: str, checked_parquet: str, checked_12_parquet: str) -> bool:
#@component(packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes"])
@dsl.component(
    base_image="us-central1-docker.pkg.dev/involuted-forge-456406-a1/pipeline-containers/my-custom-pipeline-image"
)
def load_check_data(bigquery_dataset_input: str, my_project_id: str, checked_parquet: str, checked_12_parquet: str, run_id: str, success: OutputPath(str)):

    from google.cloud import bigquery
    import pandas as pd
    import os
    import numpy as np
    from datetime import datetime, timedelta

    # Initialize BigQuery client
    client = bigquery.Client(project=my_project_id)

    #query = f"SELECT * FROM `{bigquery_dataset_input}`"
    #query = f"SELECT * FROM `{bigquery_dataset_input}` WHERE timestamp > TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL '454' MINUTE)"
    query = f"SELECT * FROM `{bigquery_dataset_input}` WHERE timestamp > DATETIME(TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 454 MINUTE))"
    #query = f"SELECT * FROM `{bigquery_dataset}` WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL '25' MINUTE)"

    # Run query
    query_job = client.query(query)
    df = query_job.to_dataframe()

    #df

    #second query
    query = f"SELECT * FROM `{bigquery_dataset_input}` WHERE timestamp > DATETIME(TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL 419 MINUTE))"

    # Run query
    query_job = client.query(query)
    df_12 = query_job.to_dataframe()

    # DECLARE SEQUENCE LENGTH CONSTANT
    ####################################################
    SEQUENCE_LENGTH = 5
    SEQUENCE_LENGTH_12 = 12

    # FUNCTIONS USED
    ####################################################
    def check_value_counts_equal(series):
        """
        Checks if all values in a Pandas Series' value_counts() are equal.

        Args:
            series: A Pandas Series.

        Returns:
            True if all value counts are equal, False otherwise.
        """
        value_counts = series.value_counts()
        if len(value_counts) <= 1: #if there is only one unique value, or zero, then they are equal.
            return True
        else:
            return (value_counts == value_counts.iloc[0]).all() #compare all to the first value.

    #Sort the data
    df.sort_values(['CarParkID', 'timestamp'], inplace=True)
    df_12.sort_values(['CarParkID', 'timestamp'], inplace=True)

    continue_flag = True

    if (check_value_counts_equal(df['CarParkID']) == False):
        #print("CarParkID is not equal. Don't continue. Trigger an error")
        continue_flag = False
    else:
        #print("CarParkEncoded is equal!")

        if (df['CarParkID'].value_counts().iloc[0] != SEQUENCE_LENGTH):
            #print("Don't have Sequence Length of Data Input. Don't continue. Trigger an error")
            continue_flag = False
    #    else:
            #print("Have right amount of data to proceed. Data Preprocessing Completed Successfully")

    if (check_value_counts_equal(df_12['CarParkID']) == False):
        #print("CarParkID is not equal. Don't continue. Trigger an error")
        continue_flag = False
    else:
        #print("CarParkEncoded is equal!")

        if (df_12['CarParkID'].value_counts().iloc[0] != SEQUENCE_LENGTH_12):
            #print("Don't have Sequence Length of Data Input. Don't continue. Trigger an error")
            continue_flag = False

    # SAVE TO PARQUET FILE
    df.to_parquet(checked_parquet, index=False)
    df_12.to_parquet(checked_12_parquet, index=False)

    # PIPELINE COMPONENT OUTPUT
    with open(success, 'w') as f:
        f.write('true' if continue_flag else 'false')

#########PIPELINE PREPROCESS########################################################################################
#@component(packages_to_install=["google-cloud-bigquery", "pandas", "google-cloud-storage", "pyarrow", "scikit-learn"])
@dsl.component(
    base_image="us-central1-docker.pkg.dev/involuted-forge-456406-a1/pipeline-containers/my-custom-pipeline-image"
)
def preprocess_data(checked_parquet: str,
                    sequence_parquet: str,
                    pkl_file_path_1: str,
                    pkl_file_path_2: str,
                    pkl_file_path_3: str,
                    pkl_file_path_4: str,
                    run_id: str,
                    success: OutputPath(str)):

    import pandas as pd
    import numpy as np
    import os
    import pickle
    from datetime import datetime, timedelta
    from google.cloud import storage

    # DECLARE SEQUENCE LENGTH CONSTANT
    ####################################################
    SEQUENCE_LENGTH = 5

    # FUNCTIONS USED
    ####################################################
    def create_sequences_per_location(df, sequence_length=5):

        # Define the columns to use for the 3D array
        selected_columns = ['CarParkEncoded_MM', 'Min_MM', 'Hour_MM', 'DayOfWeek_MM', 'AvailableLots_MM']
        subset_df = df[selected_columns]

        # Convert the DataFrame subset to a NumPy array
        numpy_array = subset_df.values

        # Calculate the number of 2D arrays
        num_sequences  = numpy_array.shape[0] // sequence_length
        remainder = numpy_array.shape[0] % sequence_length

        if remainder != 0:
            #print("ERROR! Not every Carpark has Sequence Length entries")
            return None, False
        else:
            # Create the list to store the 3D arrays
            sequence = []

            # Reshape the array into 3D chunks
            sequence = numpy_array.reshape(num_sequences, sequence_length, len(selected_columns))

            #print("Sequencing Done! ")
            return sequence, True

    def read_pkl_file(pkl_file_path):
        try:
            # Initialize the Google Cloud Storage client
            client = storage.Client()

            # Parse the GCS URI
            bucket_name = pkl_file_path.split('/')[2]
            blob_name = '/'.join(pkl_file_path.split('/')[3:])  # ERROR HERE: Using pkl_file_path_1

            # Get the bucket and blob (file)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download the blob's content to a bytes object
            blob_data = blob.download_as_bytes()

            #print("Loaded successfully from GCS.")

            # Load the object from the bytes object
            return pickle.loads(blob_data)

        except Exception as e:
            #print(f"Error loading object from GCS: {e}")
            raise RuntimeError(f"Error loading object from GCS: {e}")
            return None

    # READ PARQUET FILE
    #####################################################
    df = pd.read_parquet(checked_parquet)

    # OPEN MODEL SCALARS AND ENCODERS
    #####################################################
    le = read_pkl_file(pkl_file_path_1)
    scaler_AL = read_pkl_file(pkl_file_path_2)
    scalar_CP = read_pkl_file(pkl_file_path_3)
    scaler_temp = read_pkl_file(pkl_file_path_4)


    # DATA PREPROCESSING
    #####################################################
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # FEATURE EXTRACTION/ ENCODING/ NORMALISATION
    # Extract features
    df['Min'] = df['timestamp'].dt.minute
    df['Hour'] = df['timestamp'].dt.hour
    df['DayOfWeek'] = df['timestamp'].dt.dayofweek

    # Encode CarParkID
    df['CarParkEncoded'] = le.transform(df['CarParkID'])

    # Sort for time sequence modeling
    df.sort_values(['CarParkEncoded', 'timestamp'], inplace=True)

    # NORMALISE
    df['AvailableLots_MM'] = scaler_AL.transform(df[['AvailableLots']])
    df['CarParkEncoded_MM'] = scalar_CP.transform(df[['CarParkEncoded']])
    df[['Min_MM', 'Hour_MM', 'DayOfWeek_MM']] = scaler_temp.transform(df[['Min','Hour', 'DayOfWeek']])

    # LSTM SEQUENCING
    ######################################################

    continue_flag = True
    model_input, sequence_done_flag = create_sequences_per_location(df, SEQUENCE_LENGTH)

    # Extra check. If fail don't proceed to predictions. Need to add some code here
    if (sequence_done_flag == False):
        #print("ERROR! Not every Carpark has Sequence Length entries. Raise Error")
        continue_flag = False


    #model_input_flat = model_input.reshape(88, 25)
    model_input_flat = model_input.reshape(model_input.shape[0], -1)

    # Save as DataFrame and export to Parquet
    df_model_input = pd.DataFrame(model_input_flat)

    df_model_input.to_parquet(sequence_parquet, index=False)

    # PIPELINE COMPONENT OUTPUT
    with open(success, 'w') as f:
        f.write('true' if continue_flag else 'false')

#########PIPELINE PREDICTION AND UPDATE TABLE########################################################################################

#@component(packages_to_install=["google-cloud-bigquery", "pandas","google-cloud-storage","tensorflow","db-dtypes", "pyarrow", "scikit-learn"])
@dsl.component(
    base_image="us-central1-docker.pkg.dev/involuted-forge-456406-a1/pipeline-containers/my-custom-pipeline-image"
)
def predict_with_lstm(model_keras_30: str,
                      model_keras_45: str,
                      model_keras_60: str,
                      checked_parquet: str,
                      checked_12_parquet: str,
                      sequence_parquet: str,
                      pkl_file_path_2: str,
                      my_project_id:str,
                      bigquery_dataset_output:str,
                      run_id: str):

    import tensorflow as tf
    import os
    import numpy as np
    import pandas as pd
    import pickle
    from google.cloud import storage
    from datetime import datetime, timedelta
    from google.cloud import bigquery

    df_model_input = pd.read_parquet(sequence_parquet)

    #df_model_input

    model_input = df_model_input.to_numpy().reshape(88, 5, 5)

    # Load the model
    #loaded_LSTM_Model = tf.keras.models.load_model(model_keras)
    LSTM_30 = tf.keras.models.load_model(model_keras_30)
    LSTM_45 = tf.keras.models.load_model(model_keras_45)
    LSTM_60 = tf.keras.models.load_model(model_keras_60)

    #preds = loaded_LSTM_Model.predict(model_input)

    preds_30 = LSTM_30.predict(model_input)
    preds_45 = LSTM_45.predict(model_input)
    preds_60 = LSTM_60.predict(model_input)



    def read_pkl_file(pkl_file_path):
        try:
            # Initialize the Google Cloud Storage client
            client = storage.Client()

            # Parse the GCS URI
            bucket_name = pkl_file_path.split('/')[2]
            blob_name = '/'.join(pkl_file_path.split('/')[3:])  # ERROR HERE: Using pkl_file_path_1

            # Get the bucket and blob (file)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(blob_name)

            # Download the blob's content to a bytes object
            blob_data = blob.download_as_bytes()

            #print("Loaded successfully from GCS.")

            # Load the object from the bytes object
            return pickle.loads(blob_data)

        except Exception as e:
            #print(f"Error loading object from GCS: {e}")
            raise RuntimeError(f"Error loading object from GCS: {e}")
            return None

    scaler_AL = read_pkl_file(pkl_file_path_2)

    preds_t_30 = scaler_AL.inverse_transform(preds_30)
    preds_t_45 = scaler_AL.inverse_transform(preds_45)
    preds_t_60 = scaler_AL.inverse_transform(preds_60)



    # READ PARQUET FILE
    #####################################################
    df = pd.read_parquet(checked_parquet)

    df['timestamp'] = df['timestamp'].dt.floor('min')
    max_timestamp = df['timestamp'].max()
    df_lasttime = df[df['timestamp'] == max_timestamp]

    df_pred = df_lasttime.drop(['Area', 'Development', 'LotType', 'Agency'], axis=1)
    df_pred = df_pred.rename(columns={'timestamp': 'LatestTimestamp'})

    df_pred['PredictedLots_30min'] = np.round(preds_t_30).astype(int)
    df_pred['PredictedLots_45min'] = np.round(preds_t_45).astype(int)
    df_pred['PredictedLots_1hour'] = np.round(preds_t_60).astype(int)

    df_12 = pd.read_parquet(checked_12_parquet)

    df_12['timestamp'] = df_12['timestamp'].dt.floor('min')

    pivot_df = df_12.pivot(index='CarParkID', columns='timestamp', values='AvailableLots')

    # 4. Sort columns (timestamps) from earliest to latest
    pivot_df = pivot_df.sort_index(axis=1)

    # 5. Create new column names
    n_steps = pivot_df.shape[1]  # number of timestamps

    new_cols = [f"t-{5*(n_steps-i-1)}" if i != n_steps-1 else 't' for i in range(n_steps)]

    # 6. Rename the columns
    pivot_df.columns = new_cols

    # 7. (Optional) Reset index if you want CarParkID as a column
    pivot_df = pivot_df.reset_index()

    #pivot_df

    df_final = df_pred.merge(pivot_df, on='CarParkID', how='inner')

    #df_final

    # Initialize client
    client = bigquery.Client(project=my_project_id)


    # Configure job to overwrite the table
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
    )

    # Write DataFrame to BigQuery
    job = client.load_table_from_dataframe(df_final, bigquery_dataset_output, job_config=job_config)
    job.result()  # Wait for the job to complete







# PIPELINE
#################################################################################################################################################################
@dsl.pipeline(name="bigquery-preprocessing-pipeline")    
def pipeline(my_project_id: str = "involuted-forge-456406-a1",
             bigquery_dataset_input: str = "smart-car-park-availability-1.lta_data.view_carpark_availability_for_prediction",
             bigquery_dataset_output: str = "precise-tube-456807-h5.externalData.predictions_new",
             checked_parquet: str = "gs://carpark_predict_temp/df_checked.parquet",
             checked_12_parquet: str = "gs://carpark_predict_temp/df12_checked.parquet",
             sequence_parquet: str = "gs://carpark_predict_temp/sequenced.parquet",
             pkl_file_path_1: str = "gs://carpark_predict_models/4april2025/label_encoder_Carparks.pkl",
             pkl_file_path_2: str = "gs://carpark_predict_models/4april2025/min_max_scaler_AvailableLots.pkl",
             pkl_file_path_3: str = "gs://carpark_predict_models/4april2025/min_max_scaler_Carparks.pkl",
             pkl_file_path_4: str = "gs://carpark_predict_models/4april2025/min_max_scaler_Min_Hour_Day.pkl",
             model_keras_30: str = "gs://carpark_predict_models/26APR/LSTM_Model_26APR_30mins.keras",
             model_keras_45: str = "gs://carpark_predict_models/26APR/LSTM_Model_26APR_45mins.keras",
             model_keras_60: str = "gs://carpark_predict_models/26APR/LSTM_Model_26APR_60mins.keras",
             run_id: str = "no_run_id" ):

    from kfp import dsl

    load_table_task = load_check_data(bigquery_dataset_input = bigquery_dataset_input, 
                      my_project_id = my_project_id, 
                      checked_parquet = checked_parquet, 
                      checked_12_parquet = checked_12_parquet,
                      run_id=run_id)
    
    load_table_task.set_caching_options(False) # Disable caching for this component

    with dsl.Condition(load_table_task.outputs["success"] == 'true'):

        preprocess_task = preprocess_data(checked_parquet = checked_parquet,
                                          sequence_parquet = sequence_parquet,
                                          pkl_file_path_1 = pkl_file_path_1,
                                          pkl_file_path_2 = pkl_file_path_2,
                                          pkl_file_path_3 = pkl_file_path_3,
                                          pkl_file_path_4 = pkl_file_path_4,
                                          run_id=run_id)
        preprocess_task.set_caching_options(False) # Disable caching for this component
        
        with dsl.Condition(preprocess_task.outputs["success"] == 'true'):

            prediction_task = predict_with_lstm(model_keras_30 = model_keras_30,
                                                model_keras_45 = model_keras_45,
                                                model_keras_60 = model_keras_60,
                                                checked_parquet = checked_parquet,
                                                checked_12_parquet = checked_12_parquet,
                                                sequence_parquet = sequence_parquet,
                                                pkl_file_path_2 = pkl_file_path_2,
                                                my_project_id = my_project_id,
                                                bigquery_dataset_output = bigquery_dataset_output,
                                                run_id=run_id)
            
            prediction_task.set_caching_options(False) # Disable caching for this component


############################################################################################################################

