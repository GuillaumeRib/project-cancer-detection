from http import client
from google.cloud import storage
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/naz/ethereal-runner-356713-a4caa9e26f85.json"
client = storage.Client()
bucket = client.get_bucket('cancer-detection-small-datasets')
blob = storage.Blob('train_1k/0/0056b591aecd870591760b3964792c5b584e7cfa.tif', bucket)
with open('test.tif', 'wb') as file_obj:
    client.download_blob_to_file(blob, file_obj)
