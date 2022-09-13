from http import client
from google.cloud import storage
import os

bucket_name = 'cancer-detection-small-datasets'
#prefix = '0_test/'
dl_dir = 'raw_data/'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/naz/ethereal-runner-356713-a4caa9e26f85.json"
client = storage.Client()
bucket = client.get_bucket(bucket_name)
blobs = bucket.list_blobs()
for blob in blobs:
    filename = blob.name
    if not filename.endswith('/'):
        path= os.path.split(dl_dir+filename)[0]
        os.makedirs(path, exist_ok=True)
        print(f"Download: {dl_dir + filename}")
        blob.download_to_filename(dl_dir + filename)  #
