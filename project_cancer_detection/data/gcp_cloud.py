from google.cloud import storage
from project_cancer_detection.ml_logic.params import PROJECT,BUCKET_NAME

##### CLient ####
client = storage.Client()
bucket = storage.Bucket(client, BUCKET_NAME)

console_url = 'https://console.cloud.google.com/storage/browser/cancer-detection-small-datasets'
gsutil_URI = 'gs://cancer-detection-small-datasets'

#blobs = bucket.list_blobs()
#url = 'https://storage.googleapis.com/cancer-detection-small-datasets/test_1k/0/01132f290bf88852e10d23ea2f6f5557ba624f74.tif'

blob = bucket.blob('test_1k/photo_2022-03-25_10-21-32.jpg')
#blob.download_to_filename('test.jpg')

train_path = 'gs://cancer-detection-small-datasets/test_1k/'
