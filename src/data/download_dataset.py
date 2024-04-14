
import os
import gdown
import zipfile
#upload dataset to google drive

def download_dataset():

    # Download the dataset from the google drive
    url = 'https://drive.google.com/uc?id=1gT9DrBHVwP84KrWxHu5qpxyIA7xt_cF0'
    output = 'data1.zip'

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    # Unzip the dataset
    with zipfile.ZipFile(output, 'r') as zip_ref:
    # Extract all the contents to the specified directory
        print('Extracting the dataset...')
        zip_ref.extractall()
        print('Dataset extracted successfully!')

    # Remove the zip file
    os.remove('data1.zip')

    print('Dataset downloaded and unzipped successfully!')


if __name__ == '__main__':

    download_dataset()
