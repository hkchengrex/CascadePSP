import os
import requests
import hashlib


def download_model(destination):
    print('Downloading model file into', destination)
    URL = 'https://github.com/hkchengrex/CascadePSP/releases/download/v1.0/model'

    with open(destination, 'wb') as file:
        response = requests.get(URL)
        file.write(response.content)

    print('Download completed.')


def check_model(destination):
    correct_md5 = '7478d4a9c42ab52beb6d7e9683402fe0'

    with open(destination, 'rb') as f:
        md5_returned = hashlib.md5(f.read()).hexdigest()

    return correct_md5 == md5_returned


def download_and_or_check_model_file(destination):

    if os.path.exists(destination):
        # print('Model file exists.')
        pass
    else:
        print('Model does not exist.')
        download_model(destination)

    md5_passed = check_model(destination)

    if md5_passed:
        return

    print('MD5 of the model file does not match')
    print('Downloading the model again...')
    download_model(destination)

    md5_passed = check_model(destination)
    assert md5_passed, 'MD5 still does not pass'
    

