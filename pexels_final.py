import requests
import re
import hashlib
from pprint import pprint
import os
import time
import sys
from random import random
from multiprocessing import Pool
from functools import partial

# ignore InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def img_name_processor(src):
    """
    This function is used to handle the file name of the saved picture.
    Hash the URL of the picture as its filename.
    :param src: image url
    :return: image filename
    """
    h5 = hashlib.md5()
    h5.update(src.encode('utf-8'))
    img = h5.hexdigest() + '.jpg'
    return img


def download_images(link, path):
    image_name = img_name_processor(link)
    print('Download image {}...'.format(image_name))
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/68.0.3440.106 Safari/537.36'}
        pic = requests.get(link, timeout=3, headers=headers, verify=False)
    except Exception as e:
        print(f'{e}')
        return False
    else:
        if pic.status_code == 200:
            img = pic.content
            file = os.path.join(path, image_name)
            if os.path.exists(file):
                print(f'{file} exists.')
                return
            with open(file, 'wb') as f:
                f.write(img)
            time.sleep(random())
        return True


def pexels_spider(keyword, path):
    assert keyword is not None, "please input keyword"
    path = os.path.join(path, keyword)
    if not os.path.exists(path):
        os.makedirs(path)
    for page in range(1, 50):
        print(f'Download pictures from page {page}')
        pexels_url = f'https://www.pexels.com/search/{keyword}/?page={page}'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) '
                                 'AppleWebKit/537.36 (KHTML, like Gecko) '
                                 'Chrome/68.0.3440.106 Safari/537.36'}
        res = requests.get(pexels_url, timeout=3, headers=headers, verify=False)
        if 'Sorry, no pictures found!' in res.text:
            print('========== Finish ==========')
            sys.exit(0)

        pic_links = re.findall('<a download="true" href=\"(.*)\">', res.text)
        # for link in pic_links:
        pool = Pool(processes=4)
        pool.map(partial(download_images, path=path), pic_links)


if __name__ == '__main__':
    pexels_spider(keyword='cat', path='D:/spider/pexels')
