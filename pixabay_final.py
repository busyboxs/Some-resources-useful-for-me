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


def get_pic_num_and_page_num(keyword):
    url = f'https://pixabay.com/zh/images/search/{keyword}/'
    res = requests.get(url)

    assert "抱歉，我们没找到相关信息。" not in res.text, "抱歉，我们没找到相关信息。"

    pic_num = re.findall('<h1 .*>(.*)</h1>', res.text)[0].split()[0]
    print(f'There are total {pic_num} images about {keyword}')
    comment = re.compile('<input name=\"pagi\"(.*?)&nbsp;', re.DOTALL)
    page_num = comment.findall(res.text)[0].split()[-1]
    print(f'There are total {page_num} pages of images')
    return pic_num, page_num


def pixabay_spider(keyword, path):
    assert keyword is not None, "please input keyword"
    path = os.path.join(path, keyword)
    if not os.path.exists(path):
        os.makedirs(path)

    pic_num, page_num = get_pic_num_and_page_num(keyword)
    for page in range(1, int(page_num) + 1):
        url = f'https://pixabay.com/zh/images/search/{keyword}/?pagi={page}'
        res = requests.get(url)
        links = re.findall('data-lazy=\"(.*?)\"', res.text)  # part one image links, total 84
        links += re.findall('img srcset=\".*?\" src=\"(.*?)\"', res.text)  # part two image links, total 16, all is 100
        pool = Pool(processes=4)
        pool.map(partial(download_images, path=path), links)


if __name__ == '__main__':
    pixabay_spider(keyword='cat', path='D:/spider/pixabay')
