# -*- coding: utf-8 -*-
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import sys
import time
from random import random
import hashlib
# ignore InsecureRequestWarning
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def save_img(file, src):
    """
    This function is used to save pictures.
    Initiates an HTTP request to the picture URL,
    gets the binary code,
    writes the code to the local file,
    and completes the preservation of a picture.
    :param file:folder path
    :param src: image url
    :return:
    """
    if os.path.exists(file):
        print(f'-{file}已存在，跳过。-')
    else:  # This is done simply to dedup process
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                              'Chrome/68.0.3440.106 Safari/537.36'}
            res = requests.get(src, timeout=3, verify=False, headers=headers)
            # print(res.content)
        except Exception as e:
            print(f'--{e}--')
            return False
        else:
            if res.status_code == 200:
                img = res.content
                open(file, 'wb').write(img)
                time.sleep(random())
                return True


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


def pixabay(keyword, save_root):
    img_cnt = 0
    folder, key = keyword, keyword
    if not folder:
        sys.exit('Please input the keyword you want to search.')
    if not key:
        sys.exit('Please input the keyword you want to search.')

    query_url = f'https://pixabay.com/zh/images/search/{key}/'
    query_res = requests.get(query_url)
    query_soup = BeautifulSoup(query_res.text, 'lxml')
    pic_num = query_soup.h1.text  # str
    print(f'There are total {pic_num} images about {keyword}')
    page_num = query_soup.select('.add_search_params')[0].text.strip().lstrip('/ ')
    print(f'There are total {page_num} pages of images')
    if "抱歉，我们没找到相关信息。" in query_res.text:
        return "抱歉，我们没找到相关信息。"
    for page in range(1, int(page_num) + 1):
        url = f'https://pixabay.com/zh/images/search/{keyword}/?pagi={page}'
        res = requests.get(url)
        soup = BeautifulSoup(res.text, 'lxml')
        items = soup.select('.credits .item')
        # print(items)
        for item in items:
            img_attrs = item.a.img.attrs
            if img_attrs['src'] == '/static/img/blank.gif':
                img_src = img_attrs['data-lazy']
            else:
                img_src = img_attrs['src']
            path = os.path.join(save_root, keyword)
            if not os.path.exists(path):
                os.makedirs(path)
            filename = img_name_processor(img_src)
            file = os.path.join(path, filename)
            rt = save_img(file=file, src=img_src)  # 是save_img的返回值，
            img_cnt += 1
            if img_cnt == pic_num:
                print(f"已搜集{img_cnt}张{key}图片，程序退出...")
                break
            if rt:
                print(f'第{img_cnt}张[{key}]图片保存成功...')
            else:
                print(f'第{img_cnt}张[{key}]图片保存失败...')


if __name__ == "__main__":
    # 女脸，女性 脸，美女，
    pixabay('cat', 'D:/spider/pixabay')
