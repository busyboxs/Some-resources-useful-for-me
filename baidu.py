from lxml import etree
import requests
import re
import urllib
import json
import time
import os

local_path = 'D:/database/4/'
keyword = input('请输入想要搜索图片的关键字:')
first_url = 'http://image.baidu.com/search/flip?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1530850407660_R&pv=&ic=0&nc=1&z=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&ctd=1530850407660%5E00_1651X792&word={}'.format(
    keyword)
want_download = input('请输入想要下载图片的张数:')

global page_num
page_num = 1
global download_num
download_num = 0


# 这个函数用来获取图片格式
def get_format(pic_url):
    # url的末尾存着图片的格式，用split提取
    # 有些url末尾并不是常见图片格式，此时用jpg补全
    t = pic_url.split('.')
    if t[-1].lower() != 'bmp' and t[-1].lower() != 'gif' and t[-1].lower() != 'jpg' and t[-1].lower() != 'png':
        pic_format = 'jpg'
    else:
        pic_format = t[-1]
    return pic_format


# 这个函数用来获取下一页的url
def get_next_page(page_url):
    global page_num
    html = requests.get(page_url).text
    with open('html_info.txt', 'w', encoding='utf-8') as h:
        h.write(html)
    selector = etree.HTML(html)
    try:
        msg = selector.xpath('//a[@class="n"]/@href')
        print(msg[0])
        next_page = 'http://image.baidu.com/' + msg[0]
        print('现在是第%d页' % (page_num + 1))
    except Exception as e:
        print('已经没有下一页了')
        print(e)
        next_page = None
    page_num = page_num + 1
    return next_page


# 这个函数用来下载并保存图片
def download_img(pic_urls):
    count = 1
    global download_num
    for i in pic_urls:
        time.sleep(1)
        try:
            pic_format = get_format(i)
            pic = requests.get(i, timeout=15)
            # 按照格式和名称保存图片
            with open(local_path + 'page%d_%d.%s' % (page_num, count, pic_format), 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(count), str(pic.url)))
                count = count + 1
                download_num = download_num + 1
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(count), str(pic.url)))
            print(e)
            count = count + 1
            continue
        finally:
            if int(want_download) == download_num:
                return 0


# 这个函数用来提取url中图片的url
def get_pic_urls(web_url):
    html = requests.get(web_url).text
    # 通过正则表达式寻找图片的地址，
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)
    # 返回图片地址，是一个list
    return pic_urls


if __name__ == "__main__":
    while True:
        pic_urls = get_pic_urls(first_url)
        t = download_img(pic_urls)
        if t == 0:
            break
        next_url = get_next_page(first_url)
        if next_url is None:
            print('已经没有更多图片')
            break
        pic_urls = get_pic_urls(next_url)
        t = download_img(pic_urls)
        if t == 0:
            break
        first_url = next_url
    print('已经成功下载%d张图片' % download_num)