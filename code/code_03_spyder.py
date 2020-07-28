#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: 代码医生工作室 
@公众号：xiangyuejiqiren   （内有更多优秀文章及学习资料）
@来源: <机器视觉之TensorFlow2入门原理与应用实战>配套代码 
@配套代码技术支持：bbs.aianaconda.com  
Created on Thu Mar  7 14:55:44 2019
"""

import os
import re
import json
import socket
import urllib.request,urllib.parse,urllib.error


import time  # 引入time模块用于设置超时

timeout = 5
socket.setdefaulttimeout(timeout)


class Crawler:  # 定义爬虫类

    def __init__(self, t=0.1):  # 初始化类方法
        self.time_sleep = t  # 定义睡眠时长，时间单位是秒

    # 开始获取图片
    def __getImages(self, word='美女'):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount  # 当前采集的图片计数
        while pn < self.__amount:

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
            url = 'http://image.baidu.com/search/avatarjson?tn=resultjsonavatarnew&ie=utf-8&word=' + search + '&cg=girl&pn=' + str(
                pn) + '&rn=60&itg=0&z=0&fr=&width=&height=&lm=-1&ic=0&s=0&st=-1&gsm=1e0000001e'

            # 设置header，并进行网页爬取
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=headers)
                page = urllib.request.urlopen(req)
                data = page.read().decode('utf8')
            except UnicodeDecodeError as e:
                print('-----UnicodeDecodeErrorurl:', url)

                print("下载下一页")
                pn += 60  # 读取下一页
            except urllib.error.URLError as e:
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print("-----socket timout:", url)
            else:
                json_data = json.loads(data)  # 解析json
                self.__saveImage(json_data, word)

                print("下载下一页")
                pn += 60  # 读取下一页
            finally:
                page.close()
        print("下载任务结束")
        return

    def __saveImage(self, json, word):  # 定义保存图片的类方法

        if not os.path.exists("./" + self.__savedir):
            os.mkdir("./" + self.__savedir)

        # 获取已经采集的图片数量，用于图片命名
        self.__counter = len(os.listdir('./' + self.__savedir)) + 1
        for info in json['imgs']:
            try:
                if self.__downloadImage(info, word) == False:
                    self.__counter -= 1
            except urllib.error.HTTPError as urllib_err:
                print(urllib_err)
                pass
            except Exception as err:
                time.sleep(1)
                print(err);
                print("产生未知错误，放弃保存")
                continue
            finally:
                print("采集图片+1,已采集到" + str(self.__counter) + "张图片")
                self.__counter += 1
        return

    def __downloadImage(self, info, word):  # 定义私有方法，下载图片
        time.sleep(self.time_sleep)
        fix = self.__getFix(info['objURL'])
        urllib.request.urlretrieve(info['objURL'], './' + self.__savedir + '/'
                                   + str(self.__counter) + "_" + str(round(time.time())) + str(fix))

    def __getFix(self, name):  # 定义私有方法，获取图片后缀名
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    def __getPrefix(self, name):  # 定义私有方法，获取图片前缀名
        return name[:name.find('.')]

    # 定义start方法，实现爬虫的入口
    def start(self, keyword,  # 搜索的关键词
              savedir,  # 爬取的图片存贮目录
              spider_page_num=1,  # 需要抓取数据页数，抓取图片总数量=页数x60
              start_page=1):  # 爬取的起始页数
        self.__savedir = savedir
        self.__start_amount = (start_page - 1) * 60  # 每页60个图片
        self.__amount = spider_page_num * 60 + self.__start_amount  # 定义抓取图片的总数量=页数x60
        self.__getImages(keyword)


crawler = Crawler(0.01)
# crawler.start('模特 黑人', "org_black", 500)
crawler.start('模特 白人', "org_white2", 500)
