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
import sys
import urllib
import json
import socket
import urllib.request,urllib.parse,urllib.error
# 设置超时
import time  # 引入time模块用于设置超时

timeout = 5
socket.setdefaulttimeout(timeout)


class Crawler:  # 定义爬虫类

    def __init__(self, t=0.1): # 初始化类方法
        self.time_sleep = t # 定义睡眠时长，时间单位是秒
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}

    # 获取后缀名
    @staticmethod
    def get_suffix(name):
        m = re.search(r'\.[^\.]*$', name)
        if m.group(0) and len(m.group(0)) <= 5:
            return m.group(0)
        else:
            return '.jpeg'

    # 保存图片
    def __saveImage(self, rsp_data, word):
        if not os.path.exists("./" + self.__savedir):
            os.mkdir("./" + self.__savedir)
        # 判断名字是否重复，获取图片长度
        self.__counter = len(os.listdir('./' + self.__savedir)) + 1
        for image_info in rsp_data['data']:
            try:
                if 'replaceUrl' not in image_info or len(image_info['replaceUrl']) < 1:
                    continue
                obj_url = image_info['replaceUrl'][0]['ObjUrl']
                thumb_url = image_info['thumbURL']
                url = 'https://image.baidu.com/search/down?tn=download&ipn=dwnl&word=download&ie=utf8&fr=result&url=%s&thumburl=%s' % (urllib.parse.quote(obj_url), urllib.parse.quote(thumb_url))
                time.sleep(self.time_sleep)
                suffix = self.get_suffix(obj_url)
                # 指定UA和referrer，减少403
                opener = urllib.request.build_opener()
                opener.addheaders = [
                    ('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.116 Safari/537.36'),
                ]
                urllib.request.install_opener(opener)
                # 保存图片
                filepath = './%s/%s' % (self.__savedir, str(self.__counter) + str(suffix))
                urllib.request.urlretrieve(url, filepath)
                if os.path.getsize(filepath) < 5:
                    print("下载到了空文件，跳过!")
                    os.unlink(filepath)
                    continue
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

    # 开始获取
    def __getImages(self, word):
        search = urllib.parse.quote(word)
        # pn int 图片数
        pn = self.__start_amount
        while pn < self.__amount:

            url = 'https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord=%s&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=&hd=&latest=&copyright=&word=%s&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&fr=&expermode=&force=&pn=%s&rn=%d&gsm=1e&1594447993172=' % (search, search, str(pn), self.__per_page)
            # 设置header防403
            try:
                time.sleep(self.time_sleep)
                req = urllib.request.Request(url=url, headers=self.headers)
                page = urllib.request.urlopen(req)
                rsp = page.read()
            except UnicodeDecodeError as e:
                print('-----UnicodeDecodeErrorurl:', url)

                print("下载下一页")
                pn += 60  # 读取下一页
            except urllib.error.URLError as e:
                print("-----urlErrorurl:", url)
            except socket.timeout as e:
                print("-----socket timout:", url)
            else:
                json_data = json.loads(rsp)  # 解析json
                self.__saveImage(json_data, word)

                print("下载下一页")
                pn += 60  # 读取下一页
            finally:
                # page.close()
                pass
        print("下载任务结束")
        return


    def start(self, word,   # 搜索的关键词
              savedir,      # 爬取的图片存贮目录
              per_page,
              total_page=10, # 需要抓取数据页数
              start_page=2): # 爬取的起始页数
              
        self.__savedir = savedir
        self.__per_page = per_page - 1
        self.__start_amount = (start_page - 1) * self.__per_page
        self.__amount = total_page * self.__per_page + self.__start_amount
        self.__getImages(word)



crawler = Crawler(0.05) 
crawler.start('模特 黑人', "org_black", 5) 
crawler.start('模特 白人', "org_white", 5)
