#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import logging
import os
import sys
import threading
import subprocess

# ------------------------------------
# logging settings
# ------------------------------------
error = logging.critical
warn = logging.warning
debug = logging.debug
info = logging.info

# ------------------------------------
# functions
# ------------------------------------
def raise_error(info):
    error(info)
    sys.exit(1)


def make_dir(name, nresume):
    if nresume:
        if os.path.isdir(name):
            run_cmd('rm -rf '+name)
        run_cmd('mkdir '+name)
    else:
        if not os.path.isdir(name):
            run_cmd('mkdir '+name)


def run_cmd(cmd,type='call'):
    info(f'{cmd}')
    # out = os.system(cmd)
    if type == 'call':
        out = subprocess.call(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
        if out != 0:
            raise_error(f'Error: {cmd}')
    elif type == 'result':
        result = subprocess.Popen(cmd, stdout=subprocess.PIPE,stderr=subprocess.PIPE, shell=True)
        out = result.communicate()[0].decode().strip()
        return out


class MyThread(threading.Thread):
    def __init__(self,func,args=()):
        super(MyThread,self).__init__()
        self.func = func
        self.args = args
 
    def run(self):
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None
        
