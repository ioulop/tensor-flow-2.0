# -*- coding: utf-8 -*-
"""@autor:lifan"""
import os
import socket
#HOST, PORT = '', 8888
ss = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ss.connect(('127.0.0.1', 8888))
# ss.sendall('加湿器')
ss.sendall(bytes('9&&1',encoding='utf-8'))
#ss.sendall(bytes('57423&& 2', encoding='utf-8'))
# ss.sendall(bytes('57423',encoding='utf-8'))
# os.system('sleep 1')
ss.send(bytes('EOF', encoding='utf-8'))
data = ss.recv(1024)
data = data.decode()
print("server dafu %s" % data)
ss.close()


