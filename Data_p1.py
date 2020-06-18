# -*- coding:utf-8 -*-
import redis
r = redis.Redis("127.0.0.1", 6379, db=0)
while True:
    m = r.get("9527#1")
    if m == None:
        r.set("9527#1", 1)
    m = int(m) + 1
    r.set("9527#1", m)
    print(r.get("9527#1"))
