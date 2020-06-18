# -*- coding:utf-8 -*-
import redis
r = redis.Redis("127.0.0.1", 6379, db=0)
r2 = redis.Redis("127.0.0.1", 6380, db=0)
while True:
    m = r.get("9527#1")
    r2.set("9527#1",m)
    print(r2.get("9527#1"))