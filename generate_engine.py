# -*- coding: utf-8 -*-
import random
# file_object = open('./thefile.txt','w')
file_object = open("./logfile.txt",'w')
# 制造日志


def produce():
    # 点击 1
    # 播放 2
    # 点赞 2
    # 收藏 4
    # 付费观看 5
    # 站外分享 6
    # 评论 7
    albet_num = ["1", "2", "3", "4", "5", "6", "7", "8", "a", "b", "c", "d", "e", "f", "g", "h", "J", "K", "L", "M", "N"]
    user_list = ["one", "two", "three", "four", "five"]
    Num = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    log_type_array = ["1", "2", "3", "4", "5", "6", "7"]
    topic_array = ["空气净化器", "净水器", "加湿器", "空气净化滤芯"]
    for num in range(0, 2000):
        cookie = "".join(random.sample(albet_num, 9))
        uid = "".join(random.sample(user_list, 1))
        user_agent = "Macintosh Chrome Safari"
        ip = "192.168.125.19"
        video_id = "".join(random.sample(Num, 5))
        topic_id = "".join(random.sample(topic_array, 1))
        topic = "苹果发布会"
        order_id = "0"
        log_type = "".join(random.sample(log_type_array,1))
        
        final = cookie + "&" + uid + "&" + user_agent + "&" + ip + "&" +  video_id + "&" + topic_id + "&" + order_id + "&" + log_type +"\n"
        # print final
        file_object.write(final)
    file_object.close()
        
            
produce()
# file = open("./thefile.txt")
# click_action = {} #key:uid, value:videoid
# for line in file.readlines():
#    line = line.strip()
#    ls = line.split("&")
#    if ls[7] != "1":
#        continue
#    if ls[1] not in click_action.keys():
#        click_action[ls[1]] = []
#    click_action[ls[1]].append(ls[4])
#        
    
# for k,v in click_action.items():
#    print k + "\t" + str(len(v)) + "\t" + "&&".join(v)
file = open("./logfile.txt")
cate_its = {}
for line in file.readlines():
    line = line.strip()
    ls = line.split("&")
    if ls[5] not in cate_its.keys():
        cate_its[ls[5]]=[]
    cate_its[ls[5]].append(ls[4])
    
file_object = open('cate.log','w')    
for k, v in cate_its.items():
    line = k + "\t" + "&&".join(v) + "\n"
    file_object.write(line)
