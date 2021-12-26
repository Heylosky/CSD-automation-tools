import redis
import datetime
from Match_mode import Matchmode

# redis client
r = redis.Redis(host='172.16.10.215', port=6379, db=1, decode_responses=True, password='123456')
# pop data
# data = r.blpop("sms_tt")[1]
# print(data)

# # cd = country_code, ph=phone_number, mk=convert_mark, tm=send_time
# cd = data.split('\t')[4]
# tm = data.split('\t')[5]
# ph = data.split('\t')[11]
# mk = data.split('\t')[12]

# send_time = datetime.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")
# send_time = datetime.datetime.strftime(send_time, "%Y-%m-%dT%H:%M:%S+0800")

while True:
    data = r.blpop("sms_tt")[1]
    try:
        ch = data.split('\t')[2]
        cd = data.split('\t')[4]
        tm = data.split('\t')[5]
        sd = data.split('\t')[9]
        sd2 = data.split('\t')[10]
        ph = data.split('\t')[11]
        mk = data.split('\t')[12]
        send_time = datetime.datetime.strptime(tm, "%Y-%m-%d %H:%M:%S")
        send_time = datetime.datetime.strftime(send_time, "%Y-%m-%dT%H:%M:%S+0800")
        if cd == '55': 
            Matchmode.match55(ph,mk,send_time,sd,sd2,ch)
        else:
            Matchmode.matchPrefix(cd,ph,mk,send_time,sd,sd2,ch)
    except IndexError:
        continue