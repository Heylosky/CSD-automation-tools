from elasticsearch import Elasticsearch
import re
import datetime

ES = ['172.16.10.60:30920']
es = Elasticsearch(ES)

class OperateES:
    @staticmethod
    def es_query(countrycode):
        query_json = {
            "query": {
                "match": {
                    "Country_Code": countrycode
                    }
                },
            "_source": ["Carrier_Code","Country","Carrier","Country_Code"],
            "size": 50
            }
        country_data = es.search(index="telephonelist-v2",body=query_json)
        return country_data
    def toes(a):
        es = Elasticsearch(ES)
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        es.index(index="sms-tiktok-"+now, body=a)

class Matchmode:
    @staticmethod
    def match55(phone_num,mark,timestamp,serial_number,d_serial_number,channel):
        teldata = OperateES.es_query(55)
        output_dict = {'Country_Code': '55', 'Carrier_Code': '-', 'Country': 'Brazil', 'Carrier': '-'}
        for hits in teldata['hits']['hits']:
            if re.match(hits['_source']['Carrier_Code'].replace('x','\d')+"\d+", phone_num) :
                output_dict = hits['_source']
        side_data = {'@timestamp':timestamp, 'phone_num':phone_num, 'backfill_mark':mark, 'serial_number':serial_number, 'downstream_serial_number':d_serial_number, 'channel':channel}
        output_dict.update(side_data)
        print(output_dict)
        OperateES.toes(output_dict)
    
    def matchPrefix(country_code,phone_num,mark,timestamp,serial_number,d_serial_number,channel):
        teldata = OperateES.es_query(country_code)
        output_dict = {'Country_Code': country_code, 'Carrier_Code': '-', 'Country': '-', 'Carrier': '-'}
        for hits in teldata['hits']['hits']:
            if re.match(hits['_source']['Carrier_Code']+"\d+", phone_num) :
                output_dict = hits['_source']
        side_data = {'@timestamp':timestamp, 'phone_num':phone_num, 'backfill_mark':mark, 'serial_number':serial_number, 'downstream_serial_number':d_serial_number, 'channel':channel}
        output_dict.update(side_data)
        print(output_dict)
        OperateES.toes(output_dict)