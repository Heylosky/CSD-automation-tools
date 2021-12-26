import os
import pandas as pd
from elasticsearch import Elasticsearch

def getFileList(path):
    path=path
    file_path=[]
    for filename in os.listdir(path):
        file_path.append((os.path.join(path,filename)))
    return file_path

def ReadFile(path):
    files = getFileList(path)
    dfSent = pd.DataFrame()
    for filename in files:
        df = pd.read_excel(filename, converters={0:str})
        dfSent = pd.concat([df,dfSent],axis=0,ignore_index=True)
    return dfSent

def UpdateES(serial_number):
    update_by_query = {
      "query": {
        "term": {
          "serial_number": {
            "value": serial_number
          }
        }
      },
      "script": {
        "source": "ctx._source.backfill_mark='1'",
        "lang": "painless"
      }
    }
    es.update_by_query(index="sms-tiktok-*", body=update_by_query)


df_convert = ReadFile(r'C:\Users\PLTM7005\Desktop\Tiktok-ELK\convert')
es = Elasticsearch(['172.16.10.60:30920'])
for serial_number in df_convert[df_convert.columns[0]] :
    UpdateES(serial_number)
    print(serial_number)
