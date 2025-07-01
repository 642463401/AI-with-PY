import requests
import base64
import os
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=[your id]R&client_secret=[your secret]'
response = requests.get(host)
if response:
    t=response.json()
    token=t['access_token']
list=os.listdir('C:/Users/64246/Desktop/TP')
request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/multi_object_detect"
for i in list:
    domain = os.path.abspath('C:/Users/64246/Desktop/TP')
    i=os.path.join(domain,i)
    i=open(i,'rb')
    img=img = base64.b64encode(i.read())
    params = {"image":img}
    access_token =token
    request_url = request_url + '?access_token=' + access_token
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    response = requests.post(request_url, data=params, headers=headers)
    if response:
        a=response.json()['result']
        print(a)