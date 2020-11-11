import pandas as pd
import requests
from pandas.io.json import json_normalize
import json
import urllib
from sqlalchemy import create_engine
import sqlalchemy
import smtplib




try:
    url = 'https://api.hubapi.com/deals/v1/deal/paged?'

    querystring = {"limit":"250","archived":"false","hapikey":"redacted","properties":"marketing_deal_source"}
    querystring2 = {"limit":"250","archived":"false","hapikey":"redacted","properties":"dealname"}
    querystring3 = {"limit":"250","archived":"false","hapikey":"redacted","properties":"createdate"}
    querystring4 = {"limit":"250","archived":"false","hapikey":"redacted","properties":"team_lead"}
    querystring5 = {"limit":"250","archived":"false","hapikey":"redacted","properties":"project_id"}

    deal_list = []

    has_more = True
    while has_more:
        response = requests.request("GET", url, params=querystring)
        jsonresponse = response.json()
        has_more = jsonresponse['hasMore']
        deal_list.extend(jsonresponse['deals'])
        querystring['offset']= jsonresponse['offset']



    df = pd.DataFrame.from_dict(deal_list)
    df.drop(["imports", "stateChanges"], axis = 1, inplace = True) 

    deal_list2 = []

    has_more2 = True
    while has_more2:
        response2 = requests.request("GET", url, params=querystring2)
        jsonresponse2 = response2.json()
        has_more2 = jsonresponse2['hasMore']
        deal_list2.extend(jsonresponse2['deals'])
        querystring2['offset']= jsonresponse2['offset']
    
    df2 = pd.DataFrame.from_dict(deal_list2) 

    df = df.merge(df2[['dealId','properties']],on='dealId',how='left')


    deal_list3 = []

    has_more3 = True
    while has_more3:
        response3 = requests.request("GET", url, params=querystring3)
        jsonresponse3 = response3.json()
        has_more3 = jsonresponse3['hasMore']
        deal_list3.extend(jsonresponse3['deals'])
        querystring3['offset']= jsonresponse3['offset']
    
    df3 = pd.DataFrame.from_dict(deal_list3)

    df = df.merge(df3[['dealId','properties']],on='dealId',how='left')
    
    deal_list4 = []

    has_more4 = True
    while has_more4:
        response4 = requests.request("GET", url, params=querystring4)
        jsonresponse4 = response4.json()
        has_more4 = jsonresponse4['hasMore']
        deal_list4.extend(jsonresponse4['deals'])
        querystring4['offset']= jsonresponse4['offset']
    
    df4 = pd.DataFrame.from_dict(deal_list4)

    df = df.merge(df4[['dealId','properties']],on='dealId',how='left')
    
    deal_list5 = []

    has_more5 = True
    while has_more5:
        response5 = requests.request("GET", url, params=querystring5)
        jsonresponse5 = response5.json()
        has_more5 = jsonresponse5['hasMore']
        deal_list5.extend(jsonresponse5['deals'])
        querystring5['offset']= jsonresponse5['offset']
    
    df5 = pd.DataFrame.from_dict(deal_list5)

    df = df.merge(df5[['dealId','properties']],on='dealId',how='left')
    df.columns = pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)
    

# rename the columns with the cols list.
    df.rename(columns={'properties_x.1': 'createDate', 'properties_x': 'dealSource','properties_y':'dealName','properties_y.1': 'teamLead'}, inplace=True)


    engine = create_engine('mysql+pymysql://data-legacy:redacted@redacted.com:25060/defaultdb')

    df.to_sql("HubSpot_Raw_DealSource", con = engine, if_exists='replace', dtype={'dealSource': sqlalchemy.types.JSON,'dealName':sqlalchemy.types.JSON ,'createDate':sqlalchemy.types.JSON, 'teamLead':sqlalchemy.types.JSON});
except Exception as e:
    fromaddr = 'desiree@redacted'
    toaddrs  = 'desiree@redacted'
    sub = 'Error in AirTable_CXStatus Load'
    msg = 'Subject: {}\n\n{}'.format(sub, str(e))
    username = 'desiree@redacted'
    password = 'redacted'
    server = smtplib.SMTP('smtp.gmail.com:587')
    server.starttls()
    server.login(username,password)
    server.sendmail(fromaddr, toaddrs, msg)
    server.quit()
