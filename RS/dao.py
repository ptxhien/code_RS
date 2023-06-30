import sqlalchemy
import pandas as pd
import http.client
import urllib.parse
import json

# -*- coding: utf-8 -*-

# -------------------- CONSTANTS ----------------
db_connection_str = 'mysql+pymysql://root:Hien1987@localhost:3306/test'
try:
    with open("./secret.txt") as secretFile:
        SECRETS = secretFile.readlines()
        db_connection_str = SECRETS[0].rstrip(
            "\n") or 'mysql+pymysql://root:Hien1987@localhost:3306/test'
except:
    db_connection_str = 'mysql+pymysql://root:Hien1987@localhost:3306/test'


# -------------------- READ DATA --------------------
def create_connection():
    engine = sqlalchemy.create_engine(db_connection_str)
    return engine


def select_l(conn):
    sql = 'SELECT * FROM learner'
    df = pd.read_sql(sql, conn)
    return df


def select_courseOnline(conn):
    sql = 'SELECT * FROM onlineCourse'
    df = pd.read_sql(sql, conn)
    return df


def select_courseOffline(conn):
    sql = 'SELECT * FROM offlineCourse'
    df = pd.read_sql(sql, conn)
    return df


def select_job(conn):
    sql = 'SELECT * FROM job'
    df = pd.read_sql(sql, conn)
    return df


def select_invoice(conn):
    sql = 'SELECT * FROM Invoice'
    df = pd.read_sql(sql, conn)
    return df


def select_rating(conn):
    sql = 'SELECT * FROM RatingLearner'
    df = pd.read_sql(sql, conn)
    return df

# -------------------- TAKE USER REQUIREMENTS + USER ATTRIBUTES LEARNER --------------------


def Find_lat_long_learner(df_Learner_now):
    conn = http.client.HTTPConnection('api.positionstack.com')
    api_key = '8fd5bc022089a47a2fc5d94d5652176d'
    df_Learner_now = df_Learner_now.fillna('')
    df_Learner_now = df_Learner_now.reset_index(drop=True)

    address1_get = df_Learner_now['address1'][0]
    address_get = df_Learner_now['address'][0]

    if address1_get != "" and address_get != "":
        df_Learner_now['location'] = df_Learner_now['address1'] + \
            ', ' + df_Learner_now['address']
        learner_address = df_Learner_now.location[0]
        learner_address_region = learner_address.split(', ')[-1]
        df_Learner_now['regionVN'] = learner_address_region

        params = urllib.parse.urlencode({
            'access_key': api_key,
            'query': learner_address,
            'region': learner_address_region,
            'limit': 1,
        })
        conn.request('GET', '/v1/forward?{}'.format(params))
        res = conn.getresponse()
        data1 = json.loads(res.read())
        # data1 = {}
        if bool(data1):
            for i in data1['data']:
                df_Learner_now['longitude'] = i['longitude']
                df_Learner_now['latitude'] = i['latitude']
                df_Learner_now['region'] = i['region']
                df_Learner_now['county'] = i['county']
                df_Learner_now['label'] = i['label']
        else:
            df_Learner_now['longitude'] = ""
            df_Learner_now['latitude'] = ""
            df_Learner_now['region'] = ""
            df_Learner_now['county'] = ""
            df_Learner_now['label'] = ""

    else:
        df_Learner_now['location'] = ""
        df_Learner_now['longitude'] = ""
        df_Learner_now['latitude'] = ""
        df_Learner_now['region'] = ""
        df_Learner_now['county'] = ""
        df_Learner_now['label'] = ""

    return df_Learner_now


def User_Preq_Attributes(email, occupation, form, month, typeFilter):
    conn = create_connection()
    df_Learner = select_l(conn)
    df_Learner = df_Learner.loc[df_Learner.email == email]
    df_Learner = Find_lat_long_learner(df_Learner)

    Requirement_Learner = []
    if month != "":
        Requirement_Learner.append({'Occupation': str(occupation), 'Form_require': str(
            form), 'duration': int(month), 'typeFilter': str(typeFilter)})
    # elif month == "":
    else:
        Requirement_Learner.append({'Occupation': str(occupation), 'Form_require': str(
            form), 'duration': '00', 'typeFilter': str(typeFilter)})

    df_requirement_Learner = pd.DataFrame(Requirement_Learner)
    df_requirement_Learner['learnerID'] = df_Learner[['learnerID']]
    df_attribute_requirement = pd.merge(
        df_Learner, df_requirement_Learner, how='left', on='learnerID')

    if month != '':
        second = df_attribute_requirement['duration'] * 30 * 24 * 3600
    else:
        second = 0
    df_attribute_requirement['durationSecond'] = second

    return df_attribute_requirement
