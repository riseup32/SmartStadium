#!/home/jihye/anaconda3/bin/python
import numpy as np
import pandas as pd
from dateutil.parser import parse
import matplotlib.pyplot as plt
import seaborn as sns
import imp
import os, sys
from sklearn import metrics
import warnings

import pymysql  '''mysql 패키지'''

warnings.filterwarnings("ignore")

os_sep = os.sep 
home = os.path.expanduser("~")   '''home = os.getenv("HOME")'''
np.random.seed(42)

'''DB에서 파일 갱신'''
rows=[]
sql='''
-- 예매량 데이터 셋 추출
SELECT 
    game_date AS '경기일',
    reserve_date AS '예매일',
    zone_name AS 'zone',
    total_ticket_cnt AS '총판매량'
FROM (
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -12 DAY) AS reserve_date, zone_name, d12 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -11 DAY) AS reserve_date, zone_name, d11 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -10 DAY) AS reserve_date, zone_name, d10 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -9 DAY) AS reserve_date, zone_name, d9 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -8 DAY) AS reserve_date, zone_name, d8 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -7 DAY) AS reserve_date, zone_name, d7 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -6 DAY) AS reserve_date, zone_name, d6 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -5 DAY) AS reserve_date, zone_name, d5 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -4 DAY) AS reserve_date, zone_name, d4 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -3 DAY) AS reserve_date, zone_name, d3 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -2 DAY) AS reserve_date, zone_name, d2 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL -1 DAY) AS reserve_date, zone_name, d1 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
    UNION ALL
    SELECT game_date, adddate(str_to_date(game_date, '%Y%m%d'), INTERVAL 0 DAY) AS reserve_date, zone_name, d0 AS total_ticket_cnt FROM tb_input_data GROUP BY game_date, zone_name
) a
ORDER BY game_date ASC, reserve_date ASC, zone ASC;
'''

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)
else:
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    # SQL문 실행
    try:
        curs.execute(sql)
    except Exception:
        print("Error with query: "+sql)
        sys.exit(1)
    else:
        # 데이타 Fetch
        rows = curs.fetchall()
    #연결 종료
    conn.close()

''' db로 불러온 rows를 raw_2018에 넣는다.
'raw_갱신 데이터 년도'를 변수명으로 한다.
이때 'raw_2018'은 예매량 데이터를 저장한다.'''
raw_2018 = pd.DataFrame(list(rows))

'''Grouper함수 : 예매 데이터에서 예매일 시간과 경기일 날짜를 날짜 타입으로 바꿔주고
                날짜 차이를 통해서 경기 시작 며칠 전인지 d-day를 구하고 경기일에 따라
                존별과 날짜별 총판매량을 구해준다.
                이때, 존의 NONE값을 외야로 통합시킨다.(기존 외야가 NONE으로 사용됐기 때문)'''
def Grouper(X):
    data = X.copy()  '''데이터 복사'''
    data.columns=['경기일','예매일 시간','zone','총판매량']  '''변수명 재설정'''

    '''예매일 시간 날짜 타입으로'''
    time_res=data.loc[:,"예매일 시간"].astype(str)  ''''예매일 시간'의 타입을 str로 바꿔 time_res라 한다.'''
    time_res_day=[parse(str(x)) for x in time_res]  '''time_res를 날짜 타입으로 바꿔 time_res_day라 한다.'''
    data.loc[:,'예매일 시간']=time_res_day  '''기존 '예매일 시간'을 날짜 타입인 time_res_day로 바꿔준다.'''

    '''경기일 날짜 타입으로'''
    time_play=data.loc[:,'경기일']  ''''경기일'을 time_play로 놓는다.'''
    time_play_day=[parse(str(x)) for x in time_play]  ''' time_play를 날짜 타입으로 바꿔 time_play_day라 한다.'''
    data.loc[:,'경기일']=time_play_day  '''기존 '경기일'을 날짜 타입인 time_play_day로 바꿔준다.'''

    '''zone 'NONE'를 외야로'''
    zone=data.loc[:,'zone']  ''''zone'을 zone으로 놓는다.'''
    zone_st= [x.strip().replace('NONE', '외야') for x in zone]  ''''zone'의 NONE을 외야로 바꿔 zone_st라 한다.'''
    data.loc[:,'zone']=pd.Series(zone_st)  '''기존 'zone'을 NONE을 외야로 대체한 zone_st로 바꿔준다.'''

    '''d라는 변수 만들기 : d는 경기일과 예매일 시간의 날짜 차이이다.'''
    data['d']=data['경기일']-data["예매일 시간"]  ''''경기일'과 '예매일 시간'의 차이를 'd'라는 새로운 변수에 넣어준다.'''

    dataD=data.groupby(['경기일','zone','d'])['총판매량'].sum().unstack('zone').unstack('d')
    '''경기일에 따라 'zone'(1루, 3루, 외야, 중앙)과 'd'(d0~d13)별 '총판매량'을 구해준다.'''
    return dataD  ''' 경기일자에 따라 'zone'(1루, 3루, 외야, 중앙)과 'd'(d0~d13)별 예매량 데이터를 반환한다.'''

''' Grouper함수를 통해 전처리되어 나온 값을 grouped_2018에 넣어준다.'''
grouped_2018 = Grouper(raw_2018)  


''' Refiner함수 : multi index로 되어 있는 데이터 프레임을 단일 index 데이터 프레임 형태로 바꿔주는 함수이다.
                이때, 변수명들은 '존 이름 + d_day' 형태로 지정해준다.(ex. 3루_d13)
                또한 존별 총예매량을 합산해 전체 총예매량 변수를 추가한다.'''
def Refiner(X):
    '''경기일자'''
    time_data = X[X.columns.levels[0][0]].reset_index(level='경기일').iloc[:,0]
    ''' multi index 데이터 프레임에서 경기일자를 추출한다.'''
    
    ''' 존별로 예매량 데이터 만들기(multi index -> 단일 index)'''
    data_name = []  ''' data_name이라는 비어있는 리스트를 만든다.'''
    for i in range(0,len(X.columns.levels[0])):  ''' len(X.columns.levels[0]) : 존의 개수'''
        data_name.append('region_data'+str(i+1))
        ''' 존별 예매량 데이터를 만들기 위해 존의 개수만큼 data_name 리스트에 추가한다.'''
            
    for i in range(0,len(X.columns.levels[0])):  ''' len(X.columns.levels[0]) : 존의 개수'''
        data_name[i] = X[X.columns.levels[0][i]].reset_index(level='경기일')
        ''' 존별 예매량 데이터를 추출하여 data_name[i]에 넣어준다.'''
        data_name[i] = data_name[i].iloc[:,1:data_name[i].shape[1]]
        ''' data_name에서 경기일은 빼준다.'''
        colnames = []  ''' colnames라는 비어있는 리스트를 만든다.'''
        for j in range(0,data_name[i].shape[1]):  ''' data_name[i].shape[1] : data_name[i]의 열의 개수(d-day 개수)'''
            colnames.append(X.columns.levels[0][i]+'_d'+str(j))
            ''' '존 이름 + d_day'(ex. 3루_d13)를 변수명으로 하여 colnames라는 리스트에 추가한다.'''
        data_name[i].columns = colnames
        ''' data_name[i]의 변수명을 colnames로 대체한다.'''
            
    data = pd.concat([time_data,data_name[0],data_name[1],data_name[2],data_name[3]],axis=1)
    ''' 경기일자와 존별 데이터(1루, 3루, 외야, 중앙)를 합친다.
    
      존별 예매량 데이터를 통해 전체 예매량 데이터 만들기'''
    for i in range(0,13):  ''' d12까지 사용하므로 range(0,13)로 한다.'''
        df = pd.concat([data['1루_d'+str(i)],data['3루_d'+str(i)],data['외야_d'+str(i)],data['중앙_d'+str(i)]],axis=1)
        '''날짜별 존별 예매량을 따로 뽑아낸다.'''
        data['전체_d'+str(i)] = df.sum(1)
        ''' 존별 예매량 총합을 통해 전체 예매량을 구하여 새변를 만든다.'''
    return data  '''경기일자에 대한 존별(1루, 3루, 외야, 중앙, 전체) 예매량 데이터를 반환한다.'''

''' Refiner함수를 통해 단일 인덱스로 변환된 데이터를 reserved_data로 놓는다.
reserved_data는 최종 전처리된 예매데이터이다. '''
reserved_data = Refiner(grouped_2018)
'''print(reserved_data)'''
sql='''
-- 기본데이터 셋 추출
SELECT 
    game_date AS play_date,
    IF(visit_key='HT',1,0) AS 'HT', -- KIA
    IF(visit_key='LG',1,0) AS 'LG', -- LG
    IF(visit_key='NC',1,0) AS 'NC', -- NC
    IF(visit_key='SK',1,0) AS 'SK', -- SK
    IF(visit_key='WO',1,0) AS 'WO', -- WO
    IF(visit_key='OB',1,0) AS 'OB', -- 두산
    IF(visit_key='LT',1,0) AS 'LT', -- 롯데
    IF(visit_key='SS',1,0) AS 'SS', -- 삼성
    IF(visit_key='HH',1,0) AS 'HH', -- 한화
    IF(gweek='월',1,0) AS '월',
    IF(gweek='화',1,0) AS '화',
    IF(gweek='수',1,0) AS '수',
    IF(gweek='목',1,0) AS '목',
    IF(gweek='금',1,0) AS '금',
    IF(gweek='토',1,0) AS '토',
    IF(gweek='일',1,0) AS '일',
    IF(isWeekend+isHoliday > 0,1,0) AS '평일/주말또는공휴일'
FROM tb_input_data
GROUP BY play_date;
'''

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)
else:
    # Connection 으로부터 Cursor 생성
    curs = conn.cursor()
    # SQL문 실행
    try:
        curs.execute(sql)
    except Exception:
        print("Error with query: "+sql)
        sys.exit(1)
    else:
        # 데이타 Fetch
        rows = curs.fetchall()
    #연결 종료
    conn.close()

rawdata=pd.DataFrame(list(rows))
rawdata.columns=['play_date', 'HT', 'LG', 'NC', 'SK', 'WO', 'OB', 'LT', 'SS', 'HH', '월', '화', '수', '목', '금', '토', '일', '평일:0\n공휴일:1']
'''rawdata의 컬럼명을 위와같이 설정함. (위의 알파벳들은 각 팀명의 코드이름)'''

zone_name = ['1루','3루','외야','중앙']
'''예측을 할 네개의 존의 이름을 리스트에 저장.'''

'''32개 존별 관중수를 4개의 존으로 통합 
   rawdata에 32개의 좌석이 열 이름로 들어가있는데 그 것을 총 4개의 열 이름으로 그룹핑 하기 위함'''
for i in range(0,len(zone_name)): #'''존의 이름의 개수만큼 반복할 것이다.'''
    resultlist = [] '''존_이름이 포함된 rawdata의 열들을 넣기위한 리스트 공간'''
    for j in range(0,len(list(rawdata.columns.values))): 
        '''rawdata의 컬럼값의 개수 만큼 반복(스트링끼리 비교하여 이름을 비교하기 위함
         (rawdata의 컬럼에는 1루, 3루, 중앙, 외야 라는 스트링을 포함하는 컬럼명으로 되어 있기 때문에)'''
        if zone_name[i] in list(rawdata.columns.values)[j]: '''열벡터들의 이름이 크게 4가지로 나눈 존 이름을 포함하고 있으면'''
            resultlist.append(list(rawdata.columns.values)[j]) ''' 그 이름을 resultlist에 추가한다.'''
    
    result = [] '''뒤에서 예측된 값들은 우리가 사용하는 모델에 사용하지 않으므로 예측이 아닌 값들을 담을 리스트를 생성한다.'''
    
    ''' 데이터프레임에서 예측 관중수를 제거 
      위에서 만든 result의 리스트에 '.1' 이나 '예측'을 포함하고 있는 변수를 제외한 열 이름(스트링)을 넣는 작업.'''
    for k in range(0,len(resultlist)): 
        ''' 위에서 만든 resultlist에 있는 열 이름의 길이만큼 반복. (이 곳에는 '.1'과 '예측'스트링을 가지고 있는 열 이름들이 있음)'''
        if ('.1' in resultlist[k] or '예측' in resultlist[k])==0: ''''.1'이나 '예측'이라는 스트링을 포함하고 있지 않으면'''
            result.append(resultlist[k]) '''result리스트에 그 이름들을 추가한다.'''
     rawdata[zone_name[i]] = rawdata.loc[:,result].sum(1)
     '''rawdata에 '1루', '3루', '외야', '중앙' 4개의 컬럼변수를 추가할 것이다.
     i개를 반복하는 반복문 안에 k개를 반복하는 반복문이 포함되어 있기 때문에 한번이 끝났을 때에는 하나의 존에 대한 관중수가 나올 것이다.
     그 나온 관중수를  dataframe안에 넣는 작업이다.'''


''' 엑셀에 경기와 관련없는 행을 날리기 위한 작업'''

index = [] '''index에는 경기일이 결측값이 아닌 행들을 넣고, 그 행들만 남기기 위한 공백리스트 이다.'''
for i in range(0,len(rawdata)): '''rawdata의 행의 길이만큼 반복한다.'''
    if rawdata['play_date'].isna()[i]==0: '''만약 'play_date'(경기 날짜)가 결측값이 아니면'''
        index.append(i) '''index라는 리스트안에 그 행의 번호를 추가한다.'''
        
rawdata = rawdata.iloc[index,:] '''위에서 경기날짜가 결측값이 아닌 행들만을 선택하여 다시 데이터프레임에 덮어쓴다.'''

''''play_date'가 date의 형태인 벡터인데, 그것을 스트링형태의 변수로 만들기위해 다음작업을 시행한다.'''
time_play = rawdata['play_date'] '''먼저 벡터를 다른 공간에 저장한다.'''
time_play = time_play.astype('int') '''날짜를 바로 스트링으로 바꾸면 진짜의 날짜가 나오지 않기 때문에 먼저 인트형으로 바꾼다.'''
time_play_day = [parse(str(x)) for x in time_play] '''바꾼 인트형의 각 값들을 스트링으로 바꾼 후 다시 새로운 리스트에 넣는다.'''
rawdata['경기일자'] = time_play_day ''' rawdata의 '경기일자'라는 열이름으로 위에서 만든 스트링 날짜를 넣는다.'''


''' 변수명 재설정'''
rawdata = rawdata.rename(columns={'평일:0\n공휴일:1':'공휴일'}) '''변수명이 줄바꿈으로 되어있는 것을 편하게 아닌 것으로 바꾼다.'''
rawdata = rawdata.rename(columns={'총관중수(실제)':'총관중수'}) '''너무 긴 변수명도 가독성이 좋게 바꾼다.'''
rawdata.columns.values '''열의 이름(변수)을 확인하는 작업.'''

rawdata = pd.concat([rawdata.loc[:,'경기일자'],rawdata.loc[:,'HT':'공휴일']],axis=1)
''' 위에서 변수명을 바꾼 변수들과 모델링에 필요한 변수들만 추출하여 다시 데이터프레임에 덮어쓴다.'''
rawdata.index = range(0,len(rawdata))
''' 위의 정제과정으로 인덱스가 뒤죽박죽 되어있음을 알았고 다시 개수에맞게 순서대로 리인덱싱 해준다.'''


train = pd.concat([rawdata,reserved_data],axis=1)
train["전체_d12"].fillna(0, inplace=True)

def cut(data, day):
    ''' 예매 시작일을 며칠전으로 할 것인지 설정하고 나머지 feature는 제거'''
    resultlist = data.columns.values.copy()
    resultlist = list(resultlist)
    for i in range(0,len(data.columns.values)):
        for j in range(1,13-day):
            if (str(day+j) in data.columns.values[i])==1:
                resultlist.remove(data.columns.values[i])
    data = data.loc[:,resultlist]

    ''' 예매 시작일 당시 예매정보가 없는 행 제거
    index = []
    for i in range(0,len(data)):
        if data['전체_d'+str(day)][i]!=0:
            index.append(i)
    data = data.iloc[index,:]

     예매량이 0이라 nan으로 잡힌 부분 0으로 대체 '''
    for i in range(0,len(data.columns.values)):
        data.iloc[:,i] = data.iloc[:,i].fillna(0)
    
    ''' index 재설정'''
    data.index = range(0,len(data))

    return data

train = cut(train,12)


'''데이터 정제 과정 끝, 전체, 1루, 3루, 중앙, 외야에 맞게 각자 다른 모델을 생성'''


'''전체 '''
X = train.loc[:,'HT':'공휴일'] '''기본 정보만을 이용해 설명변수 지정'''
reserve = train.loc[:,'전체_d0':'전체_d12']  '''예매 데이터가 d- 형태로 되어있음'''
reserve = reserve.iloc[:,::-1]  '''예매 데이터를 반대로 해서 d+ 형태로 바꿔줌'''

X =  pd.concat([X,reserve],axis=1) '''기본 정보와 예매량 정보를 합치는 함수'''

X_train = X '''기본 정보와, 예매량 정보가 합쳐진 데이터'''
reserve_train = X_train.loc[:,'전체_d12':'전체_d0'] '''예매량 정보만 들어간 데이터'''

'''저장된 모델 불러오기 '''
from sklearn.externals import joblib

dict_y_models = {}              '''y(총관중수)를 예측하는 모델들이 들어가는 사전'''
dict_reserve_models = {}        '''예매량을 예측하는 모델들이 들어가는 사전'''

for i in range(0,14):
    y_model_key = '전체_y_model_' + str(i)   ''' y_model_n : n일 후 경기의 y를 예측하는 모델'''
    file_name = '/home/jihye/project_kt/model_parameter/전체_y_model_' + str(i)  ''' 경로 설정 필요'''
    dict_y_models[y_model_key] = joblib.load(file_name) '''dict_y_model에는 y를 예측하는 13개의 모델이 들어가있음'''
    
    for j in range(i,13):
        reserve_model_key = '전체_res_model_' + str(i) + '_' + str(j) '''reserve_model_n_m : n일 후 경기의 예매시작후m일차 예매량을 예측하는 모델'''
        file_name = '/home/jihye/project_kt/model_parameter/전체_res_model_' + str(i) + '_' + str(j)  ''' 경로 설정 필요'''
        dict_reserve_models[reserve_model_key] = joblib.load(file_name) '''dict_res_model에는 예매량을 예측하는 13+12+...+1=91개 모델이 들어가있음'''
        
res_start = 17 '''기본 정보 변수가 17개, 예매일에 대한 변수들이 17+1번째부터 존재함, 이후 사용될 예정'''
dic = {}
df = []
df = pd.DataFrame(df)

'''경기시작 n일 전에 예매를 오픈하면 k일 이후는 예매량 데이터가 존재하지 않음
아래에서는 d_n로 표시되며 n일 전까지의 예매량 데이터는 j=12-n개가 존재
즉, j개의 데이터를 통해 n+1개의 예매량 데이터와 y를 예측'''

for i in range(0,len(X_train)):
    pred_y = 0
    print('\n','경기일자', train['경기일자'][i])
    if(train['전체_d12'][i]==0):
        j = 0
    elif(train['전체_d11'][i]==0):
        j = 1
    elif(train['전체_d10'][i]==0):
        j = 2
    elif(train['전체_d9'][i]==0):
        j = 3
    elif(train['전체_d8'][i]==0):
        j = 4
    elif (train['전체_d7'][i]==0):
        j = 5
    elif (train['전체_d6'][i]==0):
        j = 6
    elif (train['전체_d5'][i]==0):
        j = 7
    elif (train['전체_d4'][i]==0):
        j = 8
    elif (train['전체_d3'][i]==0):
        j = 9
    elif (train['전체_d2'][i]==0):
        j = 10
    elif (train['전체_d1'][i]==0):
        j = 11
    elif (train['전체_d0'][i]==0):
        j = 12
    else :
        j = 13
    
    reserve_pred = -np.ones(j) '''이미 과거의 예매량은 -1로 표시(예측할 필요 없는 정보)'''
    
    features = X_train.iloc[i, 0:(res_start+j)].values.reshape(1,-1) '''res_start+j: 기본정보 변수와 j개의 예매량 데이터. 이들로 설명변수 feature를 구성'''
    for k in range(j,13):         
        reserve_model_key = '전체_res_model_' + str(j) + '_' + str(k) '''reserve_model_j_k : j일 후 경기의 예매시작후k일차 예매량을 예측하는 모델'''
        pred_res = dict_reserve_models[reserve_model_key].predict(features)'''pred_res : j일 후 경기의 예매시작후 k일차 예매량 예측값'''
        reserve_pred = np.append(reserve_pred, pred_res)'''j일후 경기의 예매량 예측값들의 집합'''
        print('다음날 예매량(예측)',pred_res)'''pred_res : j일 후 경기의 예매시작후 k일차 예매량 예측값'''
        
    y_model_key = '전체_y_model_' + str(j) '''y_model_j : j일 후 경기 관중수 예측하는 모델'''
    pred_y = dict_y_models[y_model_key].predict(features) '''j일 후 경기 관중수 예측값'''
    print('총관중수(예측)',pred_y)
    
    dic_key = 'vector' + str(i)
    dic[dic_key] = np.append(reserve_pred, pred_y) '''j일 후 경기 예매량들, 관중수 예측값'''
    
    
    df = pd.concat([df,pd.DataFrame(dic[dic_key])],axis=1) '''오늘의 미래의 모든 경기에 대한 예매량, 관중수 예측'''

'''행과 열 이름 붙혀주기'''
df = df.T 
df.index = train['경기일자']
colnames = ['d-12','d-11','d-10','d-9','d-8','d-7','d-6','d-5','d-4','d-3','d-2','d-1','d-0','총관중수']
df.columns = colnames

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)

for x in range(len(train.index)):
 sub_date = str(train['경기일자'].iloc[x])
 gyear = sub_date[0:4]
 gmonth = sub_date[5:7]
 gday = sub_date[8:10]
 game_date = gyear+gmonth+gday
 isWeekend = '0'
 if(train['월'].iloc[x]==1):
   gweek = '월'
 elif(train['화'].iloc[x]==1):
   gweek = '화'
 elif(train['수'].iloc[x]==1):
   gweek = '수'
 elif(train['목'].iloc[x]==1):
   gweek = '목'
 elif(train['금'].iloc[x]==1):
   gweek = '금'
 elif(train['토'].iloc[x]==1):
   gweek = '토'
   isWeekend = '1'
 elif(train['일'].iloc[x]==1):
   gweek = '일'
   isWeekend = '1'

 #공휴일
 isHoliday = '0'
 if(train['공휴일'].iloc[x]==1):
   isHoliday = '1'

 #차수
 chasu = x+1

 if(train['HT'].iloc[x]==1):
   visit_key = 'HT'
   visit = 'KIA'

 elif(train['LG'].iloc[x]==1):
   visit_key = 'LG'
   visit = 'LG'

 elif(train['NC'].iloc[x]==1):
   visit_key = 'NC'
   visit = 'NC'

 elif(train['SK'].iloc[x]==1):
   visit_key = 'SK'
   visit = 'SK'

 elif(train['WO'].iloc[x]==1):
   visit_key = 'WO'
   visit = '넥센'

 elif(train['OB'].iloc[x]==1):
   visit_key = 'OB'
   visit = '두산'

 elif(train['LT'].iloc[x]==1):
   visit_key = 'LT'
   visit = '롯데'

 elif(train['SS'].iloc[x]==1):
   visit_key = 'SS'
   visit = '삼성'

 elif(train['HH'].iloc[x]==1):
   visit_key = 'HH'
   visit = '한화'  

 zone_id = '0'
 zone_name = '전체'

 d12 = int(df['d-12'].iloc[x])
 d11 = int(df['d-11'].iloc[x])
 d10 = int(df['d-10'].iloc[x])
 d9 = int(df['d-9'].iloc[x])
 d8 = int(df['d-8'].iloc[x])
 d7 = int(df['d-7'].iloc[x])
 d6 = int(df['d-6'].iloc[x])
 d5 = int(df['d-5'].iloc[x])
 d4 = int(df['d-4'].iloc[x])
 d3 = int(df['d-3'].iloc[x])
 d2 = int(df['d-2'].iloc[x])
 d1 = int(df['d-1'].iloc[x])
 d0 = int(df['d-0'].iloc[x])

 total_ticket_cnt = d12 + d11 + d10 + d9 + d8 + d7 + d6 + d5 + d4 + d3 +d2+d1+d0
 total_ent_cnt = int(df['총관중수'].iloc[x])
 # SQL문 실행
 sql = ''' 
 INSERT INTO `tb_output_data`(`predic_date`, `game_date`,`gyear`,`gmonth`,`gday`,`gweek`,`isWeekend`,`isHoliday`,`chasu`,`visit`,`visit_key`,`zone_id`,`zone_name`,`d12`,`d11`,`d10`,`d9`,`d8`,`d7`,`d6`,`d5`,`d4`,`d3`,`d2`,`d1`,`d0`,`total_ticket_cnt`,`total_ent_cnt`,`reg_dt`,`upt_dt`)
 VALUES (date_format(now(),'%%Y%%m%%d'), %s,%s,%s,%s,%s,%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()); 
 '''
 # Connection 으로부터 Cursor 생성
 curs = conn.cursor()
 try:
    curs.execute(sql,(game_date,gyear,gmonth,gday,gweek,isWeekend,isHoliday,chasu,visit,visit_key,zone_id,zone_name,d12,d11,d10,d9,d8,d7,d6,d5,d4,d3,d2,d1,d0,total_ticket_cnt,total_ent_cnt))
 except Exception:
    print("Error with query: "+sql)
    sys.exit(1)

conn.commit()
#연결 종료
conn.close()






## 추가 내용 ##
#################### 1루 ####################
X = train.loc[:,'HT':'공휴일']
reserve = train.loc[:,'1루_d0':'1루_d12']  # 예매 데이터가 d- 형태로 되어있음
reserve = reserve.iloc[:,::-1]  # 예매 데이터를 반대로 해서 d+ 형태로 바꿔줌

X =  pd.concat([X,reserve],axis=1)

X_train = X
reserve_train = X_train.loc[:,'1루_d12':'1루_d0']

### 저장된 모델 불러오기 ###
from sklearn.externals import joblib

dict_y_models = {}              # dict of models for y(총관중수) prediction 
dict_reserve_models = {}        # dict of models for reservation prediction 

for i in range(0,14):
    y_model_key = '1루_y_model_' + str(i)   # y_model_n : n'th day prediction of y
    file_name = '/home/jihye/project_kt/model_parameter/1루_y_model_' + str(i)  # 경로 설정 필요
    dict_y_models[y_model_key] = joblib.load(file_name) 
    
    for j in range(i,13):
        reserve_model_key = '1루_res_model_' + str(i) + '_' + str(j)
        file_name = '/home/jihye/project_kt/model_parameter/1루_res_model_' + str(i) + '_' + str(j)  # 경로 설정 필요
        dict_reserve_models[reserve_model_key] = joblib.load(file_name)
        
res_start = 17
dic = {}
df = []
df = pd.DataFrame(df)

for i in range(0,len(X_train)):
    pred_y = 0
    print('\n','경기일자', train['경기일자'][i])
    if(train['전체_d12'][i]==0):
        j = 0
    elif(train['전체_d11'][i]==0):
        j = 1
    elif(train['전체_d10'][i]==0):
        j = 2
    elif(train['전체_d9'][i]==0):
        j = 3
    elif(train['전체_d8'][i]==0):
        j = 4
    elif (train['전체_d7'][i]==0):
        j = 5
    elif (train['전체_d6'][i]==0):
        j = 6
    elif (train['전체_d5'][i]==0):
        j = 7
    elif (train['전체_d4'][i]==0):
        j = 8
    elif (train['전체_d3'][i]==0):
        j = 9
    elif (train['전체_d2'][i]==0):
        j = 10
    elif (train['전체_d1'][i]==0):
        j = 11
    elif (train['전체_d0'][i]==0):
        j = 12
    else :
        j = 13
    
    reserve_pred = -np.ones(j)
    
    features = X_train.iloc[i, 0:(res_start+j)].values.reshape(1,-1)
    for k in range(j,13):         
        reserve_model_key = '1루_res_model_' + str(j) + '_' + str(k)
        pred_res = dict_reserve_models[reserve_model_key].predict(features)
        reserve_pred = np.append(reserve_pred, pred_res)
        print('다음날 예매량(예측)',pred_res)
        
    y_model_key = '1루_y_model_' + str(j)
    pred_y = dict_y_models[y_model_key].predict(features)
    print('총관중수(예측)',pred_y)
    
    dic_key = 'vector' + str(i)
    dic[dic_key] = np.append(reserve_pred, pred_y)
    
    
    df = pd.concat([df,pd.DataFrame(dic[dic_key])],axis=1)
    
df = df.T
df.index = train['경기일자']
colnames = ['d-12','d-11','d-10','d-9','d-8','d-7','d-6','d-5','d-4','d-3','d-2','d-1','d-0','총관중수']
df.columns = colnames

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)

for x in range(len(train.index)):
 sub_date = str(train['경기일자'].iloc[x])
 gyear = sub_date[0:4]
 gmonth = sub_date[5:7]
 gday = sub_date[8:10]
 game_date = gyear+gmonth+gday
 isWeekend = '0'
 if(train['월'].iloc[x]==1):
   gweek = '월'
 elif(train['화'].iloc[x]==1):
   gweek = '화'
 elif(train['수'].iloc[x]==1):
   gweek = '수'
 elif(train['목'].iloc[x]==1):
   gweek = '목'
 elif(train['금'].iloc[x]==1):
   gweek = '금'
 elif(train['토'].iloc[x]==1):
   gweek = '토'
   isWeekend = '1'
 elif(train['일'].iloc[x]==1):
   gweek = '일'
   isWeekend = '1'

 #공휴일
 isHoliday = '0'
 if(train['공휴일'].iloc[x]==1):
   isHOliday = '1'

 #차수
 chasu = x+1

 if(train['HT'].iloc[x]==1):
   visit_key = 'HT'
   visit = 'KIA'

 elif(train['LG'].iloc[x]==1):
   visit_key = 'LG'
   visit = 'LG'

 elif(train['NC'].iloc[x]==1):
   visit_key = 'NC'
   visit = 'NC'

 elif(train['SK'].iloc[x]==1):
   visit_key = 'SK'
   visit = 'SK'

 elif(train['WO'].iloc[x]==1):
   visit_key = 'WO'
   visit = '넥센'

 elif(train['OB'].iloc[x]==1):
   visit_key = 'OB'
   visit = '두산'

 elif(train['LT'].iloc[x]==1):
   visit_key = 'LT'
   visit = '롯데'

 elif(train['SS'].iloc[x]==1):
   visit_key = 'SS'
   visit = '삼성'

 elif(train['HH'].iloc[x]==1):
   visit_key = 'HH'
   visit = '한화'  

 zone_id = '1'
 zone_name = '1루'

 d12 = int(df['d-12'].iloc[x])
 d11 = int(df['d-11'].iloc[x])
 d10 = int(df['d-10'].iloc[x])
 d9 = int(df['d-9'].iloc[x])
 d8 = int(df['d-8'].iloc[x])
 d7 = int(df['d-7'].iloc[x])
 d6 = int(df['d-6'].iloc[x])
 d5 = int(df['d-5'].iloc[x])
 d4 = int(df['d-4'].iloc[x])
 d3 = int(df['d-3'].iloc[x])
 d2 = int(df['d-2'].iloc[x])
 d1 = int(df['d-1'].iloc[x])
 d0 = int(df['d-0'].iloc[x])

 total_ticket_cnt = d12 + d11 + d10 + d9 + d8 + d7 + d6 + d5 + d4 + d3 +d2+d1+d0
 total_ent_cnt = int(df['총관중수'].iloc[x])

 sql = ''' 
 INSERT INTO `tb_output_data`(`predic_date`, `game_date`,`gyear`,`gmonth`,`gday`,`gweek`,`isWeekend`,`isHoliday`,`chasu`,`visit`,`visit_key`,`zone_id`,`zone_name`,`d12`,`d11`,`d10`,`d9`,`d8`,`d7`,`d6`,`d5`,`d4`,`d3`,`d2`,`d1`,`d0`,`total_ticket_cnt`,`total_ent_cnt`,`reg_dt`,`upt_dt`)
 VALUES (date_format(now(),'%%Y%%m%%d'), %s,%s,%s,%s,%s,%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()); 
 '''
 # Connection 으로부터 Cursor 생성
 curs = conn.cursor()
 try:
    curs.execute(sql,(game_date,gyear,gmonth,gday,gweek,isWeekend,isHoliday,chasu,visit,visit_key,zone_id,zone_name,d12,d11,d10,d9,d8,d7,d6,d5,d4,d3,d2,d1,d0,total_ticket_cnt,total_ent_cnt))
 except Exception:
    print("Error with query: "+sql)
    sys.exit(1)

conn.commit()
#연결 종료
conn.close()
#################### 3루 ####################
X = train.loc[:,'HT':'공휴일']
reserve = train.loc[:,'3루_d0':'3루_d12']  # 예매 데이터가 d- 형태로 되어있음
reserve = reserve.iloc[:,::-1]  # 예매 데이터를 반대로 해서 d+ 형태로 바꿔줌

X =  pd.concat([X,reserve],axis=1)

X_train = X
reserve_train = X_train.loc[:,'3루_d12':'3루_d0']

### 저장된 모델 불러오기 ###
from sklearn.externals import joblib

dict_y_models = {}              # dict of models for y(총관중수) prediction 
dict_reserve_models = {}        # dict of models for reservation prediction 

for i in range(0,14):
    y_model_key = '3루_y_model_' + str(i)   # y_model_n : n'th day prediction of y
    file_name = '/home/jihye/project_kt/model_parameter/3루_y_model_' + str(i)  # 경로 설정 필요
    dict_y_models[y_model_key] = joblib.load(file_name) 
    
    for j in range(i,13):
        reserve_model_key = '3루_res_model_' + str(i) + '_' + str(j)
        file_name = '/home/jihye/project_kt/model_parameter/3루_res_model_' + str(i) + '_' + str(j)  # 경로 설정 필요
        dict_reserve_models[reserve_model_key] = joblib.load(file_name)
        
res_start = 17
dic = {}
df = []
df = pd.DataFrame(df)

for i in range(0,len(X_train)):
    pred_y = 0
    print('\n','경기일자', train['경기일자'][i])
    if(train['전체_d12'][i]==0):
        j = 0
    elif(train['전체_d11'][i]==0):
        j = 1
    elif(train['전체_d10'][i]==0):
        j = 2
    elif(train['전체_d9'][i]==0):
        j = 3
    elif(train['전체_d8'][i]==0):
        j = 4
    elif (train['전체_d7'][i]==0):
        j = 5
    elif (train['전체_d6'][i]==0):
        j = 6
    elif (train['전체_d5'][i]==0):
        j = 7
    elif (train['전체_d4'][i]==0):
        j = 8
    elif (train['전체_d3'][i]==0):
        j = 9
    elif (train['전체_d2'][i]==0):
        j = 10
    elif (train['전체_d1'][i]==0):
        j = 11
    elif (train['전체_d0'][i]==0):
        j = 12
    else :
        j = 13
    
    reserve_pred = -np.ones(j)
    
    features = X_train.iloc[i, 0:(res_start+j)].values.reshape(1,-1)
    for k in range(j,13):         
        reserve_model_key = '3루_res_model_' + str(j) + '_' + str(k)
        pred_res = dict_reserve_models[reserve_model_key].predict(features)
        reserve_pred = np.append(reserve_pred, pred_res)
        print('다음날 예매량(예측)',pred_res)
        
    y_model_key = '3루_y_model_' + str(j)
    pred_y = dict_y_models[y_model_key].predict(features)
    print('총관중수(예측)',pred_y)
    
    #vector = np.append(reserve_pred, pred_y)
    #df = np.vstack([df, vector])
    
    dic_key = 'vector' + str(i)
    dic[dic_key] = np.append(reserve_pred, pred_y)
    
    
    df = pd.concat([df,pd.DataFrame(dic[dic_key])],axis=1)
    
df = df.T
df.index = train['경기일자']
colnames = ['d-12','d-11','d-10','d-9','d-8','d-7','d-6','d-5','d-4','d-3','d-2','d-1','d-0','총관중수']
df.columns = colnames

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)

for x in range(len(train.index)):
 sub_date = str(train['경기일자'].iloc[x])
 gyear = sub_date[0:4]
 gmonth = sub_date[5:7]
 gday = sub_date[8:10]
 game_date = gyear+gmonth+gday
 isWeekend = '0'
 if(train['월'].iloc[x]==1):
   gweek = '월'
 elif(train['화'].iloc[x]==1):
   gweek = '화'
 elif(train['수'].iloc[x]==1):
   gweek = '수'
 elif(train['목'].iloc[x]==1):
   gweek = '목'
 elif(train['금'].iloc[x]==1):
   gweek = '금'
 elif(train['토'].iloc[x]==1):
   gweek = '토'
   isWeekend = '1'
 elif(train['일'].iloc[x]==1):
   gweek = '일'
   isWeekend = '1'

 #공휴일
 isHoliday = '0'
 if(train['공휴일'].iloc[x]==1):
   isHOliday = '1'

 #차수
 chasu = x+1

 if(train['HT'].iloc[x]==1):
   visit_key = 'HT'
   visit = 'KIA'

 elif(train['LG'].iloc[x]==1):
   visit_key = 'LG'
   visit = 'LG'

 elif(train['NC'].iloc[x]==1):
   visit_key = 'NC'
   visit = 'NC'

 elif(train['SK'].iloc[x]==1):
   visit_key = 'SK'
   visit = 'SK'

 elif(train['WO'].iloc[x]==1):
   visit_key = 'WO'
   visit = '넥센'

 elif(train['OB'].iloc[x]==1):
   visit_key = 'OB'
   visit = '두산'

 elif(train['LT'].iloc[x]==1):
   visit_key = 'LT'
   visit = '롯데'

 elif(train['SS'].iloc[x]==1):
   visit_key = 'SS'
   visit = '삼성'

 elif(train['HH'].iloc[x]==1):
   visit_key = 'HH'
   visit = '한화'  

 zone_id = '2'
 zone_name = '3루'

 d12 = int(df['d-12'].iloc[x])
 d11 = int(df['d-11'].iloc[x])
 d10 = int(df['d-10'].iloc[x])
 d9 = int(df['d-9'].iloc[x])
 d8 = int(df['d-8'].iloc[x])
 d7 = int(df['d-7'].iloc[x])
 d6 = int(df['d-6'].iloc[x])
 d5 = int(df['d-5'].iloc[x])
 d4 = int(df['d-4'].iloc[x])
 d3 = int(df['d-3'].iloc[x])
 d2 = int(df['d-2'].iloc[x])
 d1 = int(df['d-1'].iloc[x])
 d0 = int(df['d-0'].iloc[x])

 total_ticket_cnt = d12 + d11 + d10 + d9 + d8 + d7 + d6 + d5 + d4 + d3 +d2+d1+d0
 total_ent_cnt = int(df['총관중수'].iloc[x])

 sql = ''' 
 INSERT INTO `tb_output_data`(`predic_date`, `game_date`,`gyear`,`gmonth`,`gday`,`gweek`,`isWeekend`,`isHoliday`,`chasu`,`visit`,`visit_key`,`zone_id`,`zone_name`,`d12`,`d11`,`d10`,`d9`,`d8`,`d7`,`d6`,`d5`,`d4`,`d3`,`d2`,`d1`,`d0`,`total_ticket_cnt`,`total_ent_cnt`,`reg_dt`,`upt_dt`)
 VALUES (date_format(now(),'%%Y%%m%%d'), %s,%s,%s,%s,%s,%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()); 
 '''
 # Connection 으로부터 Cursor 생성
 curs = conn.cursor()
 try:
    curs.execute(sql,(game_date,gyear,gmonth,gday,gweek,isWeekend,isHoliday,chasu,visit,visit_key,zone_id,zone_name,d12,d11,d10,d9,d8,d7,d6,d5,d4,d3,d2,d1,d0,total_ticket_cnt,total_ent_cnt))
 except Exception:
    print("Error with query: "+sql)
    sys.exit(1)
    
conn.commit()
#연결 종료
conn.close()
#################### 중앙 ####################
X = train.loc[:,'HT':'공휴일']
reserve = train.loc[:,'중앙_d0':'중앙_d12']  # 예매 데이터가 d- 형태로 되어있음
reserve = reserve.iloc[:,::-1]  # 예매 데이터를 반대로 해서 d+ 형태로 바꿔줌

X =  pd.concat([X,reserve],axis=1)

X_train = X
reserve_train = X_train.loc[:,'중앙_d12':'중앙_d0']

### 저장된 모델 불러오기 ###
from sklearn.externals import joblib

dict_y_models = {}              # dict of models for y(총관중수) prediction 
dict_reserve_models = {}        # dict of models for reservation prediction 

for i in range(0,14):
    y_model_key = '중앙_y_model_' + str(i)   # y_model_n : n'th day prediction of y
    file_name = '/home/jihye/project_kt/model_parameter/중앙_y_model_' + str(i)  # 경로 설정 필요

    dict_y_models[y_model_key] = joblib.load(file_name) 
    
    for j in range(i,13):
        reserve_model_key = '중앙_res_model_' + str(i) + '_' + str(j)
        file_name = '/home/jihye/project_kt/model_parameter/중앙_res_model_' + str(i) + '_' + str(j)  # 경로 설정 필요
        dict_reserve_models[reserve_model_key] = joblib.load(file_name)
        
res_start = 17
dic = {}
df = []
df = pd.DataFrame(df)

for i in range(0,len(X_train)):
    pred_y = 0
    print('\n','경기일자', train['경기일자'][i])
    if(train['전체_d12'][i]==0):
        j = 0
    elif(train['전체_d11'][i]==0):
        j = 1
    elif(train['전체_d10'][i]==0):
        j = 2
    elif(train['전체_d9'][i]==0):
        j = 3
    elif(train['전체_d8'][i]==0):
        j = 4
    elif (train['전체_d7'][i]==0):
        j = 5
    elif (train['전체_d6'][i]==0):
        j = 6
    elif (train['전체_d5'][i]==0):
        j = 7
    elif (train['전체_d4'][i]==0):
        j = 8
    elif (train['전체_d3'][i]==0):
        j = 9
    elif (train['전체_d2'][i]==0):
        j = 10
    elif (train['전체_d1'][i]==0):
        j = 11
    elif (train['전체_d0'][i]==0):
        j = 12
    else :
        j = 13
    
    reserve_pred = -np.ones(j)
    
    features = X_train.iloc[i, 0:(res_start+j)].values.reshape(1,-1)
    for k in range(j,13):         
        reserve_model_key = '중앙_res_model_' + str(j) + '_' + str(k)
        pred_res = dict_reserve_models[reserve_model_key].predict(features)
        reserve_pred = np.append(reserve_pred, pred_res)
        print('다음날 예매량(예측)',pred_res)
        
    y_model_key = '중앙_y_model_' + str(j)
    pred_y = dict_y_models[y_model_key].predict(features)
    print('총관중수(예측)',pred_y)
    
    dic_key = 'vector' + str(i)
    dic[dic_key] = np.append(reserve_pred, pred_y)
    
    
    df = pd.concat([df,pd.DataFrame(dic[dic_key])],axis=1)
    
df = df.T
df.index = train['경기일자']
colnames = ['d-12','d-11','d-10','d-9','d-8','d-7','d-6','d-5','d-4','d-3','d-2','d-1','d-0','총관중수']
df.columns = colnames

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)

for x in range(len(train.index)):
 sub_date = str(train['경기일자'].iloc[x])
 gyear = sub_date[0:4]
 gmonth = sub_date[5:7]
 gday = sub_date[8:10]
 game_date = gyear+gmonth+gday
 isWeekend = '0'
 if(train['월'].iloc[x]==1):
   gweek = '월'
 elif(train['화'].iloc[x]==1):
   gweek = '화'
 elif(train['수'].iloc[x]==1):
   gweek = '수'
 elif(train['목'].iloc[x]==1):
   gweek = '목'
 elif(train['금'].iloc[x]==1):
   gweek = '금'
 elif(train['토'].iloc[x]==1):
   gweek = '토'
   isWeekend = '1'
 elif(train['일'].iloc[x]==1):
   gweek = '일'
   isWeekend = '1'

 #공휴일
 isHoliday = '0'
 if(train['공휴일'].iloc[x]==1):
   isHOliday = '1'

 #차수
 chasu = x+1

 if(train['HT'].iloc[x]==1):
   visit_key = 'HT'
   visit = 'KIA'

 elif(train['LG'].iloc[x]==1):
   visit_key = 'LG'
   visit = 'LG'

 elif(train['NC'].iloc[x]==1):
   visit_key = 'NC'
   visit = 'NC'

 elif(train['SK'].iloc[x]==1):
   visit_key = 'SK'
   visit = 'SK'

 elif(train['WO'].iloc[x]==1):
   visit_key = 'WO'
   visit = '넥센'

 elif(train['OB'].iloc[x]==1):
   visit_key = 'OB'
   visit = '두산'

 elif(train['LT'].iloc[x]==1):
   visit_key = 'LT'
   visit = '롯데'

 elif(train['SS'].iloc[x]==1):
   visit_key = 'SS'
   visit = '삼성'

 elif(train['HH'].iloc[x]==1):
   visit_key = 'HH'
   visit = '한화'  

 zone_id = '3'
 zone_name = '중앙'

 d12 = int(df['d-12'].iloc[x])
 d11 = int(df['d-11'].iloc[x])
 d10 = int(df['d-10'].iloc[x])
 d9 = int(df['d-9'].iloc[x])
 d8 = int(df['d-8'].iloc[x])
 d7 = int(df['d-7'].iloc[x])
 d6 = int(df['d-6'].iloc[x])
 d5 = int(df['d-5'].iloc[x])
 d4 = int(df['d-4'].iloc[x])
 d3 = int(df['d-3'].iloc[x])
 d2 = int(df['d-2'].iloc[x])
 d1 = int(df['d-1'].iloc[x])
 d0 = int(df['d-0'].iloc[x])

 total_ticket_cnt = d12 + d11 + d10 + d9 + d8 + d7 + d6 + d5 + d4 + d3 +d2+d1+d0
 total_ent_cnt = int(df['총관중수'].iloc[x])

 sql = ''' 
 INSERT INTO `tb_output_data`(`predic_date`, `game_date`,`gyear`,`gmonth`,`gday`,`gweek`,`isWeekend`,`isHoliday`,`chasu`,`visit`,`visit_key`,`zone_id`,`zone_name`,`d12`,`d11`,`d10`,`d9`,`d8`,`d7`,`d6`,`d5`,`d4`,`d3`,`d2`,`d1`,`d0`,`total_ticket_cnt`,`total_ent_cnt`,`reg_dt`,`upt_dt`)
 VALUES (date_format(now(),'%%Y%%m%%d'), %s,%s,%s,%s,%s,%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()); 
 '''

 # Connection 으로부터 Cursor 생성
 curs = conn.cursor()
 try:
    curs.execute(sql,(game_date,gyear,gmonth,gday,gweek,isWeekend,isHoliday,chasu,visit,visit_key,zone_id,zone_name,d12,d11,d10,d9,d8,d7,d6,d5,d4,d3,d2,d1,d0,total_ticket_cnt,total_ent_cnt))
 except Exception:
    print("Error with query: "+sql)
    sys.exit(1)
    
conn.commit()
#연결 종료
conn.close()
#################### 외야 ####################
X = train.loc[:,'HT':'공휴일']
reserve = train.loc[:,'외야_d0':'외야_d12']  # 예매 데이터가 d- 형태로 되어있음
reserve = reserve.iloc[:,::-1]  # 예매 데이터를 반대로 해서 d+ 형태로 바꿔줌

X =  pd.concat([X,reserve],axis=1)

X_train = X
reserve_train = X_train.loc[:,'외야_d12':'외야_d0']

### 저장된 모델 불러오기 ###
from sklearn.externals import joblib

dict_y_models = {}              # dict of models for y(총관중수) prediction 
dict_reserve_models = {}        # dict of models for reservation prediction 

for i in range(0,14):
    y_model_key = '외야_y_model_' + str(i)   # y_model_n : n'th day prediction of y
    file_name = '/home/jihye/project_kt/model_parameter/외야_y_model_' + str(i)  # 경로 설정 필요

    dict_y_models[y_model_key] = joblib.load(file_name) 
    
    for j in range(i,13):
        reserve_model_key = '외야_res_model_' + str(i) + '_' + str(j)
        file_name = '/home/jihye/project_kt/model_parameter/외야_res_model_' + str(i) + '_' + str(j)  # 경로 설정 필요
        dict_reserve_models[reserve_model_key] = joblib.load(file_name)
        
res_start = 17
dic = {}
df = []
df = pd.DataFrame(df)

for i in range(0,len(X_train)):
    pred_y = 0
    print('\n','경기일자', train['경기일자'][i])
    if(train['전체_d12'][i]==0):
        j = 0
    elif(train['전체_d11'][i]==0):
        j = 1
    elif(train['전체_d10'][i]==0):
        j = 2
    elif(train['전체_d9'][i]==0):
        j = 3
    elif(train['전체_d8'][i]==0):
        j = 4
    elif (train['전체_d7'][i]==0):
        j = 5
    elif (train['전체_d6'][i]==0):
        j = 6
    elif (train['전체_d5'][i]==0):
        j = 7
    elif (train['전체_d4'][i]==0):
        j = 8
    elif (train['전체_d3'][i]==0):
        j = 9
    elif (train['전체_d2'][i]==0):
        j = 10
    elif (train['전체_d1'][i]==0):
        j = 11
    elif (train['전체_d0'][i]==0):
        j = 12
    else :
        j = 13
    
    reserve_pred = -np.ones(j)
    
    features = X_train.iloc[i, 0:(res_start+j)].values.reshape(1,-1)
    for k in range(j,13):         
        reserve_model_key = '외야_res_model_' + str(j) + '_' + str(k)
        pred_res = dict_reserve_models[reserve_model_key].predict(features)
        reserve_pred = np.append(reserve_pred, pred_res)
        print('다음날 예매량(예측)',pred_res)
        
    y_model_key = '외야_y_model_' + str(j)
    pred_y = dict_y_models[y_model_key].predict(features)
    print('총관중수(예측)',pred_y)
    
    dic_key = 'vector' + str(i)
    dic[dic_key] = np.append(reserve_pred, pred_y)
    
    
    df = pd.concat([df,pd.DataFrame(dic[dic_key])],axis=1)
    
df = df.T
df.index = train['경기일자']
colnames = ['d-12','d-11','d-10','d-9','d-8','d-7','d-6','d-5','d-4','d-3','d-2','d-1','d-0','총관중수']
df.columns = colnames

try:
    # MySQL Connection 연결
    conn = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='predic_test_db'
    )
except Exception:
    print("Error in MySQL connection")
    sys.exit(1)

for x in range(len(train.index)):
 sub_date = str(train['경기일자'].iloc[x])
 gyear = sub_date[0:4]
 gmonth = sub_date[5:7]
 gday = sub_date[8:10]
 game_date = gyear+gmonth+gday
 isWeekend = '0'
 if(train['월'].iloc[x]==1):
   gweek = '월'
 elif(train['화'].iloc[x]==1):
   gweek = '화'
 elif(train['수'].iloc[x]==1):
   gweek = '수'
 elif(train['목'].iloc[x]==1):
   gweek = '목'
 elif(train['금'].iloc[x]==1):
   gweek = '금'
 elif(train['토'].iloc[x]==1):
   gweek = '토'
   isWeekend = '1'
 elif(train['일'].iloc[x]==1):
   gweek = '일'
   isWeekend = '1'

 #공휴일
 isHoliday = '0'
 if(train['공휴일'].iloc[x]==1):
   isHOliday = '1'

 #차수
 chasu = x+1

 if(train['HT'].iloc[x]==1):
   visit_key = 'HT'
   visit = 'KIA'

 elif(train['LG'].iloc[x]==1):
   visit_key = 'LG'
   visit = 'LG'

 elif(train['NC'].iloc[x]==1):
   visit_key = 'NC'
   visit = 'NC'

 elif(train['SK'].iloc[x]==1):
   visit_key = 'SK'
   visit = 'SK'

 elif(train['WO'].iloc[x]==1):
   visit_key = 'WO'
   visit = '넥센'

 elif(train['OB'].iloc[x]==1):
   visit_key = 'OB'
   visit = '두산'

 elif(train['LT'].iloc[x]==1):
   visit_key = 'LT'
   visit = '롯데'

 elif(train['SS'].iloc[x]==1):
   visit_key = 'SS'
   visit = '삼성'

 elif(train['HH'].iloc[x]==1):
   visit_key = 'HH'
   visit = '한화'  

 zone_id = '4'
 zone_name = '외야'

 d12 = int(df['d-12'].iloc[x])
 d11 = int(df['d-11'].iloc[x])
 d10 = int(df['d-10'].iloc[x])
 d9 = int(df['d-9'].iloc[x])
 d8 = int(df['d-8'].iloc[x])
 d7 = int(df['d-7'].iloc[x])
 d6 = int(df['d-6'].iloc[x])
 d5 = int(df['d-5'].iloc[x])
 d4 = int(df['d-4'].iloc[x])
 d3 = int(df['d-3'].iloc[x])
 d2 = int(df['d-2'].iloc[x])
 d1 = int(df['d-1'].iloc[x])
 d0 = int(df['d-0'].iloc[x])

 total_ticket_cnt = d12 + d11 + d10 + d9 + d8 + d7 + d6 + d5 + d4 + d3 +d2+d1+d0
 total_ent_cnt = int(df['총관중수'].iloc[x])

 sql = ''' 
 INSERT INTO `tb_output_data`(`predic_date`, `game_date`,`gyear`,`gmonth`,`gday`,`gweek`,`isWeekend`,`isHoliday`,`chasu`,`visit`,`visit_key`,`zone_id`,`zone_name`,`d12`,`d11`,`d10`,`d9`,`d8`,`d7`,`d6`,`d5`,`d4`,`d3`,`d2`,`d1`,`d0`,`total_ticket_cnt`,`total_ent_cnt`,`reg_dt`,`upt_dt`)
 VALUES (date_format(now(),'%%Y%%m%%d'), %s,%s,%s,%s,%s,%s,%s, %s, %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,now(),now()); 
 '''

 # Connection 으로부터 Cursor 생성
 curs = conn.cursor()
 try:
    curs.execute(sql,(game_date,gyear,gmonth,gday,gweek,isWeekend,isHoliday,chasu,visit,visit_key,zone_id,zone_name,d12,d11,d10,d9,d8,d7,d6,d5,d4,d3,d2,d1,d0,total_ticket_cnt,total_ent_cnt))
 except Exception:
    print("Error with query: "+sql)
    sys.exit(1)
    
conn.commit()
#연결 종료
conn.close()

print("fin")
