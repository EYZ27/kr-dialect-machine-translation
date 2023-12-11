import pandas as pd
import argparse
import json
import os
import re
from sklearn.model_selection import train_test_split


label = {'충' : 0, '전' :1, '경' :2, '제' : 3}

parser = argparse.ArgumentParser(description='Transformer dialect machine translation')
parser.add_argument('--data', default='/nas/home/sungchul/dia/chung_train/val',type=str,
                    help='path to dataset directory')



def preprocess(path):
    file_list = os.listdir(path)

    file_list = [file for file in file_list if file.endswith('json')]
    file_list
    
    li = []
    li2 = []
    lb = []
    for i in file_list :
        # print(i)
        with open(os.path.join(path,i), 'rb') as f: 
            df = json.load(f)
        lb_info = df['metadata']['category'].replace('솔트룩스','').strip()
        try:
            tmp = label[lb_info[:1]]
        except KeyError:
            print('데이터 id :', df['id'],'에서 지역을 찾는 데 문제가 발생했습니다.')
            continue
        for i in df['utterance']:
            lb.append(label[lb_info[:1]])
            li.append(i['standard_form'])
            li2.append(i['dialect_form'])
        # kk = [i['standard_form'] for i in df['utterance']]     
        # kk2 = [i['dialect_form'] for i in df['utterance']]
        # for i in kk:
        #     li.append(i)
        # for j in kk2:
        #     li2.append(j)
        

    df = pd.DataFrame({'standard_form' : li,
                       'dialect_form' : li2,
                       'label' : lb})

    df2 = df[df['standard_form'] != df['dialect_form']]

    filter_li = []
    for i in df2.index : 
        if len(set(df['standard_form'][i].split(' ')) - set(df['dialect_form'][i].split(' '))) > 1 : 
            filter_li.append(i)
                
    df2 = df2.loc[filter_li]
    df2 = df2.reset_index(drop=True)
    
    hangul = re.compile('[^ ㄱ-ㅣ가-힣+]')
    df2['standard_form'] = df2['standard_form'].apply(lambda x : hangul.sub('',x))
    df2['dialect_form'] = df2['dialect_form'].apply(lambda x : hangul.sub('',x))

    special = re.compile(r'[^ A-Za-z0-9가-힣+]')
    df2['standard_form'] = df2['standard_form'].apply(lambda x : special.sub('',x))
    df2['dialect_form'] = df2['dialect_form'].apply(lambda x : special.sub('',x))

    y_temp = df2['label']

    train, test, y_temp, y_test = train_test_split(df2, y_temp, test_size=0.1, random_state=42, stratify=y_temp)
    train, val, tmp, y_test = train_test_split(train, y_temp, test_size=0.1, random_state=42, stratify=y_temp)

    return train, val, test

if __name__ == '__main__':
    args = parser.parse_args() 
    path = args.data

    train, val, test = preprocess(path)    

    train.to_csv('train.csv')
    val.to_csv('val.csv')
    test.to_csv('test.csv')
