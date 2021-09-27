#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:48:06 2021

@author: Xintong Zhao
"""

import pandas as pd
import re
import os
import requests
from config import MY_API_KEYS
from datetime import datetime
import string
import time

'''
1. Full-text collection from ScienceDirect
If you have any questions regarding to the full text collection script below, please contact xintong.zhao@drexel.edu

Note: in order to use Elsevier APIs (ScienceDirect, Scopus, ...), you should have registered an API account at Elsevier Developer Portal and your
institution should subscribed some full text resources (e.g., journal).

Option 1: If you download the citation information from the Science Direct website, then go with the following option 1.
'''
meta_folder = 'name a folder to save meta data here'
# set directory
print('Getting directory...')
cwd = os.getcwd()
dir_meta = os.path.join(cwd, meta_folder)
dir_corpus = os.path.join(cwd, 'corpus')

# load the api key from config file
api_idx = 0
my_api_key = MY_API_KEYS[api_idx]

# if you download metafile manually from ScienceDirect website, then go with follows
def meta_data_processing(meta_directory, if_save=True):
    # meta file processing
    print("Processing meta data...")
    target_dois = []
    corresponding_titles = []
    # we check each folder under the meta-file directory
    for folder_file in os.listdir(meta_directory):
        if '.txt' in folder_file:
            with open(os.path.join(meta_directory, folder_file), 'r') as meta_ref:
                # read the text content of each meta file
                meta_data = meta_ref.read()
            # split the text into individual records
            meta_records = meta_data.split('\n\n')
            for meta_record in meta_records:
                # split each individual record to detailed info
                meta_record = meta_record.split('\n')
                # we record the title and doi number for download
                for sub_record in meta_record:
                    if 'https://doi.org' in sub_record:
                        # add the doi number to the download list
                        target_dois += [sub_record]
                        # since title is the second line of each record
                        corresponding_titles += [meta_record[1]]
    df_integrated_meta = pd.DataFrame(columns=['doi', 'title'])
    df_integrated_meta['doi'] = target_dois
    df_integrated_meta['title'] = corresponding_titles
    if if_save:
        df_integrated_meta.to_csv('{}.csv'.format(meta_folder), index=False)
    return df_integrated_meta


df_meta = meta_data_processing(dir_meta, True)

# check previously downloaded literature
downloaded_articles = []
for file in os.listdir(dir_corpus):
    if '.xml' in file:
        downloaded_articles += [file.replace('.xml', '')]

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print('Start downloading at {}'.format(dt_string))
count = 0
target_doi = list(df_meta['doi'])
# remove previously downloaded articles from download queue
target_doi = list(set(target_doi)-set(downloaded_articles))
# collecting full articles
for idx in range(len(target_doi)):
    if count % 200 == 0 and count != 0:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        print('{} Full articles have been scanned at {}'.format(count, dt_string))
    article_name = target_doi[idx].translate(str.maketrans('', '', string.punctuation))
    try:
        now = datetime.now()
        article_request = requests.get('https://api.elsevier.com/content/article/doi/' + target_doi[idx],
                                       params={'Accept': 'text/xml', 'apiKey': my_api_key})
        time.sleep(0.1)  # API Quota: 10 request/ second
    except:
        count += 1
        continue

    with open('./corpus/{}.xml'.format(article_name), 'wb') as f:
        f.write(article_request.content)
    count += 1


'''
Option 2: if you want to use keyword to automatically retrieve articles from Science Direct, then go with the following option 2.
Note: Be aware of your API quota.
'''
operation_count = 0

# query all results including un-subscribed contents
# API quota for ScienceDirect Search V2 : 20000/week + 2 requests/sec
# More details at dev.elsevier.com/api_key_settings.html
start_year = 2022
# we collect every subscribed articles from 2000 to now
while start_year >= 2000:
    query_dois = []
    query_piis = []
    query_titles = []
    query_resp = requests.get('https://api.elsevier.com/content/search/sciencedirect',
                              params={'Accept': 'application/json', 'apiKey': my_api_key, 'query': '{type your keyword here}',
                                      'date': str(start_year), 'count': 100, 'subs':'true'})

    query_json = query_resp.json()
    total_num_articles = int(query_json['search-results']['opensearch:totalResults'])
    try:
        batch_records = query_json['search-results']['entry']  # which contains a batch of #count articles metadata
    except:
        print('{} have been processed - Year {}.'.format(operation_count * 100, start_year))
        operation_count = 0
        start_year -= 1
        with open('{}_meta.txt'.format(str(start_year)), 'w', encoding='utf-8') as f:
            f.write('\n'.join(
                [query_dois[i] + '\t' + query_titles[i] + '\t' + query_piis[i] for i in range(len(query_dois))]))
        continue
    for entry in batch_records:
        try:
            query_dois += [entry['dc:identifier'].replace('DOI:', '')]
        except:
            query_dois += ['None']
        try:
            query_titles += [entry['dc:title']]
        except:
            query_titles += ['None']
        try:
            query_piis += [entry['pii']]
        except:
            query_piis += ['None']
    operation_count += 1
    total_num_articles -= 100
    time.sleep(0.7)  # to avoid quota exceed
    while total_num_articles > 0:
        # if the start value is greater than 6000, we stop collecting this year's articles
        # because 6000 is the Science Direct API system global maximum value, you will not be able to retrieve articles after 6000
        if operation_count * 100 >= 6000:
            print('{} have been processed - Year {}.'.format(operation_count * 100, start_year))
            start_year -= 1
            operation_count = 0
            with open('{}_meta.txt'.format(str(start_year)), 'w', encoding='utf-8') as f:
                f.write('\n'.join(
                    [query_dois[i] + '\t' + query_titles[i] + '\t' + query_piis[i] for i in range(len(query_dois))]))
            break
        query_resp = requests.get('https://api.elsevier.com/content/search/sciencedirect',
                                  params={'Accept': 'application/json', 'apiKey': my_api_key,
                                          'query': '{amorphous alloy}',
                                          'start': operation_count * 100, 'date': str(start_year), 'count': 100,
                                          'subs': 'true'})
        query_json = query_resp.json()
        try:
            batch_records = query_json['search-results']['entry']  # which contains a batch of #count articles metadata
        except:
            print('{} have been processed - Year {}.'.format(operation_count * 100, start_year))
            start_year -= 1
            operation_count = 0
            with open('{}_meta.txt'.format(str(start_year)), 'w', encoding='utf-8') as f:
                f.write('\n'.join(
                    [query_dois[i] + '\t' + query_titles[i] + '\t' + query_piis[i] for i in range(len(query_dois))]))
        for entry in batch_records:
            try:
                query_dois += [entry['dc:identifier'].replace('DOI:', '')]
            except:
                query_dois += ['None']
            try:
                query_titles += [entry['dc:title']]
            except:
                query_titles += ['None']
            try:
                query_piis += [entry['pii']]
            except:
                query_piis += ['None']

        operation_count += 1
        total_num_articles -= 100
        time.sleep(0.7)  # to avoid quota exceed
        # if the total number of articles from the current year is done retrieving, we continue to the next year
        # record article identifiers in a txt file
        if total_num_articles <= 0:
            print('{} have been processed - Year {}.'.format(operation_count * 100, start_year))
            start_year -= 1
            operation_count = 0
            with open('{}_meta.txt'.format(str(start_year)), 'w', encoding='utf-8') as f:
                f.write('\n'.join(
                    [query_dois[i] + '\t' + query_titles[i] + '\t' + query_piis[i] for i in range(len(query_dois))]))

# create a dataframe to store the metafiles
df_meta = pd.DataFrame(columns=['title', 'doi', 'pii'])
meta_files = os.listdir(os.path.join(dir_meta, 'amorphous_alloy'))
for meta_file in meta_files:
    with open(os.path.join(os.path.join(dir_meta, 'amorphous_alloy', meta_file)), 'r', encoding='utf-8') as f:
        content = f.read()
    rows_content = content.split('\n')
    for row in rows_content:
        row_list = row.split('\t')
        try:
            article_title = row_list[1]
        except:
            article_title = 'None'
        try:
            article_doi = row_list[0]
        except:
            article_doi = 'None'
        try:
            article_pii = row_list[2]
        except:
            article_pii = 'None'
        temp_df = {'title': article_title, 'doi': article_doi, 'pii': article_pii}
        df = df_meta.append(temp_df, ignore_index=True)

df_meta = df_meta.drop_duplicates(subset=['doi'], keep='first')
df_meta = df_meta[df_meta['doi'].str.contains("None") == False]
df_meta.to_csv('query_result{}.csv'.format(datetime.now()), index=False)

target_doi = [i.replace('https://doi.org/', '') for i in list(df_meta['doi'])]
titles = list(df_meta['title'])


'''
For using Scopus API to perform abstract retrieval, see below
Before using this script, you will need to get a list of eids (a type of unique identifier of abstracts) for target articles

'''
list_eids = []
list_abstracts = []
count = 0
idx_api = 0
for eid in eids:
    # for every 2000 articles, we save the progress
    if count % 2000 == 0 and count != 0:
        data = pd.DataFrame(columns=['eid', 'abstract'])
        data['eid'] = list_eids
        data['abstract'] = list_abstracts

        data.to_csv('the name of your csv file%d.csv' % (count), index=False)

    try:
        # here we send request to Scopus
        # be aware of API quota
        resp = requests.get("https://api.elsevier.com/content/abstract/eid/" + eid,
                            headers={'Accept': 'application/json',
                                     'X-ELS-APIKey': my_api_key})

        abstract = resp.json()['abstracts-retrieval-response']['coredata']['dc:description']
        list_eids += [eid]
        list_abstracts += [abstract]
        time.sleep(0.7)
    except:
        time.sleep(0.7)
        continue

    count += 1
    if count % 100 == 0 and count != 0:
        print("%d have been collected." % (count))

output = pd.DataFrame(columns=['eid', 'abstract'])
output['eid'] = list_eids
output['abstract'] = list_abstracts
output.to_csv('the name of your file.csv', index=False)



