from jira import JIRA
import os
import pandas as pd
import numpy as np
from sklearn.utils import resample
from ml_service import get_model

options = {
    'server': 'https://itcentral.chgcompanies.com/jira'
}

jira = JIRA(options, basic_auth=(os.getenv('JIRA_USER'), os.getenv('JIRA_PASS')))

# customfield_10284 = epic
# customfield_11281 = user story
# customfield_11282 = acceptance criteria
# customfield_10182 = story points

query = jira.search_issues('project = LTC AND issuetype = Story AND "Story Points" != null AND "Story Points" > 0 AND "Story Points" <= 13 AND createdDate < 2020-01-01',
                             json_result=True,
                             maxResults=1000,
                             fields='summary,description,customfield_10284,customfield_11281,customfield_11282,labels,customfield_10182')

issues = {issue['key']: issue['fields'] for issue in query['issues']}

df = pd.DataFrame.from_dict(issues, orient='index')

df.replace(np.nan, '', inplace=True)

df.rename(columns={
    'customfield_10284': 'epic', 
    'customfield_11281': 'user_story', 
    'customfield_11282': 'acceptance_criteria',
    'customfield_10182': 'story_points'}, inplace=True)

df['story_points'] = df['story_points'].astype(int)

# resample minority classes to match three's 179 records
minority = [0, 5, 8, 13]
df_majority = df[(df['story_points'] == 1) | (df['story_points'] == 2) | (df['story_points'] == 3)]

for story_point in minority:
    df_minority = df[df['story_points'] == story_point]
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=179,
                                     random_state=27)
    df_majority = pd.concat([df_majority, df_minority_upsampled])

df = df_majority

# fig = plt.figure(figsize=(8,6))
# df.groupby('story_points')['summary'].count().plot.bar(ylim=0)
# plt.show()

df['features'] = df['summary'] + ' ' + df['description'] + ' ' + \
                df['user_story'] + ' ' + df['acceptance_criteria'] + \
                df['epic']

model, count_vect = get_model(df)

def predict(ticket):
    issue = jira.issue(ticket)
    summary = issue.fields.summary
    pred = model.predict(count_vect.transform([summary]))
    return f'{pred[0]}'