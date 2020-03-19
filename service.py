from dotenv import load_dotenv
from jira import JIRA
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns


load_dotenv()

options = {
    'server': 'https://itcentral.chgcompanies.com/jira'
}

jira = JIRA(options, basic_auth=(os.getenv('JIRA_USER'), os.getenv('JIRA_PASS')))

# customfield_10284 = epic
# customfield_11281 = user story
# customfield_11282 = acceptance criteria
# customfield_10182 = story points

query = jira.search_issues('project = LTC AND issuetype = Story AND "Story Points" != null AND "Story Points" > 0  AND createdDate < 2020-01-01',
                             json_result=True,
                             maxResults=1000,
                             fields='summary,description,customfield_10284,customfield_11281,customfield_11282,labels,customfield_10182')

issues = {issue['key']: issue['fields'] for issue in query['issues']}

df = pd.DataFrame.from_dict(issues, orient='index')

df.replace('', inplace=True)

df.rename(columns={
    'customfield_10284': 'epic', 
    'customfield_11281': 'user_story', 
    'customfield_11282': 'acceptance_criteria',
    'customfield_10182': 'story_points'}, inplace=True)

df['story_points'] = df['story_points'].astype(int)
 
fig = plt.figure(figsize=(8,6))
df.groupby('story_points')['summary'].count().plot.bar(ylim=0)
# plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df['summary']).toarray()
labels = df['story_points']
# print(tfidf.get_feature_names())

X_train, X_test, y_train, y_test = train_test_split(df['summary'], df['story_points'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
model = MultinomialNB().fit(X_train_tfidf, y_train)

# y_pred = model.predict(count_vect.transform(X_test))
# conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()
# print(conf_mat)

def predict(ticket):
    issue = jira.issue(ticket)
    summary = issue.fields.summary
    pred = model.predict(count_vect.transform([summary]))
    return f'{pred[0]}'
