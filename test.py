from jira import JIRA
import os
import pandas as pd

options = {
    'server': 'https://itcentral.chgcompanies.com/jira'
}

jira = JIRA(options, basic_auth=('dbennion', os.environ['JIRA_PASS']))

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