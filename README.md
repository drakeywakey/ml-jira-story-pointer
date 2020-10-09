# ml-jira-story-pointer

## CHG Hackathon March 2020

### Motivation
Our team had a large JIRA backlog of very similar stories that tended to follow certain patterns. For example, when we saw the words "Specialty update" in the title, we almost always pointed the story as a 5. We thought it would be interesting to try using machine learning to help us point future stories in our backlog based on previous story titles and descriptions.
### Results
- 68.55% accuracy
### Learnings
- Using a dataset with heavily imbalanced class representations can very easily throw your predictor off
- Learned how the Multinomial Bayes classifier works and the math behind it
- Learned that ensemble methods like XGBoost or neural networks using the likes of Tensorflow or Pytorch would likely outperform this classifier.
### Built with
- Python
- pandas
- scikit-learn
