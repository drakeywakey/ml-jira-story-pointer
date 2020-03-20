from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

def get_model(df):
    X_train, X_test, y_train, y_test = train_test_split(df['features'], df['story_points'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    model = RandomForestClassifier(random_state=27).fit(X_train_tfidf, y_train)

    ticklabels=sorted(df['story_points'].unique())
    y_pred = model.predict(count_vect.transform(X_test))
    conf_mat = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots(figsize=(10,10))
    # sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=ticklabels, yticklabels=ticklabels)
    # plt.ylabel('Actual')
    # plt.xlabel('Predicted')
    print(conf_mat)
    print(accuracy_score(y_test, y_pred))
    print(precision_score(y_test, y_pred, average=None))
    print(recall_score(y_test, y_pred, average=None))
    print(f1_score(y_test, y_pred, average=None))

    # plt.show()

    return model, count_vect