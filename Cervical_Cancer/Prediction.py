import pandas as p
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

df = p.read_csv('risk_factors_cervical_cancer.csv', na_values='?')
print df
print df.columns.values

hinselmann_cancer = df['Hinselmann']
schiller_cancer = df['Schiller']
citology_cancer = df['Citology']
biopsy_cancer = df['Biopsy']
all_cancer = hinselmann_cancer | schiller_cancer | citology_cancer | biopsy_cancer
print all_cancer
df['AllCancerFlag'] = all_cancer

print df.describe().T

# STDs:cervical condylomatosis and STDs:AIDS have min and max = 0, so they are not representative
df.drop(['STDs:cervical condylomatosis', 'STDs:AIDS'], axis=1, inplace=True)
print df.columns

# TODO: Calculate number of patients
n_patients = df.shape[0]

# TODO: Calculate number of features
n_features = df.shape[1] - 5

# TODO: Calculate patients, having cervical cancer
n_has_cancer = len(df[df['AllCancerFlag'] == 1])

# TODO: Calculate patients, not having cervical cancer
n_has_no_cancer = len(df[df['AllCancerFlag'] == 0])

# TODO: Calculate cancer rate
cancer_rate = n_has_cancer / float(n_has_no_cancer) * 100

# Print the results
print "Total number of patients: {}".format(n_patients)
print "Number of features: {}".format(n_features)
print "Number of patients, having cervical cancer: {}".format(n_has_cancer)
print "Number of patients, not having cervical cancer: {}".format(n_has_no_cancer)
print "Cancer rate: {:.2f}%".format(cancer_rate)

# Extract feature columns
feature_cols = list(df.columns[:-5])

# Extract target column
target_col = df.columns[-1]

# Show the list of columns
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

df_wo_null = df.dropna(axis=0, how='any')
print df_wo_null.shape[0]

print "\nNaN sum:"
print df.isnull().sum()

print "\nInfo:"
print df.info()

#Fill N/A with median values for continious data and 0/1 for categorical data
df['Number of sexual partners'] = df['Number of sexual partners'].fillna(df['Number of sexual partners'].median())
df['First sexual intercourse'] = df['First sexual intercourse'].fillna(df['First sexual intercourse'].median())
df['Num of pregnancies'] = df['Num of pregnancies'].fillna(df['Num of pregnancies'].median())
df['Smokes'] = df['Smokes'].fillna(1)
df['Smokes (years)'] = df['Smokes (years)'].fillna(df['Smokes (years)'].median())
df['Smokes (packs/year)'] = df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].median())
df['Hormonal Contraceptives'] = df['Hormonal Contraceptives'].fillna(1)
df['Hormonal Contraceptives (years)'] = df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].median())
df['IUD'] = df['IUD'].fillna(1)
df['IUD (years)'] = df['IUD (years)'].fillna(df['IUD (years)'].median())
df['STDs'] = df['STDs'].fillna(1)
df['STDs (number)'] = df['STDs (number)'].fillna(df['STDs (number)'].median())
df['STDs:condylomatosis'] = df['STDs:condylomatosis'].fillna(1)
df['STDs:vaginal condylomatosis'] = df['STDs:vaginal condylomatosis'].fillna(1)
df['STDs:vulvo-perineal condylomatosis'] = df['STDs:vulvo-perineal condylomatosis'].fillna(1)
df['STDs:syphilis'] = df['STDs:syphilis'].fillna(1)
df['STDs:pelvic inflammatory disease'] = df['STDs:pelvic inflammatory disease'].fillna(1)
df['STDs:genital herpes'] = df['STDs:genital herpes'].fillna(1)
df['STDs:molluscum contagiosum'] = df['STDs:molluscum contagiosum'].fillna(1)
df['STDs:HIV'] = df['STDs:HIV'].fillna(1)
df['STDs:Hepatitis B'] = df['STDs:Hepatitis B'].fillna(1)
df['STDs:HPV'] = df['STDs:HPV'].fillna(1)
df['STDs: Time since first diagnosis'] = df['STDs: Time since first diagnosis'].fillna(df['STDs: Time since first diagnosis'].median())
df['STDs: Time since last diagnosis'] = df['STDs: Time since last diagnosis'].fillna(df['STDs: Time since last diagnosis'].median())


# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = df[feature_cols]
y_all = df[target_col]

# Show the feature information by printing the first five rows
print "\nFeature values:"
print X_all.head()


num_train = 645
num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, train_size=num_train, random_state=1)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])

print "\n\nTraining set:"
print  X_train
print "\n\nTraining labels:"
print  y_train
print "\n\nTest set:"
print  X_test
print "\n\nTest label:"
print  y_test
print "\n\nTraining sum of 1:"
print  y_train.sum()
print "\n\nTest sum of 1:"
print  y_test.sum()


clf = DecisionTreeClassifier(random_state=1)
clf.fit(X_train, y_train)

print 'DecisionTreeClassifier original:'
print 'Training set:'
print metrics.classification_report(y_train.values, clf.predict(X_train))
print metrics.confusion_matrix(y_train.values, clf.predict(X_train))
print 'Testing set:'
print metrics.classification_report(y_test.values, clf.predict(X_test))
print metrics.confusion_matrix(y_test.values, clf.predict(X_test))


clf_extratree = ExtraTreesClassifier(n_estimators=100, random_state=1)
clf_extratree.fit(X_train, y_train)
print 'ExtraTreesClassifier original:'
print 'Training set:'
print metrics.classification_report(y_train.values, clf_extratree.predict(X_train))
print metrics.confusion_matrix(y_train.values, clf_extratree.predict(X_train))
print 'Testing set:'
print metrics.classification_report(y_test.values, clf_extratree.predict(X_test))
print metrics.confusion_matrix(y_test.values, clf_extratree.predict(X_test))

# display the relative importance of each attribute
print clf_extratree.feature_importances_
print 'Features importance:'
print zip(X_all.columns, clf_extratree.feature_importances_)

df_important_features = p.DataFrame()
for idx, value in enumerate(clf_extratree.feature_importances_):
    if value >= 0.1:
        print X_all.columns[idx]
        df_important_features[X_all.columns[idx]] = X_all[X_all.columns[idx]]
df_important_features['AllCancerFlag'] = df['AllCancerFlag']

feature_cols_important = list(df_important_features.columns[:-1])
print feature_cols_important
X_all2 = df_important_features[feature_cols_important]

clf2 = DecisionTreeClassifier(random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all2, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf2.fit(X_train2, y_train2)
print 'DecisionTreeClassifier first features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train2.values, clf2.predict(X_train2))
print metrics.confusion_matrix(y_train2.values, clf2.predict(X_train2))
print 'Testing set:'
print metrics.classification_report(y_test2.values, clf2.predict(X_test2))
print metrics.confusion_matrix(y_test2.values, clf2.predict(X_test2))

clf2_extra = ExtraTreesClassifier(n_estimators=100, random_state=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all2, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf2_extra.fit(X_train2, y_train2)
print 'ExtraTreesClassifier first features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train2.values, clf2_extra.predict(X_train2))
print metrics.confusion_matrix(y_train2.values, clf2_extra.predict(X_train2))
print 'Testing set:'
print metrics.classification_report(y_test2.values, clf2_extra.predict(X_test2))
print metrics.confusion_matrix(y_test2.values, clf2_extra.predict(X_test2))




df_important_features2 = p.DataFrame()
for idx, value in enumerate(clf_extratree.feature_importances_):
    if value >= 0.01:
        print X_all.columns[idx]
        df_important_features2[X_all.columns[idx]] = X_all[X_all.columns[idx]]
df_important_features2['AllCancerFlag'] = df['AllCancerFlag']

feature_cols_important2 = list(df_important_features2.columns[:-1])
print feature_cols_important2
X_all3 = df_important_features2[feature_cols_important2]

clf3 = DecisionTreeClassifier(random_state=1)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_all3, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf3.fit(X_train3, y_train3)
print 'DecisionTreeClassifier second features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train3.values, clf3.predict(X_train3))
print metrics.confusion_matrix(y_train3.values, clf3.predict(X_train3))
print 'Testing set:'
print metrics.classification_report(y_test3.values, clf3.predict(X_test3))
print metrics.confusion_matrix(y_test3.values, clf3.predict(X_test3))

clf3_extra = ExtraTreesClassifier(n_estimators=100, random_state=1)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X_all3, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf3_extra.fit(X_train3, y_train3)
print 'ExtraTreesClassifier second features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train3.values, clf3_extra.predict(X_train3))
print metrics.confusion_matrix(y_train3.values, clf3_extra.predict(X_train3))
print 'Testing set:'
print metrics.classification_report(y_test3.values, clf3_extra.predict(X_test3))
print metrics.confusion_matrix(y_test3.values, clf3_extra.predict(X_test3))





df_important_features4 = p.DataFrame()
for idx, value in enumerate(clf_extratree.feature_importances_):
    if value >= 0.005:
        print X_all.columns[idx]
        df_important_features4[X_all.columns[idx]] = X_all[X_all.columns[idx]]
df_important_features4['AllCancerFlag'] = df['AllCancerFlag']

feature_cols_important4 = list(df_important_features4.columns[:-1])
print feature_cols_important4
X_all5 = df_important_features4[feature_cols_important4]

clf5 = DecisionTreeClassifier(random_state=1)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_all5, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf5.fit(X_train5, y_train5)
print 'DecisionTreeClassifier third features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train5.values, clf5.predict(X_train5))
print metrics.confusion_matrix(y_train5.values, clf5.predict(X_train5))
print 'Testing set:'
print metrics.classification_report(y_test5.values, clf5.predict(X_test5))
print metrics.confusion_matrix(y_test5.values, clf5.predict(X_test5))


clf5_extra = ExtraTreesClassifier(n_estimators=100, random_state=1)
X_train5, X_test5, y_train5, y_test5 = train_test_split(X_all5, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf5_extra.fit(X_train5, y_train5)
print 'ExtraTreesClassifier third features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train5.values, clf5_extra.predict(X_train5))
print metrics.confusion_matrix(y_train5.values, clf5_extra.predict(X_train5))
print 'Testing set:'
print metrics.classification_report(y_test5.values, clf5_extra.predict(X_test5))
print metrics.confusion_matrix(y_test5.values, clf5_extra.predict(X_test5))




df_important_features3 = p.DataFrame()
for idx, value in enumerate(clf_extratree.feature_importances_):
    if value >= 0.01:
        print X_all.columns[idx]
        df_important_features3[X_all.columns[idx]] = X_all[X_all.columns[idx]]
df_important_features3['AllCancerFlag'] = df['AllCancerFlag']
df_important_features3.drop(['Smokes', 'Hormonal Contraceptives', 'IUD', 'STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'], axis=1, inplace=True)

feature_cols_important3 = list(df_important_features3.columns[:-1])
print feature_cols_important3
X_all4 = df_important_features3[feature_cols_important3]

clf4 = DecisionTreeClassifier(random_state=1)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_all4, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf4.fit(X_train4, y_train4)
print 'DecisionTreeClassifier final features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train4.values, clf4.predict(X_train4))
print metrics.confusion_matrix(y_train4.values, clf4.predict(X_train4))
print 'Testing set:'
print metrics.classification_report(y_test4.values, clf4.predict(X_test4))
print metrics.confusion_matrix(y_test4.values, clf4.predict(X_test4))


clf4_extra = ExtraTreesClassifier(n_estimators=100, random_state=1)
X_train4, X_test4, y_train4, y_test4 = train_test_split(X_all4, y_all, test_size=num_test, train_size=num_train, random_state=1)
clf4_extra.fit(X_train4, y_train4)
print 'ExtraTreesClassifier final features adjustment:'
print 'Training set:'
print metrics.classification_report(y_train4.values, clf4_extra.predict(X_train4))
print metrics.confusion_matrix(y_train4.values, clf4_extra.predict(X_train4))
print 'Testing set:'
print metrics.classification_report(y_test4.values, clf4_extra.predict(X_test4))
print metrics.confusion_matrix(y_test4.values, clf4_extra.predict(X_test4))

#Plots
cancer_by_age = df.groupby(['Age', 'AllCancerFlag'])['AllCancerFlag'].count().unstack('AllCancerFlag')
cancer_by_age.columns = ['No', 'Yes']
cancer_by_age.plot.bar(title='Cancer cases by Age')
plt.xlabel('Age')
plt.ylabel('Cancer cases')
plt.tight_layout()
plt.show()


cancer_by_num_of_pregnancies = df.groupby(['Num of pregnancies', 'AllCancerFlag'])['AllCancerFlag'].count().unstack('AllCancerFlag')
cancer_by_num_of_pregnancies.columns = ['No', 'Yes']
cancer_by_num_of_pregnancies.plot.bar(title='Cancer cases by Num of pregnancies')
plt.xlabel('Num of pregnancies')
plt.ylabel('Cancer cases')
plt.tight_layout()
plt.show()


cancer_by_smokes = df.groupby(['Smokes', 'AllCancerFlag'])['AllCancerFlag'].count().unstack('AllCancerFlag')
cancer_by_smokes.columns = ['No', 'Yes']
cancer_by_smokes.plot.bar(title='Cancer cases by Smokes')
plt.xlabel('Smokes')
plt.ylabel('Cancer cases')
plt.tight_layout()
plt.show()