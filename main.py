import numpy as np
import pandas as pd

from dataset import *
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
def main():

    df = pd.read_csv('./data/ECG' + "1"+'.csv', index_col=0)
    dataset_raw = slice_ecg(df)
    a = dataset_raw[0][0:5].plot()
    a.set_ylabel("Voltage [mV]")
    plt.show()
    ##########
    #Load all ECGdata and create artificial noise for every window
    #########

    dataset_list_10 = []
    val = 1
    p = 0
    b = 0
    e = 0
    c = 0
    print("Window size 10s")
    for i in range(11):
        df = pd.read_csv('./data/ECG' + str(val)+'.csv', index_col=0)
        print("Processing: {}".format('ECG'+str(val)))
        dataset_raw_10 = slice_ecg(df,t=10)
        dataset_10,num_power,num_base,num_emg, num_cable = add_noise_col(dataset_raw_10,t=10)
        p += num_power
        b += num_base
        e += num_emg
        c += num_cable
        dataset_list_10.append(dataset_10)
        if(val==1):
            print("Plotting 5 example ECGs with different noise types ")
            for i in range(5):
                temp1 = dataset_10[i]['noise_ECG'].plot()
                plt.show()

        val +=1


    s = p + b+e+c
    print("Num_power: {} {}%, Num_base: {} {}%, Num_emg: {} {}%, Num_cable: {} {}%".format(p,(p/s)*100, b,(b/s)*100, e,(e/s)*100, c, (c/s)*100))
    dataset_list_5 = []
    val = 1
    p = 0
    b = 0
    e = 0
    c = 0
    print("Window size 5s")
    for i in range(11):
        df = pd.read_csv('./data/ECG' + str(val) + '.csv', index_col=0)
        print("Processing: {}".format('ECG' + str(val)))
        dataset_raw_5 = slice_ecg(df, t=5)
        dataset_5,num_power,num_base,num_emg, num_cable  = add_noise_col(dataset_raw_5,t=5)
        p += num_power
        b += num_base
        e += num_emg
        c += num_cable
        dataset_list_5.append(dataset_5)
        val += 1
    s = p + b + e + c
    print("Num_power: {} {}%, Num_base: {} {}%, Num_emg: {} {}%, Num_cable: {} {}%".format(p,(p/s)*100, b,(b/s)*100, e,(e/s)*100, c, (c/s)*100))

    ##########
    #create dataframe with all features and SNR
    #########
    it = 0
    print("Calculating features...")
    for ecg_list in dataset_list_10:
        if(it == 0):
            features_10 = calculate_all_features(ecg_list,it+1)
            it = 1
        else:
            features_10 = features_10.append(calculate_all_features(ecg_list,it+1), ignore_index=True)
            it +=1

    print(features_10)
    it = 0
    print("Calculating features...")
    for ecg_list in dataset_list_5:
        if (it == 0):
            features_5 = calculate_all_features(ecg_list,it+1)
            it = 1
        else:
            features_5 = features_5.append(calculate_all_features(ecg_list,it+1), ignore_index=True)
            it +=1

    print(features_5)

    count_zero = 0
    count_one = 0
    for i in list(features_5['Quality']):
        if i == 0:
            count_zero += 1
        else:
            count_one += 1
    print("5 sekunden Fenster")
    print("#0:", count_zero, "percent: ", count_zero / (count_one + count_zero))
    print("#1:", count_one, "percent: ", count_one / (count_one + count_zero))

    count_zero = 0
    count_one = 0
    for i in list(features_10['Quality']):
        if i == 0:
            count_zero += 1
        else:
            count_one += 1
    print("10 sekunden Fenster")
    print("#0:", count_zero, "percent: ", count_zero / (count_one + count_zero))
    print("#1:", count_one, "percent: ", count_one / (count_one + count_zero))
    '''
    #train svm and rf
    '''

    X = features_10.drop('Quality', axis=1)
    y = features_10['Quality']
    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    cv_inner = StratifiedKFold(n_splits=5)

    classifier_svm = SVC()
    classifier_rf = RandomForestClassifier(random_state=1)

    space_rf = dict()
    space_rf['n_estimators'] = np.linspace(10,101,5,dtype=int)
    space_rf['max_depth'] = np.linspace(200,2000,5)
    search_rf = dict()
    res_list = ['accuracy', 'recall', 'f1']
    for i in res_list:
        search_rf[i] = GridSearchCV(classifier_rf, space_rf, scoring=i, n_jobs=1, cv=cv_inner, refit=True)
    result = search_rf['accuracy'].fit(X_train, y_train)
    best_model = result.best_estimator_
    print("RF10 best model: {}", best_model)


    cv_outer = StratifiedKFold(n_splits=5)
    for i in res_list:
        scores_rf_acc = cross_val_score(search_rf[i], X, y, scoring=i, cv=cv_outer, n_jobs=-1)
        print(i,' RF_10: %.3f (%.3f)' % (scores_rf_acc.mean(), scores_rf_acc.std()))
    print('precision RF_10: %.3f (%.3f)' % (scores_rf_acc[0].mean(), scores_rf_acc[0].std()))
    print("##################")

    cv_inner = StratifiedKFold(n_splits=5)
    space_svm = dict()
    space_svm['C'] = [0.1, 1, 10, 100]
    space_svm['gamma'] = [1, 0.1, 0.01, 0.001]
    space_svm['kernel'] = ['rbf', 'poly', 'sigmoid']
    search_svm = dict()
    for i in res_list:
        search_svm[i] = GridSearchCV(classifier_svm, space_svm, scoring=i, n_jobs=1, cv=cv_inner, refit=True)
    result = search_svm['accuracy'].fit(X_train, y_train)
    best_model = result.best_estimator_
    print("SVM10 best model: {}", best_model)
    cv_outer = StratifiedKFold(n_splits=10, shuffle=False)
    for i in res_list:
        scores_svm_acc = cross_val_score(search_svm[i], X, y, scoring=i, cv=cv_outer, n_jobs=-1)
        print(i,' SVM_10: %.3f (%.3f)' % (scores_svm_acc.mean(), scores_svm_acc.std()))
    print('precision SVM_10: %.3f (%.3f)' % (scores_rf_acc[0].mean(), scores_rf_acc[0].std()))
    print("##################")

    ####5sec window
    X = features_5.drop('Quality', axis=1)
    y = features_5['Quality']
    y = y.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    cv_inner = StratifiedKFold(n_splits=5)

    classifier_svm = SVC()
    classifier_rf = RandomForestClassifier(random_state=1)

    space_rf = dict()
    space_rf['n_estimators'] = np.linspace(10, 101, 5, dtype=int)
    space_rf['max_depth'] = np.linspace(200, 2000, 5)
    search_rf = dict()
    for i in res_list:
        search_rf[i] = GridSearchCV(classifier_rf, space_rf, scoring=i, n_jobs=1, cv=cv_inner, refit=True)
    result = search_rf['accuracy'].fit(X_train, y_train)
    best_model = result.best_estimator_
    print("RF5 best model: {}", best_model)
    cv_outer = StratifiedKFold(n_splits=5)
    for i in res_list:
        scores_rf_acc = cross_val_score(search_rf[i], X, y, scoring= i, cv=cv_outer, n_jobs=-1)
        print(i,' RF_5: %.3f (%.3f)' % (scores_rf_acc.mean(), scores_rf_acc.std()))
    print('precision RF_5: %.3f (%.3f)' % (scores_rf_acc[0].mean(), scores_rf_acc[0].std()))
    print("##################")

    space_svm = dict()
    space_svm['C'] = [0.1, 1, 10, 100]
    space_svm['gamma'] = [1, 0.1, 0.01, 0.001]
    space_svm['kernel'] = ['rbf', 'poly', 'sigmoid']
    search_svm = dict()
    for i in res_list:
        if (i=='precision'):
            search_svm[i] = GridSearchCV(classifier_svm, space_svm, scoring='accuracy', n_jobs=1, cv=cv_inner, refit=True)
        else:
            search_svm[i] = GridSearchCV(classifier_svm, space_svm, scoring=i, n_jobs=1, cv=cv_inner, refit=True)
    result = search_svm['accuracy'].fit(X_train, y_train)
    best_model = result.best_estimator_
    print("SVM5 best model: {}", best_model)
    cv_outer = StratifiedKFold(n_splits=10)
    for i in res_list:
        scores_svm_acc = cross_val_score(search_svm[i], X, y, scoring=i, cv=cv_outer, n_jobs=-1)
        print(i,' SVM_5: %.3f (%.3f)' % (scores_svm_acc.mean(), scores_svm_acc.std()))
    print('precision RF_5: %.3f (%.3f)' % (scores_rf_acc[0].mean(), scores_rf_acc[0].std()))
    print("##################")
    
    '''
    '''
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    X = features_5.drop(['Quality','SNR','group'],axis=1)
    y = features_5['Quality']
    y = y.astype('int')

    #change path either to './data/qualitative_analysis/ID_17_bad_ECG_Data_0943.csv' or to './data/qualitative_analysis/ID_7_good_ECG_Data.csv'
    df = pd.read_csv('./data/qualitative_analysis/ID_17_bad_ECG_Data_0943.csv', index_col=0)
    cleanecg = slice_ecg(df,t=5)

    '''
    change the following variables in calculate_all_features: 
    for good data --> snr > 1 & qual = 1
    for bad data -->  snr < 1 & qual = 0
    '''
    features_test = calculate_all_features(cleanecg,1,snrbool=False)
    X_test_1 = features_test.drop(['Quality','SNR','group'],axis=1)

    y_good = features_test['Quality']
    y_good = y_good.astype('int')
    X_train, X_test, y_train, y_test = train_test_split(X_test_1, y_good, test_size=0.20)

    rf_clf = RandomForestClassifier(n_estimators=10,max_depth=200,n_jobs=-1)
    rf_clf.fit(pd.concat([X,X_test]),pd.concat([y,y_test]))
    y_pred_rf = rf_clf.predict(X_train)
    acc = accuracy_score(y_train,y_pred_rf)
    print("accuracy: ", acc)



if __name__ == "__main__":
    main()