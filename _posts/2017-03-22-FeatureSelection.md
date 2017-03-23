---
layout: post
title: Feature Selection Using Lending Club Loans
comments: True
permalink: feature-selection
---

Learn how to select the right variables in a machine learning algorithm by building a predictive model to identify loans that are likely to default. In the process, encounter oversampling, error types and the LASSO within the context of the Python sklearn environment. 

One of the most relevant topics in machine learning is identifying which data attributes to use in your predictive models. Data scientists can develop ever more robust machine learning algorithms, but at its core, garbage data in = garbage data out. Today, it is increasingly easy to store large quantities of data. However, too many attributes can confuse models and render the analysis useless. Therefore, before investing time into tuning algorithms we need to identify a set of predictors with high predictive accuracy without loss of dimensionality. Benefits of a reduced set of predictors include decreased computational complexity, lower storage costs, and easier to interpret results.

### Imbalanced Datasets and Oversampling

The Lending Club dataset poses a number of challenges for efficient prediction. Predicting whether a loan will pay or default is an example of an imbalanced dataset. We have two classes for the main outcome, normal samples where the loan is current and eventually paid off with interest, and relevant samples, loans that are defaulted. 

In the original dataset, we have a small percentage of loans that default (relevant samples) and a large percentage of loans that are healthy. Therefore, a general predictor would say that all loans will be healthy, achieving a quite high rate of accuracy. This approach would completely misclassify all of our relevant data points, which goes against what we want to do.

In general, classifiers are more sensitive to detecting the majority class and less sensitive to minority class observations. If we don't preprocess our data, our outputs will be misleadingly accurate. In this exercise we use an oversampled dataset where our observations are 50% healthy loans and 50% defaulted loans. This will allow us to identify features that distinguish between the two classes of loans.

Another important note regarding predictive power for datasets that have a small class of interest is to use metrics other than accuracy. In particular, we can identify two types of misclassifications. Type I errors represent False Positives. These are loans that are predicted as safe, but actually default. Type II errors represent False Negatives, that is, loans predicted as defaults but are actually safe. The ideal model would minimize both, but having a low Type I error is more important in this instance. Because the loss incurred from a defaulted loan is higher than the lost revenue from not granting a loan, we want to prioritize reducing False Positives.

### Data Ecosystem

As with many machine learning analyses, we will work with the SciPy ecosystem using numpy, pandas, and sklearn. These libraries allow us to train multiple machine learning algorithms all within the umbrella of cross-validation. We can then compare their performance to select an optimal predictor to use in production.


```python
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
```

### Data Importing and Preprocessing

The datafile used for the analysis can be found [here](/assets/downloads/LCloanbook.xls). It consists of 123 attributes for over 60,000 loans collected between 2007 and 2015 First, we import our data as a pandas dataframe, and split our predictor and outcome variables. 

To prevent any of our variables from dominating our analysis, we scale our predictor variables. This lets the classification algorithms assign relatively equal weight to all of the potential predictors and prevents variables of larger magnitude from dominating. 


```python
seedValue = 99

loans = pd.read_excel('LCloanbook.xls', sheetname = 'Data')

xDF = loans.iloc[ : , 1:]
yDF = loans.iloc[ :, 0]

x = xDF.as_matrix()
y = yDF.as_matrix()

xScaled = StandardScaler().fit_transform(x)
```

### Full Model

First, we will attempt to train a number of machine learning algorithms on the full subset of predictors at our disposal. For all of the models, we use 10 fold cross validation to create train / test samples upon which we evaluate our accuracy. This lets us form unbiased predictions since we test on "uncontaminated" data.

We see that the logistic regression correctly predicts about 82% of the loans correctly. The decision tree predicts about 84% of the loans correctly, and finally the k-Nearest Neighbor algorithm performs quite poorly at 60%.


```python
loans_full_logistic = cross_val_score(LogisticRegression(), 
                                      x, y, scoring = 'accuracy', 
                                      cv = KFold(10, shuffle = True, 
                                                 random_state = seedValue))
```


```python
loans_full_tree = cross_val_score(DecisionTreeClassifier(random_state = seedValue), 
                                  x, y, scoring = 'accuracy', 
                                  cv = KFold(10, shuffle = True, 
                                             random_state = seedValue))
```


```python
loans_full_knn = cross_val_score(KNeighborsClassifier(n_neighbors = 1),
                                 xScaled, y, scoring = 'accuracy', 
                                 cv = KFold(10, shuffle = True, 
                                            random_state = seedValue))
```


```python
print("Accuracy of Logistic Regression: %.2f"%(np.mean(loans_full_logistic)*100), "%")
print("Accuracy of Decision Tree: %.2f" % (np.mean(loans_full_tree) * 100), "%")
print("Accuracy of k-Nearest Neighbor: %.2f" % (np.mean(loans_full_knn) * 100), "%")
```

    Accuracy of Logistic Regression: 82.04 %
    Accuracy of Decision Tree: 84.36 %
    Accuracy of k-Nearest Neighbor: 59.83 %


The k-Nearest Neighbor algorithm performs exceptionally poorly due to a phenomenon known as the curse of dimensionality. As we increase the number of predictors, the feature space's volume increases such that our data points become sparse. Since k-NN relies on a distance metric to identify similar data points, a sparse feature space will make calculating distances difficult. Because all of the observations appear sparse and dissimilar to each other, k-Nearest Neighbors cannot find any patterns in the data. Additionally, the algorithm runs for a very long time because it cannot quickly reject data points by using one coordinate as a lower bound when calculating distance.

### Reduced Model Attributes

With such a large number of variables, we can intuitively sense that not all of them will be useful in predicting if a loan will default. A solution to this problem would be to reduce the number of attributes we use in the model. A reduced model would ideally be easily interpretable while retaining predictive accuracy.

First, we select 10 attributes we believe to be most informative regarding loan default.

acc_now_delinq - If a borrower currently has a non-zero number of delinquent accounts, they would be much more likely to default an another loan.

delinq_2yrs - If a borrower has fallen over 30 days past due on a loan within the past two years, they are liable to fall overdue again in future loans.
   
dti - The borrower's total debt to income would be critical in determining delinquency because small loans would be more easily repaid. If a borrower falls behind on a large loan, then they would have difficulty meeting interest payments.
   
home_ownership_MORTGAGE - A borrower that currently has a mortgage would be more likely to default on a loan given that they have existing debt obligations.
   
home_ownership_OWN - A homeowner would be much less likely to default on a loan, considering they likely have no existing debt payments and much higher likelihood of free cashflow.
   
home_ownership_RENT - A person who rents their home would be more likely to default on a loan considering that rent payments are often a person's largest monthly cash outflow. Therefore, since rent payments come first in a person's budget, they might not have enough money left for loan repayments.
   
int_rate - By using the prevailing public's judgement in setting an interest rate on a loan, higher rates would correspond with delinquencies. Loans that are assessed as more risky are given larger interest payments to compensate for the higher risk of default.
   
mths_since_last_delinq - We see the length of time since a borrower has been delinquent as a key predictor because a borrower who has recently defaulted is likely to default again. Conversely, a borrower who defaulted ages ago may have since improved his or her financial status.
   
open_acc_6m - The number of open credit lines in the last 6 months is an important attribute because people who lean on credit for their daily expenses are more likely to default. A person would only seek to open multiple lines of credit if they have exhausted their currently open lines.
   
pub_rec - The number of derogatory public records is critical because it reflects on the borrower's past history with credit. With prior bankruptcies or liens, the borrower has shown a history of non-repayment.

### Reduced Model

Using the same three machine learning techniques as above, we compute the accuracy of the models using 10 fold Cross Validation. We see that our model's predictive power has significantly decreased as the feature space shrunk. Therefore, we should attempt to algorithmically select predictors, rather than relying on our own intuition.


```python
attrSlct = ['acc_now_delinq', 'delinq_2yrs', 'dti', \
            'home_ownership_MORTGAGE', 'home_ownership_OWN', \
            'home_ownership_RENT', 'int_rate', \
            'mths_since_last_delinq', 'open_acc_6m', 'pub_rec']

x_Reduced = xDF.loc[ : , attrSlct].as_matrix()
xScaled_Reduced = StandardScaler().fit_transform(x_Reduced)

loans_reduced_logistic = cross_val_score(LogisticRegression(), 
                                         x_Reduced, y, scoring = 'accuracy', 
                                         cv = KFold(10, shuffle = True, 
                                                    random_state = seedValue))
loans_reduced_tree = cross_val_score(DecisionTreeClassifier(random_state = seedValue), 
                                     x_Reduced, y, 
                                     scoring = 'accuracy', 
                                     cv = KFold(10, shuffle = True, 
                                                random_state = seedValue))
loans_reduced_knn = cross_val_score(KNeighborsClassifier(n_neighbors = 1), 
                                    xScaled_Reduced, y, 
                                    scoring = 'accuracy', 
                                    cv = KFold(10, shuffle = True, 
                                               random_state = seedValue))
```


```python
print("Accuracy of Logistic Regression: %.2f"%(np.mean(loans_reduced_logistic)*100), "%")
print("Accuracy of Decision Tree: %.2f" % (np.mean(loans_reduced_tree) * 100), "%")
print("Accuracy of 1-Nearest Neighbor: %.2f" % (np.mean(loans_reduced_knn) * 100), "%")
```

    Accuracy of Logistic Regression: 64.14 %
    Accuracy of Decision Tree: 63.14 %
    Accuracy of 1-Nearest Neighbor: 56.97 %


### LASSO  (Least Absolute Shrinkage and Selection Operator)

The LASSO (least absolute shrinkage and selection operator) is a way to algorithmically reduce the number of variables in your feature space. By selecting only a subset of all possible predictors, the LASSO hopes to improve prediction accuracy. The method forces certain coefficients to be zero, essentially turning them off. By setting a penalty term in our algorithm, we can control how many predictors we want in our final model, and the LASSO will select the most important subset with that number of variables. A high penalty term will decrease the number of selected attributes by forcing more terms to have a zero coefficient.

To keep our model size the same as above, we set the penalty term such that our final model will have 10 predictors.


```python
LR_l1 = LogisticRegression(C = 0.002, penalty='l1')
LR_l1.fit(xScaled, y)

interceptDF = pd.DataFrame(LR_l1.intercept_, index = ['Intercept'], 
		     columns = ['Value'])
coefDF = pd.DataFrame(LR_l1.coef_[0][np.where(LR_l1.coef_[0] != 0)], 
                      index = xDF.columns[np.where(LR_l1.coef_[0] != 0)], 
                      columns = ['Value'])

finalDF = pd.concat([interceptDF, coefDF])    
```

Our Lasso selected predictors are shown below:


```python
print("Number of Attributes:", sum(LR_l1.coef_[0] != 0))
print("In-Sample Accuracy: %.2f" % (LR_l1.score(xScaled, y) * 100), "%")
print("The coefficient values of LASSO-MODEL are: ")
print(finalDF)
```

    Number of Attributes: 10
    In-Sample Accuracy: 81.92 %
    The coefficient values of LASSO-MODEL are: 
                        Value
    Intercept        0.118507
    loan_amnt        1.140100
    int_rate         0.569931
    installment      0.093255
    annual_inc      -0.009162
    dti              0.032033
    inq_last_6mths   0.037828
    out_prncp       -1.463954
    total_rec_prncp -1.291003
    issue_year      -0.500611
    GRADE_A         -0.023372


### Revisit the Techniques with LASSO-selected attributes

Now that we have an algorithmically selected set of predictors, we want to see again how the new feature space performs. We run the same logistic regression, decision tree, and k-nearest neighbor algorithms and evaluate their cross-validated accuracy.


```python
attrLASSO = ['loan_amnt', 'int_rate', 'installment', \ 
	     'annual_inc', 'dti', 'inq_last_6mths', \
             'out_prncp', 'total_rec_prncp', 'issue_year', 'GRADE_A']

x_LASSO = xDF.loc[ : , attrLASSO].as_matrix()
xScaled_LASSO = StandardScaler().fit_transform(x_LASSO)

loans_LASSO_logistic = cross_val_score(LogisticRegression(), x_LASSO, y, 
				       scoring = 'accuracy', 
                                         cv = KFold(10, shuffle = True, 
                                         random_state = seedValue))
loans_LASSO_tree = cross_val_score(DecisionTreeClassifier(random_state = seedValue), 
				  x_LASSO, y, 
                                   scoring = 'accuracy', cv = KFold(10, 
                                   shuffle = True, random_state = seedValue))
loans_LASSO_knn = cross_val_score(KNeighborsClassifier(n_neighbors = 1), 
				   xScaled_LASSO, y, scoring = 'accuracy', 
                                    cv = KFold(10, shuffle = True, 
                                    random_state = seedValue))
```


```python
print("Accuracy of Logistic Regression: %.2f"%(np.mean(loans_LASSO_logistic)*100), "%")
print("Accuracy of Decision Tree: %.2f" % (np.mean(loans_LASSO_tree) * 100), "%")
print("Accuracy of 1-Nearest Neighbor: %.2f" % (np.mean(loans_LASSO_knn) * 100), "%")
```

    Accuracy of Logistic Regression: 81.97 %
    Accuracy of Decision Tree: 86.06 %
    Accuracy of 1-Nearest Neighbor: 81.50 %


Predictive performance improves considerably when using the LASSO selected predictors. This highlights a key finding of machine learning: more predictors are not always better. Sometimes, having too many variables confuses the algorithm.

Our decision tree classifier now has quite high accuracy, indicating that a non-linear separation of variables via a tree structure would do well to predict defaults. 

The curse of dimensionality that affects the k-Nearest Neighbor algorithm can also be clearly visible. With only a few predictors, it performs similarly to the Decision Tree and Logistic Regressions. With many predictors, it's performance lags behind considerably.

### Random Forest using LASSO-selected Attributes

From our initial predictive algorithms, a tree structure appears most effective at classifying loans likely to default. Therefore, it is reasonable to extend the algorithm to he Decision Tree's more robust cousin, the Random Forest. By selecting only a subset of all possible predictors on each fold of the cross validation, the model averages out the loan's classification. 

The Random Forest considerably outperforms all of our prior classification algorithms. Therefore, a Tree structure with a selected subset of attributes serves as the most effective method to predict loan defaults.


```python
loans_LASSO_RandomForest = cross_val_score(RandomForestClassifier(), 
	       x_LASSO, y, scoring = 'accuracy',
               cv = KFold(10, shuffle = True, random_state = seedValue)).mean()

print("Accuracy of Random Forest: %.2f" % (loans_LASSO_RandomForest * 100), "%")
```

    Accuracy of Random Forest: 88.89 %


Overall, for the loan dataset, the predictive model's attributes have a larger impact on performance than the model itself. Multiple models perform similarly on the dataset, but selecting the wrong attributes can significantly decrease predictive performance. While there exist models that perform poorly, we can achieve similar performance through a number of methods. Therefore, we should first identify a set of attributes with which to predict, and then apply classification techniquest to maximize performance.
