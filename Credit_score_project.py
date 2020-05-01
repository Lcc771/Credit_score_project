#!/usr/bin/env python
# coding: utf-8

# In[131]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core import display as ICD
from scipy import stats
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import tree
import seaborn as sns
from IPython.display import Image, display
import pydotplus
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[132]:


conda install python-graphviz


# In[133]:


conda install pip


# In[134]:


pip install pydotplus


# In[135]:


pip install --upgrade scikit-learn


# # Part I
# # Section I - Variable Mapping

# In[136]:


raw_data = pd.read_csv('Credit_data_RSM6305.txt', sep = ' ')
raw_data.columns = ['Chk_acct', 'Duration', 'Credit_history', 'Purpose', 'Credit_amt',
                    'Savings_acct', 'Present_emp', 'Instalment_rate', 'Personal_status',
                    'Other_debt', 'Present_res', 'Property', 'Age', 'Other_installment',
                    'Housing', 'n_credits', 'Job', 'n_ppl', 'Telephone', 'Foreign_worker',
                    'Credit_status']


# In[137]:


# label categorical varibles as per the guidelines to better understand varibles and conduct further analysis

# Chk_acct
raw_data['Chk_acct'].replace(['A11','A12','A13','A14'], 
                             ['chk_acc < 0 DM',
                              '0 <= chk_acc < 200 DM',
                              'chk_acc >= 200 DM',
                              'no chk_acc'], inplace=True)

# Credit_history
raw_data['Credit_history'].replace(['A30','A31','A32','A33','A34'], 
                                   ['no credits taken/all credits paid back duly',
                                    'all credits at this bank paid back duly',
                                    'existing credits paid back duly till now',
                                    'delay in paying off in the past',
                                    'critical account/other credits existing'], inplace=True)

# Purpose
raw_data['Purpose'].replace(['A40','A41','A42','A43','A44','A45','A46','A47','A48','A49','A410'], 
                            ['car (new)',
                             'car (used)',
                             'furniture/equipment',
                             'radio/television',
                             'domestic appliances',
                             'repairs',
                             'education',
                             'vacation',
                             'retraining',
                             'business',
                             'others'], inplace=True)

# Savings_acct
raw_data['Savings_acct'].replace(['A61','A62','A63','A64','A65'], 
                                 ['savings_acct < 100 DM',
                                  '100 <= savings_acct < 500 DM',
                                  '500 <= savings_acct < 1000 DM',
                                  'savings_acct >= 1000 DM',
                                  'unknown/no savings account'], inplace=True)

# Present_emp
raw_data['Present_emp'].replace(['A71','A72','A73','A74','A75'], 
                                 ['unemployed',
                                  'emp < 1 year',
                                  '1 <= emp < 4 years',
                                  '4 <= emp < 7 years',
                                  'emp >= 7 years'], inplace=True)

# Personal_status
raw_data['Personal_status'].replace(['A91','A92','A93','A94','A95'], 
                                 ['male: divorced/separated',
                                  'female: divorced/separated/married',
                                  'male: single',
                                  'male: married/widowed',
                                  'female: single'], inplace=True)

# Other_debt
raw_data['Other_debt'].replace(['A101','A102','A103'], 
                               ['none',
                                'co-applicant',
                                'guarantor'], inplace=True)

# Property
raw_data['Property'].replace(['A121','A122','A123','A124'], 
                             ['real estate',
                              'building society savings agreement/life insurance',
                              'car or other',
                              'unknown/no property'], inplace=True)

# Other_installment
raw_data['Other_installment'].replace(['A141','A142','A143'], 
                                      ['bank',
                                       'stores',
                                       'none'], inplace=True)

# Housing
raw_data['Housing'].replace(['A151','A152','A153'], 
                            ['rent',
                             'own',
                             'for free'], inplace=True)

# Job
raw_data['Job'].replace(['A171','A172','A173','A174'], 
                        ['unemployed/unskilled - non-resident',
                         'unskilled - resident',
                         'skilled employee/official',
                         'management/self-employed/highly qualified employee/officer'], inplace=True)

# Telephone
raw_data['Telephone'].replace(['A191','A192'], 
                              ['none',
                               'yes, registered under the customer’s name'], inplace=True)

# Foreign_worker
raw_data['Foreign_worker'].replace(['A201','A202'], 
                                   ['yes',
                                    'no'], inplace=True)


# In[138]:


# replace credit status column with 0 and 1 for later calculation
# 0 for bad credit, 1 for good credit
raw_data['Credit_status'].replace([1,2], [0,1], inplace=True)
raw_data.head()


# In[139]:


raw_data['Credit_status'].value_counts()


# In[140]:


raw_data.info()


# # Section II - Exploratory Data Analysis & Wrangling

# ## 1) Plot Analysis
# ### Categorical variables - Bar Chart

# In[141]:


target = 'Credit_status'
cat_var = ['Chk_acct', 'Credit_history', 'Purpose', 'Savings_acct', 'Present_emp',
           'Personal_status', 'Other_debt', 'Property', 'Other_installment', 'Housing',
           'Job', 'Telephone', 'Foreign_worker']

for i in cat_var:
    cross_df = pd.crosstab(raw_data[i], raw_data['Credit_status'])
    cross_df.plot.bar(rot=45, figsize=(10,5))    


# ### Continuous variables - Histogram

# In[142]:


cts_var = ['Duration', 'Credit_amt', 'Instalment_rate', 'Present_res','Age','n_credits','n_ppl']

for i in cts_var:
    d0 = raw_data[raw_data['Credit_status']==0][i]
    d1 = raw_data[raw_data['Credit_status']==1][i]
    plt.figure(figsize=(10,5))
    plt.hist([d0, d1])
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.legend(['0', '1'])
    plt.title(i + ' Histogram')
    plt.show()


# ### Continuous variables - Box Plot

# In[143]:


for i in cts_var:
   plt.figure()
   sns.set_style("whitegrid") 
   sns.boxplot(x='Credit_status', y=i, data=raw_data)


# In[144]:


raw_data.describe()


# ## Outliers Treatment: Winsorization

# In[145]:


raw_data2 = raw_data.copy()
cts_outliers = ['Duration', 'Credit_amt', 'Age']

# Duration: use 90% percentile
data_dur = raw_data2['Duration']
upper_p_dur = data_dur.quantile(0.9)
raw_data2['Duration'] = np.where(data_dur > upper_p_dur, upper_p_dur, data_dur)

# Credit_amt: use 90% percentile
data_cre = raw_data2['Credit_amt']
upper_p_cre = data_cre.quantile(0.9)
raw_data2['Credit_amt'] = np.where(data_cre > upper_p_cre, upper_p_cre, data_cre)

# Age: use 95% percentile
data_age = raw_data2['Age']
upper_p_age = data_age.quantile(0.95)
raw_data2['Age'] = np.where(data_age > upper_p_age, upper_p_age, data_age)


# In[146]:


raw_data2.describe()


# In[147]:


# box plots after removing outliers
for i in cts_outliers:
    plt.figure()
    sns.set_style("whitegrid") 
    sns.boxplot(x='Credit_status', y=i, data=raw_data2)


# ## 2) Dealing with Missing Data: Data Imputation

# #### Continuous variables: replace NaN with median

# In[148]:


for i in cts_var:
    raw_data2[i].fillna(raw_data2[i].median(), inplace=True)

raw_data2.describe()


# #### Categorical variables: replace NaN with the highest occurance

# In[149]:


for i in cat_var:
    raw_data2[i].fillna((raw_data2[i].value_counts(ascending=True).index)[-1], inplace=True)
    
raw_data2.describe(include=np.object)


# ## 3) & 4) Cross Contingency Table & Chi-square Test - Categorical Variables

# In[150]:


# Chi-square Test Assumptions

# (1) The levels (groups) of the variables being tested are mutually exclusive
# (2) The groups being tested are independent
# (3) The value of expected cells are greater than 5 for at least 20% of the cells


# In[151]:


significant_cats = []

for i in cat_var:
    table = pd.crosstab(raw_data2.Credit_status, raw_data2[i], margins=True, margins_name='Total')
    table_perc = pd.crosstab(raw_data2.Credit_status, raw_data2[i], normalize='index')
    
    table2 = pd.crosstab(raw_data2.Credit_status, raw_data2[i])
    chi_value, p_value, dof, ex = stats.chi2_contingency(table2)
    ICD.display(table, table_perc)
    
    print('H0: ' + i + ' is independent from Credit_status')
    print('Chi-square value for ' + i + ' is ' + str(chi_value))
    print('P-value for ' + i + ' is ' + str(p_value))
    print('Degree of freedom for ' + i + ' is ' + str(dof))
    print('Expected values for ' + i + ' are: ' + str(ex))
    
    if p_value < 0.05:
        significant_cats.append(i)
        print('Reject null hypothesis since p-value < 0.05. Categorical variable: ' + i + ' is statistically significant.')
    else:
        print('Fail to reject null hypothesis.')
        
    print('------------------------------------------------------------------------------------------')


# In[152]:


significant_cats


# ## 5) Descriptive Statistics - Continous Variables

# In[153]:


# Use MinMaxScaler to standardize cts var
# create the Scaler object
scaler = MinMaxScaler()
std_cts_data = scaler.fit_transform(raw_data2[cts_var])
std_cts_data = pd.DataFrame(std_cts_data, columns=cts_var)
std_cts_data.index += 1

std_cts_data


# In[154]:


std_cts_data.describe()


# In[155]:


# replace df with standardized columns for cts var
for i in cts_var:
    raw_data2[i] = std_cts_data[i]
    
raw_data2


# ## 6) Correlation Analysis

# #### Linear correlation: continuous - continuous

# In[156]:


# build the corr. matrix for cts-cts variables
corr_cts_cts = raw_data2[cts_var].corr() #same corr. matrix after standardizing cts. variables
corr_cts_cts


# #### Cramer’s V: categorical - categorical

# In[157]:


# Cramer’s V is used as post-test to determine strengths of association after chi-square has determined significance 
# Output is in the range of [0,1], where 0 means no association and 1 is full association
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    return np.sqrt(phi2/min((k-1),(r-1)))


# In[158]:


# build the corr. matrix for categorical-categorical variables
# add 'Credit_status' to categorical variables to create corr. matrix
cat_var_target = cat_var.copy()
cat_var_target.append('Credit_status')
corr_cat_cat = pd.DataFrame(columns=cat_var_target, index=cat_var_target)

x = 0
for i in cat_var_target:
    y = 0
    for j in cat_var_target:
        result = cramers_v(raw_data2[i], raw_data2[j])
        corr_cat_cat.iloc[x,y] = round(result,4)
        y += 1
    x += 1

corr_cat_cat


# #### Correlation ratio: categorical - continuous

# In[159]:


def correlation_ratio(categories, observations):
    codes, uniques = pd.factorize(categories)
    cat_num = np.max(codes)+1
    # initialize arrays
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    
    for i in range(0,cat_num):
        cat_observations = observations[np.argwhere(codes==i).flatten()]
        n_array[i] = len(cat_observations)
        y_avg_array[i] = np.average(cat_observations)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(observations,y_total_avg),2))
    
    if numerator == 0:
        eta = 0
    else:
        eta = np.sqrt(numerator/denominator)
    return eta


# In[160]:


raw_data3 = raw_data2.copy()

# set index from 0 to apply 'correlation_ratio' function
raw_data3.reset_index(inplace=True) 

# build the corr. matrix for categorical-continuous variables
corr_cat_cts = pd.DataFrame(columns=cts_var, index=cat_var_target)

x = 0
for i in cat_var_target:
    y = 0
    for j in cts_var:
        result = correlation_ratio(raw_data3[i], raw_data3[j])
        corr_cat_cts.iloc[x,y] = round(result,4)
        y += 1
    x += 1
corr_cat_cts


# In[161]:


# another transposed corr. matrix for categorical-continuous variables 
# use for later concatenation
corr_cat_cts2 = pd.DataFrame(columns=cat_var_target, index=cts_var)

x = 0
for i in cts_var:
    y = 0
    for j in cat_var_target:
        result = correlation_ratio(raw_data3[j], raw_data3[i])
        corr_cat_cts2.iloc[x,y] = round(result,4)
        y += 1
    x += 1
corr_cat_cts2


# #### Correlation matrix - All variables

# In[162]:


# concatenate 3 matrices above
corr_temp1 = pd.concat([corr_cat_cat, corr_cat_cts], axis=1)
corr_temp2 = pd.concat([corr_cat_cts2, corr_cts_cts], axis=1)
corr_matrix = pd.concat([corr_temp1, corr_temp2])
corr_matrix


# In[163]:


def heatmap_matrix(matrix):
    # heatmap the matrix
    matrix = matrix[matrix.columns].astype(float)
    sns.set(style="white")
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(matrix, dtype=np.bool))
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 8))
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[164]:


heatmap_matrix(corr_matrix)


# ## 7) Create New Features for Predictivity Improvement

# In[165]:


# raw_data3 = raw_data2.copy()

# raw_data3['Duration*Credit_amt'] = raw_data3['Duration']*raw_data3['Credit_amt']
# raw_data3


# # Section III - Variable Selection and Transformation

# ## 1) Binning Continuous Variables - Decision Tree Approach

# In[166]:


cts_var


# In[167]:


def decision_tree_binning(x, y, df, x_var):
    tree_model = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
    tree_model.fit(x.to_frame(), y)
    df['tree_'+ x_var] = tree_model.predict_proba(x.to_frame())[:,1] 
    return tree_model


# In[168]:


data_DTbinning = raw_data2.copy()
graph_list = []
bins_list = []

for i in cts_var:
    DT = decision_tree_binning(data_DTbinning[i], data_DTbinning['Credit_status'], data_DTbinning, i)
    
    #data.groupby(['tree_' + i])['Credit_status'].count().plot.bar(rot=20)
    
    bins = pd.concat([data_DTbinning.groupby(['tree_' + i])[i].min(),
                      data_DTbinning.groupby(['tree_' + i])[i].max()], axis=1)
    
    dot_data = tree.export_graphviz(DT, 
                                    feature_names=[i],
                                    #class_names = ['1','2'],
                                    filled=True)
    graph_list.append(pydotplus.graph_from_dot_data(dot_data))
    bins_list.append(bins)


# In[169]:


# Checking limit buckets generated by the tree
for bins in bins_list:
    display(bins)


# In[170]:


# Visualizing the tree
for graph in graph_list:
    display(Image(graph.create_png())) 


# In[171]:


# check the monotonic relationship between the feature and target
# use Age as an example
fig_monotonic = plt.figure()
fig_monotonic = data_DTbinning.groupby(['tree_Age'])['Credit_status'].mean().plot()
fig_monotonic.set_title('Monotonic relationship between feature and target')
fig_monotonic.set_ylabel('Credit_status')


# In[172]:


# check the number of appearance in each bucket/bin
# use Age as an example
data_DTbinning.groupby(['tree_Age'])['Credit_status'].count().plot.bar(rot=20)


# In[173]:


data_DTbinning = data_DTbinning.drop(cts_var, axis=1)
data_DTbinning.head()


# ## 2) WOE and IV

# In[174]:


def calculate_WOE_IV(df, feature, target):
    attr_info = []
    feature_attr = df[feature].unique()
    for i in range(len(feature_attr)):
        attribute = list(feature_attr)[i]
        attr_info.append({
            'Attribute/Bin': attribute,
            'All': df[df[feature]==attribute].count()[feature],
            'Default': df[(df[feature]==attribute) & (df[target]==1)].count()[feature],
            'Non-default': df[(df[feature]==attribute) & (df[target]==0)].count()[feature]    
        })
    
    df_woe = pd.DataFrame(attr_info)
    df_woe['Percent_Default'] = df_woe['Default'] / df_woe['Default'].sum()
    df_woe['Percent_Non-default'] = df_woe['Non-default'] / df_woe['Non-default'].sum()
    df_woe['WOE'] = np.log(df_woe['Percent_Non-default'] / df_woe['Percent_Default'])
    df_woe = df_woe.replace({'WOE': {np.inf: 0, -np.inf: 0}})
    df_woe['IV'] = (df_woe['Percent_Non-default']-df_woe['Percent_Default']) * df_woe['WOE']
    iv = df_woe['IV'].sum()
    
    df_woe = df_woe.sort_values(by='WOE')
    
    return df_woe, iv


# In[175]:


estimation_var = [] #keep a final set of variables for estimation
lst_iv = []
lst_drop_iv = []
df_iv = pd.DataFrame(data_DTbinning.drop(['Credit_status'],axis=1).columns, columns=['Variable'])
df_woe = data_DTbinning.copy()

for col in data_DTbinning.drop(['Credit_status'],axis=1).columns:
    print('WOE & IV for column: {}'.format(col))
    df, iv = calculate_WOE_IV(data_DTbinning, col, 'Credit_status')
    
    # WOE transformation -> Question 3)
    df_woe[col] = df_woe[col].map(df.set_index('Attribute/Bin')['WOE'])
    
    # track IV of each variable
    lst_iv.append(iv)
    
    # track varibles that are used for estimation
    if iv >= 0.02:
        estimation_var.append(col)
        lst_drop_iv.append('-')
    else: 
        lst_drop_iv.append('YES')
    
    display(df)
    print('IV score: {:.2f}'.format(iv))
    print('\n')

# put all IV into the dataframe
df_iv.insert(1, 'IV', lst_iv)
df_iv.insert(2, 'Drop Variable?', lst_drop_iv)


# In[176]:


df_iv


# In[177]:


estimation_var


# ## 3) Use the WOEs as model inputs

# In[178]:


# combine categories/attributes with similar WOE values and replace categories/attributes with WOE values
# use WOE values rather than input values in the model

df_WOE = df_woe.copy()
df_WOE = df_WOE.drop(['Job', 'Telephone', 'tree_Present_res', 'tree_n_credits', 'tree_n_ppl'], axis=1)
df_WOE = df_WOE.add_prefix('WOE_')
df_WOE = df_WOE.rename(columns={'WOE_tree_Duration': 'WOE_Duration', 
                                'WOE_tree_Credit_amt': 'WOE_Credit_amt',
                                'WOE_tree_Instalment_rate': 'WOE_Instalment_rate',
                                'WOE_tree_Age': 'WOE_Age',
                                'WOE_Credit_status': 'Credit_status'})

df_WOE


# ## 4) Correlation matrix (All varibles are continuous)

# In[179]:


corr_matrix_WOE = df_WOE.drop(['Credit_status'], axis=1).corr()
corr_matrix_WOE


# In[180]:


heatmap_matrix(corr_matrix_WOE)


# # Section IV - Estimation

# In[181]:


# Use 15 varibles for estimation (based on IV)
X = df_WOE.copy()
y = X['Credit_status']

X = X.drop(['Credit_status'], axis=1)
X


# In[182]:


# randomly sample 70% of the data as training set and keep the rest of 30% as test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# In[183]:


X_train


# ## 1) Logistic Regression

# ### Logistic Function using Newton's Method

# In[184]:


from numpy.linalg import inv

def logit(X, y, max_iter, fit_intercept=1):
    tol = 1/np.power(10, 11)
    change = tol+1
    
    N = X.shape[0] #number of instances
    if fit_intercept:
        K = X.shape[1]+1 #number of varibles (include one constant for intercept)
        intercept = np.ones((N, 1)) #add a vector of ones to X to be multiplied with intercept term if fit_intercept=1 
        X = np.c_[intercept, X]
    else:
        X = X.to_numpy() #convert to numpy narray
        K = X.shape[1]
    
    # initialize coefficiant(b) and score(bx) vectors
    b = np.zeros(K)
    score = np.zeros(N)
    if fit_intercept:
        y_bar = np.mean(y)
        b[0] = np.log(y_bar/(1-y_bar))  
    for i in range(N):
        score[i]=b[0]
    
    # initialize variables used in the Newton method
    lambda_ = np.zeros(N)
    lnl_ = np.zeros(max_iter)
    
    iter_ = 1
    while abs(change) > tol and iter_ < max_iter:
        # print('iter: ', iter_, ' change: ', change)
        iter_ += 1
        
        # reset gradient and hesse for each iteration
        gradient_ = np.zeros(K)
        hesse_ = np.zeros((K,K))
        
        # Compute prediction lambda_, gradient gradient_, Hessian hesse_, and log likelihood lnl_
        for i in range(N):
            x_i = X[i] #K*1
            y_i = y.iloc[i] #float
            lambda_[i] = 1 / (1 + np.exp(-np.dot(x_i.T, b))) #N*1
            
            for j in range(K):
                x_i_j = X[i, j] #float
                gradient_[j] = gradient_[j] + (y_i-lambda_[i])*x_i_j
                
                for jj in range(K):
                    x_i_jj = X[i, jj] #float
                    hesse_[jj,j] = hesse_[jj,j] - lambda_[i] * (1-lambda_[i]) * x_i_jj * x_i_j
            
            lnl_[iter_] = lnl_[iter_] + y_i*np.log(lambda_[i]) + (1-y_i)*np.log(1-lambda_[i])
        
        # take inverse of Hessian
        hesse_inv = inv(hesse_) #K*K
        hesse_invg = np.dot(hesse_inv, gradient_) #K*1
        
        change = lnl_[iter_] - lnl_[iter_-1]
        
        # if convergent, exit and keep b corresponding with the estimated hessian
        if abs(change) <= tol:
            break
        
        # apply Newton's scheme for updating coefficient/weight vector (b)
        for j in range(K):
            b[j] = b[j] - hesse_invg[j]
         
        # compute the new score (bx)
        for i in range(N):
            score[i] = 0
            for j in range(K):
                score[i] = score[i] + b[j] * x_i_j
    
    return b, hesse_inv


# ### Computing Statistics After Estimation

# In[185]:


def logit_statistics(b_opt, Hinv_opt, X, fit_intercept=1): #note: k variables vs (k+1) coeff. estimated by logit if intercept is included
    lst_col = list(X.columns.values) 
    if fit_intercept:
        lst_col.insert(0, 'Intercept')
    dict_pvalue = {}
    all_signif = True
    
    for j in range(len(lst_col)): 
        col = lst_col[j] #jth variable
        
        b = b_opt[j]
        SE = np.sqrt(-Hinv_opt[j,j])
        t = b/SE
        pvalue = 2*(1 - stats.norm.cdf(abs(t)))
        
        if pvalue <= 0.05:
            dict_pvalue[col] = pvalue
        else:
            all_signif = False
    
    return dict_pvalue, all_signif


# ### Selecting Significant Varibles

# In[186]:


# loop until all the variables are significant at the 5% level

def logit_var_select(X_train, X_test, y_train, max_iter, fit_intercept=1):
    while True:
        b_opt, Hinv_opt = logit(X_train, y_train, max_iter, fit_intercept=1)
        dict_pvalue, all_signif = logit_statistics(b_opt, Hinv_opt, X_train, fit_intercept=1)
        signif_var = list(dict_pvalue.keys())
        if fit_intercept:
            signif_var.remove('Intercept')
        
        if all_signif:
            break
        
        # update varibles in the training X and test X
        X_train = X_train[signif_var]
        X_test = X_test[signif_var]
        
    return signif_var, X_train, X_test, b_opt, dict_pvalue


# In[187]:


signif_var_logit, X_train_logit, X_test_logit, b_logit, pvalues_logit = logit_var_select(X_train, X_test, y_train, 50, fit_intercept=1)

# final chosen variables
signif_var_logit


# In[188]:


# validate: all p-values are less than 0.05
pvalues_logit


# In[189]:


# All coefficients are negative, which makes sense because:

# (0) Intercept
# (1) Chk_acct
# (2) Credit_history
# (3) Purpose
# (4) Savings_acct
# (5) Present_emp
# (6) Other_debt 
# (7) Property
# (8) Duration
# (9) Credit_amt
# (10) Instalment_rate

b_logit


# ### In-sample Prediction

# In[190]:


# new training set
X_train_logit


# In[191]:


# new test set
X_test_logit


# In[192]:


# predict default prob.
def logit_predict_prob(X, b, fit_intercept=1):  
    if fit_intercept:
        intercept = np.ones((X.shape[0], 1)) #add a vector of ones to X to be multiplied with intercept term if fit_intercept=1 
        X = np.c_[intercept, X]
    exp_score = np.exp(np.dot(b, X.T))
    prob = exp_score/(1+exp_score)
    return prob


# ### ROC Curve for the in-sample prediction

# In[193]:


# ROC curve is a plot of the true positive rate against the false positive rate
# It shows the tradeoff between sensitivity and specificity

def ROC_plot(y, y_pred_prob):
    # AUC score
    lr_auc = metrics.roc_auc_score(y, y_pred_prob)
    # ROC curve
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y, y_pred_prob)

    # plot the roc curve for the model
    plt.figure(figsize=(7,5))
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='ROC curve (area = %0.3f)' % lr_auc)
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    # show the legend
    plt.legend()
    # show the title
    plt.title('Receiver Operating Characteristic (ROC)')
    # show the plot
    plt.show()
    
    return lr_auc


# In[194]:


# predict in-sample probabilities
y_pred_logit_insample = logit_predict_prob(X_train_logit, b_logit, fit_intercept=1)

# plot ROC curve
AUC_logit_insample = ROC_plot(y_train, y_pred_logit_insample)


# ### KS Test - Choose Cutoff Value

# In[195]:


def KS_test(y, y_pred_prob):
    # choose the threshold where fpr and tpr differ the most as the cutoff value
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob)
    diff = tpr-fpr
    opt_idx = np.argmax(diff)
    opt_threshold = thresholds[opt_idx]
    KS_statistic = diff[opt_idx]
    
    # visualize in the graph
    plt.figure(figsize=(7,5))
    fpr_line, = plt.plot(thresholds, fpr, label='False Positive Rate')
    tpr_line, = plt.plot(thresholds, tpr, label='True Positive Rate')
    plt.axvline(x=opt_threshold, color='r', linestyle='--', label='Cutoff') #the cutoff line
    plt.xlim(0, 1)
    plt.legend(handles=[fpr_line, tpr_line])
    plt.show()
    
    print("Maximum KS is " + str(np.round(KS_statistic,3)) + "%")
    print("Cut-off value is chosen as " + str(np.round(opt_threshold,3)))
    return opt_threshold, KS_statistic


# In[196]:


cutoff_logit, KS_logit = KS_test(y_train, y_pred_logit_insample)


# In[197]:


cutoff_logit


# ### Optional: Compare Logistic Regression Results with Sklearn Built-in Package

# In[198]:


from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot
# using LogisticRegression package 

# fit a model
lr_model = LogisticRegression(solver='newton-cg')
lr_model.fit(X_train_logit, y_train)
# predict in-sample probabilities
lr_probs = lr_model.predict_proba(X_train_logit)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate score
lr_auc = metrics.roc_auc_score(y_train, lr_probs)
# roc curve
lr_fpr, lr_tpr, _ = metrics.roc_curve(y_train, lr_probs)
# plot the roc curve for the model
plt.figure(figsize=(7,5))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(lr_fpr, lr_tpr, marker='.', label='ROC curve (area = %0.3f)' % lr_auc)
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.xlim([0, 1])
plt.ylim([0, 1])
# show the legend
plt.legend()
# show the title
plt.title('Receiver Operating Characteristic (ROC)')
# show the plot
plt.show()


# In[199]:


lr_model.intercept_


# In[200]:


lr_model.coef_


# In[201]:


'''So, results are consistent in terms of using our user-defined logit function & sklearn package'''


# ## 2) Stepwise Logistic Regression - Backward Selection

# ### Using P-value

# In[202]:


def swlogit_statistics(b_opt, Hinv_opt, X, fit_intercept=1): #note: k variables vs (k+1) coeff. estimated by logit if intercept is included
    lst_col = list(X.columns.values) 
    if fit_intercept:
        lst_col.insert(0, 'Intercept')
    dict_pvalue = {}
    all_signif = True
    
    for j in range(len(lst_col)): 
        col = lst_col[j] #jth variable
        
        b = b_opt[j]
        SE = np.sqrt(-Hinv_opt[j,j])
        t = b/SE
        pvalue = 2*(1 - stats.norm.cdf(abs(t)))
        
        dict_pvalue[col] = pvalue
        if pvalue > 0.05:
            all_signif = False
    
    return dict_pvalue, all_signif


# In[203]:


# start from the full model containing all 𝑘 predictors/variables
# remove one insignificant varible at per iteration
# loop until all the variables are significant at the 5% level

def swlogit_var_select(X_train, X_test, y_train, max_iter, fit_intercept=1):
    while True:
        b_opt, Hinv_opt = logit(X_train, y_train, max_iter, fit_intercept=1)
        dict_pvalue, all_signif = swlogit_statistics(b_opt, Hinv_opt, X_train, fit_intercept=1)
        signif_var = list(dict_pvalue.keys())
        
        # find the largest p-value, if it's larger than 0.05 -> remove it
        max_value = max(dict_pvalue.values())
        if max_value > 0.05:
            max_key = max(dict_pvalue, key=lambda k: dict_pvalue[k])
            signif_var.remove(max_key)
        
        if fit_intercept:
            signif_var.remove('Intercept')
        
        if all_signif:
            break
        
        # update varibles in the training X and test X
        X_train = X_train[signif_var]
        X_test = X_test[signif_var]
        
    return signif_var, X_train, X_test, b_opt, dict_pvalue


# In[204]:


signif_var_swlogit, X_train_swlogit, X_test_swlogit, b_swlogit, pvalues_swlogit = swlogit_var_select(X_train, X_test, y_train, 50, fit_intercept=1)

# final chosen variables
signif_var_swlogit  #get one extra varible Personal_status compared to logistic regression


# In[205]:


# validate: all p-values are less than 0.05
pvalues_swlogit


# In[206]:


# All coefficients are negative, which makes sense because:

# (0) Intercept
# (1) Chk_acct
# (2) Credit_history
# (3) Purpose
# (4) Savings_acct
# (5) Present_emp
# (6) Other_debt 
# (7) Property
# (8) Duration
# (9) Credit_amt
# (10) Instalment_rate

b_swlogit


# In[207]:


'''Note: by setting random_state=0, every split will be always the same, 
so in this case, logistic and stepwise logictis have the same performance'''


# ### KS Test - Choose Cutoff Value

# In[208]:


y_pred_swlogit_insample = logit_predict_prob(X_train_swlogit, b_swlogit, fit_intercept=1)
cutoff_swlogit, KS_swlogit = KS_test(y_train, y_pred_swlogit_insample)


# In[209]:


cutoff_swlogit


# ## 3) Decision Tree

# In[210]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[211]:


def Pruning(trainx, trainy):
    clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)
    path = clf.cost_complexity_pruning_path(trainx, trainy)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    return ccp_alphas, impurities


# In[212]:


from sklearn.model_selection import KFold

def optimal_alpha(x, y, k):
    kfold = KFold(n_splits=k)
    alpha_dict = {}
    clfs_list = []
    
    # for k = 1,...,10, using every fold except the kth
    for train_index, test_index in kfold.split(x):
        clfs = []
        trainX, testX = x[train_index], x[test_index]
        trainY, testY = y[train_index], y[test_index]
        ccp_alphas, impurities = Pruning(trainX, trainY)
        
        # construct a sequence of trees T1 , . . . , Tm for a range of values of alpha
        for ccp_alpha in ccp_alphas:
            decision_tree = tree.DecisionTreeClassifier(criterion='entropy', ccp_alpha=ccp_alpha, random_state=0)
            decision_tree.fit(trainX, trainY)
            clfs.append(decision_tree)
        # for each tree Ti, calculate the RSS or score on the test set
        test_scores = [clf.score(testX, testY) for clf in clfs]
        max_scores = max(test_scores)
        ind = test_scores.index(max_scores)
        alpha = ccp_alphas[ind]
        alpha_dict[alpha] = max_scores
#         print("Number of nodes in the optimal tree is: {} with ccp_alpha: {}".format(
#               clfs[ind].tree_.node_count, ccp_alphas[ind]))
#         print("Threshold the optimal tree is: {} with ccp_alpha: {}".format(
#               clfs[ind].tree_.threshold, ccp_alphas[ind]))
#         print("Features of the optimal tree is: {} with ccp_alpha: {}".format(
#               clfs[ind].tree_.feature, ccp_alphas[ind]))
        clfs_list.append(clfs[ind])
    return clfs_list, alpha_dict


# In[213]:


import os
os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

trees, alpha_dict = optimal_alpha(np.array(X_train), np.array(y_train), 10)
opt_alpha = max(alpha_dict, key=alpha_dict.get)
max_testscores = alpha_dict[opt_alpha]
opt_idx = list(alpha_dict.keys()).index(opt_alpha)
print("The higest test score of the optimal tree is: {}% with ccp_alpha: {}".format(
      round(max_testscores*100,2), round(opt_alpha,4)))


# In[214]:


opt_tree = trees[opt_idx]
dot_data = tree.export_graphviz(opt_tree, filled=True, out_file=None, feature_names=X.columns)
graph = pydotplus.graph_from_dot_data(dot_data)
display(Image(graph.create_png()))


# ## 4) Random Forest

# ### Grid search

# In[88]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def random_forest(trainx, trainy):
    rf = RandomForestClassifier(criterion='entropy')
    rf.fit(trainx, trainy)
    
    n_estimators = list(np.arange(200, 2010, 100))
    max_features = list(np.arange(2, 11, 1))
    grid_params = {'n_estimators': n_estimators,
                   'max_features': max_features}
    grid_search = GridSearchCV(rf, param_grid=grid_params, cv=10).fit(trainx, trainy)
    
    best_estimator = grid_search.best_estimator_
    best_param = grid_search.best_params_
    
    return best_estimator, best_param


# In[89]:


best_estimator, best_param = random_forest(X_train, y_train)


# In[90]:


best_estimator


# In[91]:


best_param


# ### Results based on optimal hyper parameters

# In[215]:


# fit random forest using the optimal set of hyper parameters 
rf_opt = RandomForestClassifier(criterion='entropy', n_estimators=best_param['n_estimators'], max_features=best_param['max_features'], random_state=0)
rf_opt.fit(X_train, y_train)

# in-sample prediction
y_pred_rf_insample = rf_opt.predict(X_train)
rf_score_insample = metrics.accuracy_score(y_train, y_pred_rf_insample)
rf_score_insample


# In[216]:


def cnf_matrix(y_test, y_predict_class):
    cnf_matrix = metrics.confusion_matrix(y_test, y_predict_class)
    ax = plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, ax=ax, fmt='g', cmap='Blues')
    ax.set_ylim([0,2])
    ax.set_xlabel('Predicted Credit Status')
    ax.set_ylabel('Actual Credit Status')
    ax.set_title('Confusion Matrix')


# In[217]:


cnf_matrix(y_train, y_pred_rf_insample)


# In[218]:


'''In sample accuracy is pretty high, but possibly the model is overfitted'''


# ### Visualizing variable's importance

# In[219]:


# Get variables' importance
vars_ = X.columns
importances = list(rf_opt.feature_importances_)

# Visualize variables' importance
plt.figure(figsize=(10,8))
plt.bar(vars_, importances, align="center")
plt.ylabel('Importance')
plt.xlabel('Variable')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', color='#D3D3D3', linestyle='solid')
plt.show()


# # Section V - Performance Validation

# ## 1) Out of Sample Prediction - Default Probability 

# ### (a) Logistic Regression

# In[220]:


# cutoff value is obtained from KS test
cutoff_logit


# In[221]:


# predict out of sample default probability 
y_pred_logit = logit_predict_prob(X_test_logit, b_logit, fit_intercept=1)
d_logit = {'Default Prob - Logistic Regression': y_pred_logit}
df_prob_logit = pd.DataFrame(data=d_logit)
df_prob_logit


# In[222]:


def logit_predict_class(X, b, cutoff, fit_intercept=1):
    predictProb = logit_predict_prob(X, b, fit_intercept=1)
    predictClass = [0 if x <= cutoff else 1 for x in predictProb]
    return predictClass


# In[223]:


predict_class_logit = logit_predict_class(X_test_logit, b_logit, cutoff_logit, fit_intercept=1)
accuracy_logit = metrics.accuracy_score(y_test, predict_class_logit)
accuracy_logit


# In[224]:


cnf_matrix(y_test, predict_class_logit)


# ### (b) Stepwise Logistic Regression

# In[225]:


# cutoff value is obtained from KS test
cutoff_swlogit


# In[226]:


# predict out of sample default probability 
y_pred_swlogit = logit_predict_prob(X_test_swlogit, b_swlogit, fit_intercept=1)
d_swlogit = {'Default Prob - Stepwise Logistic Regression': y_pred_swlogit}
df_prob_swlogit = pd.DataFrame(data=d_swlogit)
df_prob_swlogit


# In[227]:


predict_class_swlogit = logit_predict_class(X_test_swlogit, b_swlogit, cutoff_swlogit, fit_intercept=1)
accuracy_swlogit = metrics.accuracy_score(y_test, predict_class_swlogit)
accuracy_swlogit


# In[228]:


cnf_matrix(y_test, predict_class_swlogit)


# ### (c) Decision Tree

# In[229]:


# cutoff value is obtained from KS test
y_pred_dt_insample = opt_tree.predict_proba(X_train)[:,1]
cutoff_dt, KS_dt = KS_test(y_train, y_pred_dt_insample)


# In[230]:


# predict out of sample default probability 
y_pred_dt = opt_tree.predict_proba(X_test)[:,1]
d_dt = {'Default Prob - Decision Tree': y_pred_dt}
df_prob_dt = pd.DataFrame(data=d_dt)
df_prob_dt


# In[231]:


predict_class_dt = (opt_tree.predict_proba(X_test)[:,1] >= cutoff_dt).astype(bool)
accuracy_dt = metrics.accuracy_score(y_test, predict_class_dt)
accuracy_dt


# In[232]:


cnf_matrix(y_test, predict_class_dt)


# ### (d) Random Forest

# In[233]:


# cutoff value is obtained from KS test
y_pred_rf_insample = rf_opt.predict_proba(X_train)[:,1]
cutoff_rf, KS_rf = KS_test(y_train, y_pred_rf_insample)


# In[234]:


# predict out of sample default probability 
y_pred_rf = rf_opt.predict_proba(X_test)[:,1]
d_rf = {'Default Prob - Decision Tree': y_pred_rf}
df_prob_rf = pd.DataFrame(data=d_rf)
df_prob_rf


# In[235]:


predict_class_rf = (rf_opt.predict_proba(X_test)[:,1] >= cutoff_rf).astype(bool)
accuracy_rf = metrics.accuracy_score(y_test, predict_class_rf)
accuracy_rf


# In[236]:


cnf_matrix(y_test, predict_class_rf)


# ## 2) Brier Score

# In[237]:


# Brier score calculates the mean squared error between predicted probabilities and the expected values
# The error score is always between 0 and 1, where a model with perfect skill has a score of 0

def Brier_Score(actuals, preds):
    n = float(len(preds))
    return (1/n) * np.sum((preds-actuals)**2)


# ### (a) Logistic Regression

# In[238]:


BrierScore_logit = Brier_Score(y_test, y_pred_logit)
BrierScore_logit


# ### (b) Stepwise Logistic Regression

# In[239]:


BrierScore_swlogit = Brier_Score(y_test, y_pred_swlogit)
BrierScore_swlogit


# ### (c) Decision Tree

# In[240]:


BrierScore_dt = Brier_Score(y_test, y_pred_dt)
BrierScore_dt


# ### (d) Random Forest

# In[241]:


BrierScore_rf = Brier_Score(y_test, y_pred_rf)
BrierScore_rf


# ## 3) Hosmer–Lemeshow (HL) Test

# In[242]:


# The Hosmer-Lemeshow goodness of fit test is based on dividing the sample up according to their predicted prob.
# To calculate how many Y=1 observations we would expect, the HL test takes the average of the predicted prob.
# in the group, and multiplies this by the number of observations in the group
# The test also performs the same calculation for Y=0, and then calculates a Pearson goodness of fit statistic

# The test statistic approximately followed a chi-squared distribution on g-2 degrees of freedom

# The p-value can be calculated as the right hand tail probability of the corresponding chi-squared distribution 
# using the calculated test statistic. If the p-value is small, this is indicative of poor fit

def HL_Test(data, q):
    # arrange the dataset ranking by Prob.
    data = data.sort_values('Prob')
    # divide the dataset into q groups on the basis of the deciles
    data['Prob_decile'] = pd.qcut(data['Prob'], q, duplicates='drop')
    # number of defaults(1s) in each group
    y_default = data['Target'].groupby(data['Prob_decile']).sum()
    # total number in each group
    y_total = data['Target'].groupby(data['Prob_decile']).count()
    # number of non-defaults(0s) in each group
    y_nondefault = y_total - y_default
    
    # sum the default probs in each group
    prob_default = data['Prob'].groupby(data['Prob_decile']).sum()
    # total prob in each group
    prob_total = data['Prob'].groupby(data['Prob_decile']).count()
    # the non-default probs in each group
    prob_nondefault = prob_total - prob_default
    
    # Hosmer-Lemeshow (chi-squared) test statistic
    hl_test = ( ((y_default-prob_default)**2 / prob_default) + ((y_nondefault-prob_nondefault)**2 / prob_nondefault) ).sum()
    
    # p-value
    p_val = 1 - stats.chi2.cdf(hl_test, q-2)
    degree_freedom = q-2
    
    print('\n HL-chi2 (df={}): {}, \n p-value: {}\n'.format(degree_freedom, hl_test, p_val))
    return hl_test, p_val


# ### (a) Logistic Regression

# In[243]:


# create a dataframe containing two columns (y_test & y_pred_logit)
d_logit = {'Target': y_test, 'Prob': y_pred_logit}
df_HL_logit = pd.DataFrame(data=d_logit)
df_HL_logit = df_HL_logit.reset_index()
df_HL_logit = df_HL_logit.drop(['index'], axis=1)

HL_logit, HL_pvalue_logit = HL_Test(df_HL_logit, 10)


# ### (b) Stepwise Logistic Regression

# In[244]:


# create a dataframe containing two columns (y_test & y_pred_swlogit)
d_swlogit = {'Target': y_test, 'Prob': y_pred_swlogit}
df_HL_swlogit = pd.DataFrame(data=d_swlogit)
df_HL_swlogit = df_HL_swlogit.reset_index()
df_HL_swlogit = df_HL_swlogit.drop(['index'], axis=1)

HL_swlogit, HL_pvalue_swlogit = HL_Test(df_HL_swlogit, 10)


# ### (c) Decision Tree

# In[254]:


# create a dataframe containing two columns (y_test & y_pred_dt)
d_dt = {'Target': y_test, 'Prob': y_pred_dt}
df_HL_dt = pd.DataFrame(data=d_dt)
df_HL_dt = df_HL_dt.reset_index()
df_HL_dt = df_HL_dt.drop(['index'], axis=1)

HL_dt, HL_pvalue_dt = HL_Test(df_HL_dt, 10)


# ### (d) Random Forest

# In[246]:


# create a dataframe containing two columns (y_test & y_pred_rf)
d_rf = {'Target': y_test, 'Prob': y_pred_rf}
df_HL_rf = pd.DataFrame(data=d_rf)
df_HL_rf = df_HL_rf.reset_index()
df_HL_rf = df_HL_rf.drop(['index'], axis=1)

HL_rf, HL_pvalue_rf = HL_Test(df_HL_rf, 10)


# ## 4) ROC Curves

# ### (a) Logistic Regression

# In[247]:


AUC_logit = ROC_plot(y_test, y_pred_logit)


# ### (b) Stepwise Logistic Regression

# In[248]:


AUC_swlogit = ROC_plot(y_test, y_pred_swlogit)


# ### (c) Decision Tree

# In[249]:


AUC_dt = ROC_plot(y_test, y_pred_dt)


# ### (d) Random Forest

# In[250]:


AUC_rf = ROC_plot(y_test, y_pred_rf)


# ## 5) Final Model Recommendation

# In[251]:


df_result = pd.DataFrame(columns = ['Logistic','Stepwise Logistic','Decision Tree','Random Forest'], 
                         index=['Accuracy Score','Brier Score','HL Test P-value', 'AUC'])

def model_comparison():
    # Accuracy_score
    accuracy_list = [accuracy_logit, accuracy_swlogit, accuracy_dt, accuracy_rf]
    df_result.loc['Accuracy Score'] = accuracy_list
    
    # Brier score
    brier_score_list = [BrierScore_logit, BrierScore_swlogit, BrierScore_dt, BrierScore_rf]
    df_result.loc['Brier Score'] = brier_score_list
    
    # Brier score
    HL_list = [HL_pvalue_logit, HL_pvalue_swlogit, HL_pvalue_dt, HL_pvalue_rf]
    df_result.loc['HL Test P-value'] = HL_list
    
    # AUC
    AUC_list = [AUC_logit, AUC_swlogit, AUC_dt, AUC_rf]
    df_result.loc['AUC'] = AUC_list
    
    return df_result


# In[252]:


model_comparison()


# In[ ]:




