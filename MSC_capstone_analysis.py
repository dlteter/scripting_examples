import pandas as pd
import country_converter as coco
import sklearn
import pycountry_convert as pc
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import linear_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pingouin import partial_corr
from numpy import inf
from sklearn.compose import TransformedTargetRegressor
import sklearn.metrics as metr
from sklearn.tree import export_graphviz
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV




agfinal = pd.read_csv('latest.csv')
wardata = pd.read_csv('warlatest.csv')

agdata = pd.read_csv('Value_of_Production_E_All_Data_(Normalized).csv', encoding = 'ISO-8859-1')
wardata = pd.read_excel('number_of_disorder_events_by_country-year_as-of-12Sep2020-1.xlsx')

agdata = agdata.loc[(agdata['Year'] >= 1997) & (agdata['Year'] <= 2017)]
agdata = agdata.loc[(agdata['Element'] == 'Gross Production Value (constant 2004-2006 1000 I$)') | (agdata['Element'] == 'Gross Production Value (constant 2004-2006 million SLC)') | (agdata['Element'] == 'Gross Production Value (constant 2004-2006 million US$)')]
testt = agdata.groupby(['Area','Element']).size().reset_index().rename(columns={0:'count'})
testt2 = testt.groupby(['Area']).size().reset_index().rename(columns={0:'count'})
USDlist = testt2.loc[(testt2['count'] == 3)]
Twolist = testt2.loc[(testt2['count'] == 2)]
Onelist = testt2.loc[testt2['count'] == 1]
 
agUSD = agdata.loc[agdata['Area'].isin(USDlist['Area'])]
agUSD = agUSD.loc[agUSD['Unit'] == 'USD']

agother1 = agdata.loc[agdata['Area'].isin(Twolist['Area'])]
agother1USD = agother1.loc[agother1['Unit'] == 'USD']
agother1notusd = agother1.loc[~agother1['Area'].isin(agother1USD['Area'])]
agother1notusd = agother1notusd.loc[agother1notusd['Unit'] == 'SLC']

agalone =  agdata.loc[agdata['Area'].isin(Onelist['Area'])]

agfinal = pd.concat([agUSD,agother1USD,agother1notusd, agalone])
agfinal = agfinal.groupby(['Area', 'Year', 'Element', 'Unit','Item'])['Value'].agg('sum').reset_index()

agfinal['Area'].loc[agfinal['Area'] == 'Belgium-Luxembourg'] = 'Belgium'
agfinal['Area'].loc[agfinal['Area'] == 'Australia & New Zealand'] = 'Australia'
agfinal['Area'].loc[agfinal['Area'] == 'China, Taiwan Province of'] = 'Taiwan'
agfinal['Area'].loc[agfinal['Area'] == 'Congo Republic'] = 'Republic of the Congo'
agfinal['Area'].loc[agfinal['Area'] == 'Cote d\'Ivoire'] = 'Ivory Coast'  
agfinal['Area'].loc[agfinal['Area'] == 'DR Congo'] = 'Democratic Republic of the Congo'
agfinal[['Area']] = agfinal[['Area']].astype(str)

agfinal.dtypes

for index, row in agfinal.iterrows():
    name = row[0]
    shortname = coco.convert(names=name, to='name_short')
    agfinal.loc[index,'Area'] = shortname

agfinal = agfinal.loc[agfinal['Area'] != 'not found']

agfinal.dtypes

for index, row in agfinal.iterrows():
    name = row[0]
    alpha = pc.country_name_to_country_alpha2(name, cn_name_format="default")
    contname = pc.country_alpha2_to_continent_code(alpha)
    agfinal.loc[index,'Continent'] = contname
    
agfinal = agfinal[agfinal['Area'] != 'Faeroe Islands']
agfinal = agfinal[agfinal['Area'] != 'Micronesia, Fed. Sts.']
agfinal = agfinal[agfinal['Area'] != 'Reunion']
agfinal = agfinal[agfinal['Area'] != 'St. Vincent and the Grenadines']
agfinal = agfinal[agfinal['Area'] != 'Timor-Leste']
agfinal = agfinal[agfinal['Area'] != 'Wallis and Futuna Islands']
agfinal = agfinal[agfinal['Area'] != 'World']

agfinal.loc[agfinal['Area'] == 'Zambia',['Continent']] = 'AF'
agfinal.loc[agfinal['Area'] == 'Western Sahara',['Continent']] = 'AF'

agfinal = agfinal[agfinal['Area'] != 'Western Sahara']
agfinal = agfinal.loc[agfinal['Continent'] == 'AF']


agfinal = pd.read_csv('latest.csv')
wardata = pd.read_csv('warlatest.csv')

for index, row in wardata.iterrows():
    name = row[1]
    shortname = coco.convert(names=name, to='name_short')
    wardata.loc[index,'Country'] = shortname
    
        
agfinal.to_csv('latest.csv')
wardata.to_csv('warlatest.csv')

agfest = agfinal.loc[agfinal['Area'] == 'Algeria']
wartest = wardata.loc[wardata['Country'] == 'Algeria']

agfinal['Year'] = pd.to_numeric(agfinal['Year']) 
wardata.dtypes

agsorted.to_csv('frigginreally.csv')
agfinal[2]

for index, row in agfest.iterrows():
    count = 0
    for inx, war in wartest.iterrows():
        print(row[2])
        print(war[2])
        print(row[1])
        print(war[1])
        print(war[3])


for index, row in agfest.iterrows():
    count = 0
    for inx, war in wartest.iterrows():
        if (row[2] == war[2]) & (row[1] == war[1]):
            print(row[2])
            print(war[2])
            print(row[1])
            print(war[1])
            print(war[3])
            agfest.loc[index,'warno'] = war[3]
        else:
            agfest.loc[index,'warno'] = agfest.loc[index,'warno']
    if agfest.loc[index,'warno'] > 0:
        agfest.loc[index,'waryn'] = 1
    else:
        agfest.loc[index,'waryn'] = 0
        

for index, row in agfinal.iterrows():
    count = 0
    agfinal.loc[index,'warno'] = 0
    for inx, war in wardata.iterrows():
        if (row[2] == war[2]) & (row[1] == war[1]):
            agfinal.loc[index,'warno'] = war[3]
        else:
            agfinal.loc[index,'warno'] = agfinal.loc[index,'warno']
    if agfinal.loc[index,'warno'] > 0:
        agfinal.loc[index,'waryn'] = 1
    else:
        agfinal.loc[index,'waryn'] = 0
    print('still going lol')
agsorted = pd.read_csv('intermed.csv')

## calculate % change year over year for country/item groups
agsorted = agfinal.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)
agsorted['Change'] = agsorted.groupby(['Area','Item'], sort=False)['Value'].apply(
     lambda x: x.pct_change()).to_numpy()

agsorted = agsorted.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)
agsorted['Change'] = agsorted.groupby(['Area','Item'], sort=False)['Value'].apply(
     lambda x: x.pct_change()).to_numpy()

agsorted = agsorted.sort_values(['Area', 'Item', 'Year']).reset_index(drop=True)
agsorted['warchange'] = agsorted.groupby(['Area','Item'], sort=False)['warno'].apply(
     lambda x: x.pct_change()).to_numpy()
    
aganalyze = agsorted[['Area','Year','Item','Change','warno','waryn','warchange']]
aganalyze = aganalyze[aganalyze['Change'].notna()]
aganalyze.to_csv('finaldataset.csv')
aganalyze = pd.read_csv('finaldataset.csv')
aganalyze = aganalyze[['Area','Year','Item','Change','warno','waryn','warchange']]
aganalyze['Area'] = aganalyze['Area'].astype('category')
aganalyze['Item'] = aganalyze['Item'].astype('category')
aganalyze['Area_cat'] = aganalyze['Area'].cat.codes
aganalyze['Item_cat'] = aganalyze['Item'].cat.codes
aganalyze['warchange'] = aganalyze['warchange'].replace([np.inf, -np.inf], np.nan)
aganalyze['warchange'] = aganalyze['warchange'].fillna(0)
aganalyze[(np.abs(stats.zscore(aganalyze['Change'])) < 2)]

## data profiling to check for issues and multicollinearity
prof = ProfileReport(aganalyze) 
prof.to_file('output.html')

##prepare training and test sets
dependent = aganalyze['Change']
dependent = dependent.to_numpy()
dependent = dependent.reshape(-1,1)
independentpd = aganalyze.drop(columns=['Change','Area','Item'])

 # Calculating VIF to check for multicollinearity
VIF = pd.DataFrame()
VIF["variables"] = independentpd.columns
VIF["VIF"] = [variance_inflation_factor(independentpd.values, i) for i in range(independentpd.shape[1])]
print(VIF)

independentpd = independentpd.drop(columns=['Year','waryn'])
independent = independentpd.to_numpy()
X_train, X_test, Y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)


spearman = partial_corr(data=aganalyze, x='warchange', y='Change', x_covar=['warno','Area_cat','waryn'], y_covar =['Item_cat','Area_cat'], method='spearman')
spearman = partial_corr(data=aganalyze, x='warno', y='Change', x_covar=['warchange','Area_cat','waryn'], y_covar =['Item_cat','Area_cat'], method='spearman')

residtestmodel =  LinearRegression().fit(X_train, Y_train)
residpredict = residtestmodel.predict(X_test)
residtest = pd.DataFrame()
plotresidpredict = pd.DataFrame(residpredict)
plottest = pd.DataFrame(y_test)
residtest['resid'] = (plottest[0] - plotresidpredict[0])

plt.scatter(plottest,residtest)
plt.xlim(-2, 6)
plt.ylim(-15, 8)
plt.savefig('linearresiduals.png')
plt.clf()


##add constant to dependent variable to prep for log transform
dependent = (aganalyze['Change'] + 10)
dependent = dependent.to_numpy()
dependent = dependent.reshape(-1,1)
independent = aganalyze.drop(columns=['Change','Area','Item','Year','waryn'])
independent = independent.to_numpy()
X_train, X_test, Y_train, y_test = train_test_split(independent, dependent, test_size=0.2, random_state=0)

##log transform dependent variable.
translinear = TransformedTargetRegressor(regressor=LinearRegression(),func=np.log, inverse_func=np.exp)
transrf = TransformedTargetRegressor(regressor=RandomForestRegressor(),func=np.log, inverse_func=np.exp)
translasso = translinear = TransformedTargetRegressor(regressor=linear_model.Lasso(alpha=0.1),func=np.log, inverse_func=np.exp)

##prepare polynomial regression
polyfeat = PolynomialFeatures(degree=5)
pipeline = Pipeline([('p',polyfeat),('l',translasso)])
rfpipeline = Pipeline(steps=[('p',polyfeat),('r',transrf)])

seed = 7
models = []
##append random forest
models.append(('RF',transrf))
##append linear regression
models.append(('LNR',translinear))
##append polynomial regression
models.append(('PLM',pipeline))
models.append(('RPipe',rfpipeline))
##append Lasso, switch to lasso because residuals suggest need for regularization
models.append(('LAS',translasso))

##scoring with 4 folds
output = []
labels = []
scoring = 'explained_variance'
for label, model in models:
    val = model_selection.KFold(n_splits = 4, random_state=seed)
    kfoldresults = model_selection.cross_val_score(model,X_train,Y_train,cv=val,scoring=scoring)
    output.append(kfoldresults)
    labels.append(label)
    read ="%s: %f (%f)" % (label, kfoldresults.mean(), kfoldresults.std())
    print(read)

##scoring wth 10 folds 
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state=seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg="%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    model = ols('Change ~ warno + waryn + Item_cat + Area_cat + Year', data=aganalyze).fit()
fig = plt.figure()
axes = fig.gca()
axes.set_xlim([-1000,1000])
axes.set_ylim([-1000,1000])
fig = sm.graphics.plot_regress_exog(model, 'warno', fig=fig)
fig.savefig('graph.png')
agdata.iloc[500,1]
unique()
print(standard_names)

polynomial_features= PolynomialFeatures(degree=5)
xp = polynomial_features.fit_transform(X_train)
xpred = polynomial_features.fit_transform(X_test)
xp.shape
finalmodel = sm.OLS(Y_train, xp).fit()
finalmodel.summary()

predictions = finalmodel.predict(xpred)
plt.scatter(X_train,Y_train)
fig = plt.scatter(xpred,predictions)


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf_random = RandomizedSearchCV(estimator = rfpipeline, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train, Y_train)
rf_random.best_params_

randommodel = rfpipeline.fit(X_train, Y_train)
rfpredict = randommodel.predict(X_test)
rfresiduals = pd.DataFrame()
plotpredict = pd.DataFrame(rfpredict)
plotpredict[0] = (plotpredict[0] - 10)
plottest = pd.DataFrame(y_test)
plottest[0] = (plottest[0] - 10)
rfresiduals['resid'] = (plottest[0] - plotpredict[0])

plotinput = pd.DataFrame(X_test)
plotoriginput = pd.DataFrame(X_train)
plotorigoutput = pd.DataFrame(Y_train)

##R2 score for model using test data

randommodel.score(X_test,y_test)
evaluate(randommodel,X_test,y_test)

transrfparams = TransformedTargetRegressor(regressor=RandomForestRegressor(n_estimators= 1600,
 min_samples_split= 2,
 min_samples_leaf= 4,
 max_features= 'sqrt',
 max_depth= 80,
 bootstrap= True),func=np.log, inverse_func=np.exp)
rfpipelineparams = Pipeline(steps=[('p',polyfeat),('r',transrfparams)])

tunedmodel = rfpipeline.fit(X_train, Y_train)
rfpredict = tunedmodel.predict(X_test)
rfresiduals = pd.DataFrame()
plotpredict = pd.DataFrame(rfpredict)
plotpredict[0] = (plotpredict[0] - 10)
plottest = pd.DataFrame(y_test)
plottest[0] = (plottest[0] - 10)
rfresiduals['resid'] = (plottest[0] - plotpredict[0])

plotinput = pd.DataFrame(X_test)
plotoriginput = pd.DataFrame(X_train)
plotorigoutput = pd.DataFrame(Y_train)

##R2 score for model using test data

tunedmodel.score(X_test,y_test)
errors = abs(plotpredict - plottest)
m = 100 * np.mean(errors / plottest)
accuracy = 100 - m

print(accuracy)

