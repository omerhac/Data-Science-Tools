import numpy as np

def mean_encode_categorical(df, target_col):
    """Mean encode categorical features with epanding mean. nans will be replaced with mean of column. df: pd dataframe, target_col: target column to get values from"""
    
    df_with_mean = df.copy()
    for column in df.columns:
        if (df[column].nunique() <= 60) & (column != target_col): # only features with less than 60 unique values will be encoded
            gb_col = df.groupby([column]) # groupby categorical feature
            cumsum = gb_col[target_col].cumsum() - df[target_col] # sum target var uptill now (without this row)
            df_with_mean[column] = cumsum / gb_col.cumcount() # mean encode feature
            df_with_mean[column].fillna(df_with_mean[column].mean(), inplace=True) # fill nans with means
            
            
    return df_with_mean

def test_mean_encode_categorical(df_test, df_train, target_col):
    """Mean encode categorical features with epanding mean. nans will be replaced with mean of column. df: pd dataframe, target_col: target column to get values from"""
    
    df_with_mean = df_test.copy()
    for column in df_test.columns:
        if (df_test[column].nunique() <= 60) & (column != target_col): # only features with less than 60 unique values will be encoded
            gb_col = df_train.groupby([column]) # groupby categorical feature
            means = gb_col[target_col].mean()
            df_with_mean[column] = df_with_mean[column].map(means)
            df_with_mean[column].fillna(df_with_mean[column].mean(), inplace=True) 
            
    return df_with_mean

def print_outliers(df):
    "Print all numeric columns in boxplots. allows to see outliers. Dependencies: matplotlib.pyplot as plt, seaborn as sns"
    
    reduce_df = df.select_dtypes(np.number)
    num_plots = len(reduce_df.columns)
    num_rows = num_plots / 4 + 1
    plt.figure(figsize=(20,10 * num_rows)) # define the figure
    
    for i in range (num_plots): # print each numric col
        plt.subplot(num_rows, 4, i + 1)
        sns.boxplot(reduce_df[reduce_df.columns[i]])
    
    plt.show()

def print_null_places(df):
    """Print two graphs that shows how nans are placed across the data"""
    
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(df.isnull().sum(axis=0)) # columnwise nulls
    plt.title("columnwise nulls")

    plt.subplot(1,2,2)
    plt.plot(df.isnull().sum(axis=1), ) # rowwise nulls
    plt.title("rowwise nulls")
    
    plt.show()

def replace_numeric_columns_nulls(df):
    """Replaces dataframe numeric column nulls with columns means / 0 if there is no mean"""
    
    numeric_cols = df.select_dtypes(np.number).columns
    filled_df = df.copy()
    for col in numeric_cols:
        filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
    filled_df = filled_df.fillna(0)
    return filled_df

def get_rows_above_threshold(df, column, threshold):
    """Gets rows from a dataframe which columns values are bigger than threshold"""
    
    return df.loc[df[column] > threshold, column]

def get_correlated_features(df, feature, threshold_corr):
    """Gets features with pearson correlation greater than threshold. args --> df: pd dataframe, feature: feature name to get correlated features for, threshold_corr: correlation threshold"""
    
    corr_feats = []
    for feat in df.columns:
        if (abs(df[feature].corr(df[feat])) > threshold_corr) & (feat != feature):
            corr_feats.append(feat)
    
    return corr_feats

def get_correlation_graph(df, threshold):
    """Creats an undirectd graph of feature correlation. If two features correlation > threshold they will be connected with an edge"""
    
    corr_graph = Graph()
    
    for feat in df.columns:
        if not corr_graph.has_node(feat):
            corr_feats = get_correlated_features(df, feat, threshold)
            for corr_feat in corr_feats:
                corr_graph.add_edge((feat, corr_feat))
    
    return corr_graph

class Graph():
    """Undireced graph class"""
    
    def __init__(self):
        self._dict = {}
    
    def add_node(self, node):
        if node not in self._dict:
            self._dict[node] = set()
            
    def add_edge(self, edge):
        """Adds an edge. If node didn't exist in the graph, adds it."""
        
        (node1, node2) = edge
        if node1 not in self._dict:
            self._dict[node1] = set([node2])
        else:
            self._dict[node1].add(node2)
        if node2 not in self._dict:
            self._dict[node2] = set([node1])
        else:
            self._dict[node2].add(node1)
    
    def has_edge(self, edge):
        (node1, node2) = edge
        return (node2 in self._dict[node1])
    
    def has_node(self, node):
        return node in self._dict
    
    def get_edges(self, node):
        if self.has_node(node):
            return self._dict[node]
        else:
            return None
    
    def print(self):
        print(self._dict)
    
def get_uncorrolated_features(df, threshold):
    """Gets all the features whic correlation is smaller than threshold.
    Basiclly it returns [feature if feature_correlation_with_all_other_features < threshold]. 
    All features have to be numerical of encoded"""
    
    feats = list(df.columns)
    corr_graph = get_correlation_graph(df, threshold)
    un_corr_feats = []
    
    for feat in feats:
        un_corr_feats.append(feat)
        corr_feats = corr_graph.get_edges(feat)
        if(corr_feats):
            for to_del in corr_feats:
                if to_del in feats:
                    feats.remove(to_del)
    
    return un_corr_feats
  
    
def lgbm_eval(df,target_col, n_splits=1, test_size=0.20, verbose=False, get_model=False, n_rounds=200
              , lgb_params={
               'feature_fraction': 0.75,
               'metric': 'auc',
               'nthread':4, 
               'min_data_in_leaf': 2**7, 
               'bagging_fraction': 0.75, 
               'learning_rate': 0.01, 
               'objective': 'binary', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**7,
               'bagging_freq':1,
               'verbose':0,
               'early_stopping_rounds': 100
    }):
    
    """Evaluates default parameters lgbm model's auc, uses stratified shuffle split. args --> df: pd dataframe, n_splits: int number of data splits,
    test_size: each splits test data precentage, verbose: bool verbosity""" 
    
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.metrics import roc_auc_score
    
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    X_train = df.drop([target_col], axis=1)
    Y_train = df[target_col]
    
    loss = 0
    
    for train_indices, test_indices in sss.split(X_train,Y_train):
        X_train_enc = mean_encode_categorical(df.iloc[train_indices], target_col).drop([target_col], axis=1) # mean encode train set
        X_test_enc = test_mean_encode_categorical(df.iloc[test_indices].drop([target_col], axis=1), df.iloc[train_indices], target_col) # mean encode test set
        model = lgb.train(lgb_params, lgb.Dataset(X_train_enc, label=Y_train.iloc[train_indices]), n_rounds, valid_sets=lgb.Dataset(X_test_enc, Y_train.iloc[test_indices]),
                          verbose_eval=verbose) # train model
        loss += roc_auc_score(Y_train.iloc[test_indices], model.predict(X_test_enc)) # accumulate loss
        
         # free up spaxe
        del X_train_enc, X_test_enc
        gc.collect()
        
    if get_model:
        return (loss / sss.get_n_splits(), model) # avarage loss, model
    else:
        return loss / sss.get_n_splits() # loss
    

def print_feature_correlation(df):
    """Print feature correlation"""
    
    plt.figure(figsize=(30,20))
    sns.heatmap(df.corr())
    plt.show()
    

def normalize_dataframe(df):
    """Normalize dateframe numeric columns. Will not normalize binary columns. Dataframe has to be numeric""" 
    
    df_norm = df.copy()
    for column in df.columns:
        if (list(df[column].unique()) != [0,1]) & (list(df[column].unique()) != [1,0]): # column in not part of one-hot-vector
            c_range = df[column].max() - df[column].min() # range of column
            df_norm[column] = (df[column] - df[column].mean()) / c_range
    
    return df_norm


def explore_target(df, target_column):
    """Print target balance and distirbution"""
    
    plt.figure(figsize=(30,60))
    # target balnce
    plt.subplot(2,1,1)
    plt.hist(df[target_column])
    plt.title("Target balance:")
    
    # target distirbution
    plt.subplot(2,1,2)
    sns.stripplot(data= df, x="TARGET",y=range(len(df)))
    plt.title("Target distirbution")


def nn_classifier(input_shape):
    """Build a nural network classifier"""
    
    from keras.models import Sequential
    from keras.layers import InputLayer, Dense, Dropout
    import keras
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

    model.compile(optimizer='adam', 
                  loss=keras.losses.BinaryCrossentropy(from_logits=False), metrics=["accuracy"])
    return model

class Data():
    """An high level data class.
    
    Supports getting train / test set and encoding
    """
    
    def __init__(self, train_set, test_set, target_column, unique_coulmn):
        """Init a data structure.
        
        Positional arguments:
        train_set -- pandas dataframe for training
        test_set -- pandas dataframe for testing
        target_column -- target column of train set
        unique_column -- the identifier coloumn of the data ""ID""
        """
        
        self._train_set = train_set
        self._test_set = test_set
        self._target = target_column
        self._id = unique_column
    
    def get_all(self):
        """Returns all the data without target column"""
        
        return self._train_set.drop([self._target], axis=1).append(self._test_set).reset_index(drop=True)
    
    def get_train(self):
        """Returns train set with target column"""
        
        return self._train_set.copy()
    
    def get_test(self):
        """Return test set"""
        
        return self._test_set.copy()
    
    def get_train_enc(self):
        """Returns train set mean encoded with target"""
        
        return mean_encode_categorical(self._train_set, self._target)
    
    def get_test_enc(self):
        """Return test set mean encoded"""
        
        return test_mean_encode_categorical(self._test_set, self._train_set, self._target)
    
    def get_all_enc(self):
        """Returns all data mean encoded"""
        
        return self.get_train_enc().drop([self._target], axis=1).append(self.get_test_enc()).reset_index(drop=True)
    
    def add_features(self, df_with_features):
        """Adds features from dataframe df to our data. df has to have the unique column"""
        
        self._train_set = self._train_set.join(df_with_features, on=self._id)
        self._test_set = self._test_set.join(df_with_features, on=self._id)

def change_aggragate_column_names(aggs, prefix):
    """Changes aggragates columns names from tuples to strings."""
    
    aggs.columns = pd.Index([prefix+ "_" + e[0] + "_" + e[1].upper() for e in aggs.columns.tolist()])
    
def get_numeric_columns_aggragates(df, gp_column, prefix, to_agg=['min', 'max', 'mean', 'sum', 'var']):
    """Returns numerical columns aggragated features for dataframe grouped by gb_column"""
    
    numeric_columns = df.select_dtypes(np.number).columns # get numeric columns
    if gp_column in numeric_columns:
        numeric_columns = numeric_columns.drop([gp_column])
    aggs = {column: to_agg for column in numeric_columns}
    gp_aggs = df.groupby([gp_column]).agg(aggs)
    change_aggragate_column_names(gp_aggs, prefix) # tidy up columns names
    return gp_aggs

def get_bayesian_hp_space():
    """Returns a dictionary with """
    from hyperopt import hp
    space = {
    'boosting_type': hp.choice('boosting_type', 
                                            [{'boosting_type': 'gbdt', 'subsample': hp.uniform('gdbt_subsample', 0.5, 1)}, 
                                             {'boosting_type': 'dart', 'subsample': hp.uniform('dart_subsample', 0.5, 1)},
                                             {'boosting_type': 'goss', 'subsample': 1.0}]),
    'num_leaves': hp.quniform('num_leaves', 20, 150, 1),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
    'min_child_samples': hp.quniform('min_child_samples', 20, 500, 5),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
    'colsample_bytree': hp.uniform('colsample_by_tree', 0.6, 1.0),
    'is_unbalance': hp.choice('is_unbalance', [True, False]),
    }
    return space
