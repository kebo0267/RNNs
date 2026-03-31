import base64
import json
from datetime import datetime, timedelta
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras as ks
import tensorflow as tf

from sklearn.datasets import make_friedman1
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import KNNImputer, SimpleImputer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.utils import class_weight

class DataProcessing:
    def __init__(self):
        self.df = None
        self.df_orig = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target:str = None
        self.dateFormat = "%Y-%m-%d %H:%M:%S"
   
    def loadFromJson(self,filename):
        self.df_orig = pd.read_json(filename)
        self.reset_processor()

    def loadFrom_CSV(self,filename):
        self.df_orig = pd.read_csv(filename)
        self.reset_processor()

    def reset_processor(self):
        self.df = self.df_orig.copy()

    def set_target(self,target:str):
        self.target = target

    def get_train_features(self):
        return self.X_train

    def get_train_target(self):
        return self.y_train
    
    def get_test_features(self):
        return self.X_test
    
    def get_test_target(self):
        return self.y_test
    
    def get_data(self):
        return self.df

    def get_original(self):
        return self.df_orig
    
    def get_info(self):
        return self.df.info()
    
    def get_mean(self,parameter):
        ret_val = None
        if (not self.df[parameter].dtype is object):
            ret_val = self.df[parameter].mean()

        return ret_val;

    def get_min(self,parameter):
        ret_val = None
        if (not self.df[parameter].dtype is object):
            ret_val = self.df[parameter].min()

        return ret_val;

    def get_max(self,parameter):
        ret_val = None
        if (not self.df[parameter].dtype is object):
            ret_val = self.df[parameter].max()

        return ret_val;

    def get_std(self,parameter):
        ret_val = None
        if (not self.df[parameter].dtype is object):
            ret_val = self.df[parameter].std()

        return ret_val;

    def describe(self,parameter):
        ret_val = None
        if (parameter in self.df.columns):
            if (not self.df[parameter].dtype is object):
                ret_val = self.df[parameter].describe()

        return ret_val


    def locateMissingData(self):
        missing_data = {}
        if not self.df is None:
            df_size = len(self.df)
            for col in self.df.columns:
                if (self.df[col].dtype == object):
                    if len(self.df[self.df[col].astype(str).str.strip().eq("")]) > 0:
                        missing_data[col] = self.df[self.df[col].astype(str).str.strip().eq("")].index.tolist()
                else:
                    if (len(self.df[self.df[col].isna()] == True) > 0) or (len(self.df[self.df[col].isnull()] == True) > 0):
                        null_indexes = set()
                        null_indexes.update(self.df[self.df[col].isna()].index.tolist())
                        null_indexes.update(self.df[self.df[col].isnull()].index.tolist())
                        missing_data[col] = null_indexes
        return missing_data
    

    def getMissingValues(self, columns:list):
        missing_list = self.df.select_dtypes(include=[np.number]).columns.tolist()
        #imputer = KNNImputer(n_neighbors=2)
        print(missing_list)
        imputer = SimpleImputer(strategy="mean")
        self.df[missing_list] = imputer.fit_transform(self.df[missing_list])
        
    def processNumericDataTypes(self):
        # Change data types to minimum required to represnt data
        if not self.df is None:
            for col in self.df.columns:
                if self.df[col].dtype == "int64":
                    if (self.df[col].max() <= np.iinfo(np.int8).max) and (self.df[col].min() >= np.iinfo(np.int8).min):
                        self.df[col] = self.df[col].astype("int8")
                    elif (self.df[col].max() <= np.iinfo(np.int16).max) and (self.df[col].min() >= np.iinfo(np.int16).min):
                      self.df[col] = self.df[col].astype("int16")
                elif self.df[col].dtype == "float64":
                    if (self.df[col].max() <= np.finfo(np.float32).max) and (self.df[col].min() >= np.finfo(np.float32).min):
                        self.df[col] = self.df[col].astype("float32")

    def processBinaryDataTypes(self,parameter:str,options:dict =None):
        if not self.df is None:

            if self.df[parameter].dtype == object:
                unique_values = self.df[parameter].unique()
                if (len(unique_values) == 2):
                    choice_map = {}
                    if options == None:
                        choice_map[unique_values[0]] = False
                        choice_map[unique_values[1]] = True
                    else:
                        choice_map = options

                    self.df[parameter] = self.df[parameter].map(choice_map).astype("bool")
                else:
                    print(f'{parameter} is not a boolean feature')



    def encodeObjectDataTypes(self,excluded_parameters=None):
        #Binary encode
        if not self.df is None:
            df_columns = self.df.columns
            for col in df_columns:
                if col in excluded_parameters:
                    continue
                if self.df[col].dtype == object:
                    unique_values = self.df[col].unique()
                    if (len(unique_values) == 2):
                        self.processBinaryDataTypes(col)
                    else:
                        if (len(unique_values)/len(self.df[col]) < 0.15):
                            encoder = OneHotEncoder(sparse_output=False,handle_unknown="ignore").set_output(transform='pandas')
                            encode_data = encoder.fit_transform(self.df[[col]])
                            self.df = pd.concat([self.df, encode_data],axis=1).drop(columns=[col])

    def encodeDateTimeParameters(self,parameter,format='%d/%m/%Y %H'):
        if not self.df is None:
            #self.df[parameter] = pd.to_datetime(self.df[parameter],format=format)
            dt_parameter = pd.to_datetime(self.df[parameter],format=format)
            self.df[f'{parameter}_Year'] = dt_parameter.dt.year.astype("int16")
            self.df[f'{parameter}_Month'] = dt_parameter.dt.month.astype("int16")
            self.df[f'{parameter}_Day'] = dt_parameter.dt.day.astype("int16")
            self.df[f'{parameter}_hour'] = dt_parameter.dt.hour.astype("int16")
            self.df[f'{parameter}_Min'] = dt_parameter.dt.minute.astype("int16")
            self.df[f'{parameter}_sec'] = dt_parameter.dt.second.astype("int16")
            self.df[f'{parameter}_micro'] = dt_parameter.dt.microsecond.astype("int16")

    def processWithStandardScaler(self,scale_features:list=None,inplace:bool=False):
        scale_columns = []
        if scale_features is None:            
            for col in self.df.columns:
                if (str(self.df[col].dtype).startswith("float")) or (str(self.df[col].dtype).startswith("int")):
                    if col != self.target:
                        scale_columns.append(col)
        else:
            scale_columns = scale_features
        
        scaler = StandardScaler()
        if (not self.X_train is None):
            scaled_data = pd.DataFrame(scaler.fit_transform(self.X_train[scale_columns]),
                                       columns=scale_columns,
                                       index=self.X_train.index)
            self.set_scaled_data(self.X_train,scaled_data,inplace,scale_columns=scale_columns)

        if (not self.X_test is None):
            scaled_data = pd.DataFrame(scaler.fit_transform(self.X_test[scale_columns]),
                                       columns=scale_columns,
                                       index=self.X_test.index)
            self.set_scaled_data(self.X_test,scaled_data,inplace,scale_columns=scale_columns)

        if (self.X_train is None) and (not self.df is None):
            scaled_data = pd.DataFrame(scaler.fit_transform(self.df[scale_columns]),
                                       columns=scale_columns,
                                       index=self.df.index)
            self.set_scaled_data(self.df,scaled_data,inplace,scale_columns=scale_columns)

        
    def set_scaled_data(self,data:pd.DataFrame,scaled_data:pd.DataFrame,inplace:bool=False,scale_columns:list =None):
        if inplace == True:
            data[scaled_data.columns] = scaled_data
        else:
            rename = {}
            for col in scaled_data.columns:
                rename[col] = f'{col}_T'
            scaled_data.rename(columns=rename)
            data = pd.concat([data,scaled_data],axis=1)

        
        #self.scaled_df = pd.DataFrame(scaled_data,columns=scale_columns)

    def get_groupby_data(self,categories=None):
        ret_value = self.get_copy()
        if not categories is None:
           ret_value = ret_value.groupby(categories)

        return ret_value
    
    def get_histogram_plot(self,features:list=None,target:str=None,ignore:list=None):
        # Visualize
        if features is None:
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            features.remove(self.target)
            #Remove any features that are also in the ignore list
            if ignore is not None:
                features = [feature for feature in features if feature not in ignore]   
                
            #Remove any features that have two or fewer unique values as these are likely to be binary features
            features = [feature for feature in features if len(self.df[feature].unique()) > 2]
        num_features = len(features)
        cols = 3
        # Calculate number of rows needed
        rows = (num_features // cols) + (num_features % cols > 0)
        
        plt.figure(figsize=(15, 5 * rows))
        for i, feature in enumerate(features):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(self.df[feature], kde=True, bins=30)
            plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()

    def get_scatter_plot(self,features:list=None,target:str=None,ignore:list=None):
        # Visualize

        if features is None:
            features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            features.remove(self.target)
            #Remove any features that are also in the ignore list
            if ignore is not None:
                features = [feature for feature in features if feature not in ignore]   
        num_features = len(features)
        cols = 3
        # Calculate number of rows needed
        rows = (num_features // cols) + (num_features % cols > 0)
        
        plt.figure(figsize=(15, 5 * rows))

        #Create a scatter plot of features vs the other features in the list with the hue set to the target variable if it is not None
        if self.target is not None:
            features2 = features.copy()
            for i, feature in enumerate(features):
                #Remove feture from features2 list so that it is not plotted against itself
                features2.remove(feature)
                for j, feature2 in enumerate(features2):
                    plt.subplot(rows, cols, i + 1)
                    sns.scatterplot(data=self.df, x=feature, y=feature2)
                    plt.title(f'Distribution of {feature} vs {feature2}')        
    
        plt.tight_layout()
        plt.show()

    def get_image_plot(self):
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        axes[0].set_title('Original distribution')
        axes[0].hist(df[feature], bins=50, edgecolor='black', color='grey')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Frequency')

        axes[1].set_title('Square root-transformed distribution')
        axes[1].hist(df[f'{feature}_sqrt'], bins=50, edgecolor='black', color='grey')
        axes[1].set_xlabel(f'{feature}_sqrt')
        axes[1].set_ylabel('Frequency')



        plt.tight_layout()
        plt.show()

    def set_train_test_data(self, train_split:float=0.8, random_state:int=315):
 
        dataset = self.df.copy()
    
        # Assume this is a categorization problem
        # Get data from self.df and remove any non-numeric columns
        dataset = dataset.select_dtypes(include=[np.number])
        # Assign target variable to y if it is in the dataset and remove it from the dataset
        y = None
        if self.target in dataset.columns:
            y = dataset[self.target]
            dataset = dataset.drop(self.target,axis=1)
            
        X = dataset.copy()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y,train_size=train_split,random_state=random_state)
        
            

    def plot_confusion_matrix(self, y_true, y_pred, 
                         labels=None, 
                         title="Confusion Matrix",
                         cmap="Blues",
                         figsize=(8, 6),
                         normalize=False,
                         annot=True,
                         fmt='.2f'):
        if normalize:
            fmt = 'd'
        """
        Creates a nicely styled confusion matrix using Seaborn.
        
        Parameters:
        -----------
        y_true : array-like
            True labels (0 = Fully Paid, 1 = Not Fully Paid)
        y_pred : array-like
            Predicted labels
        labels : list, optional
            Label names for classes. Default: ['Fully Paid', 'Not Fully Paid']
        title : str
            Title of the plot
        cmap : str
            Colormap for the heatmap (e.g., "Blues", "Greens", "Reds", "YlOrBr")
        figsize : tuple
            Figure size (width, height)
        normalize : bool
            Whether to show percentages (True) or raw counts (False)
        annot : bool
            Whether to annotate cells with numbers
        fmt : str
            String formatting for annotations
        """
        
        # Default labels for your loan dataset
        if labels is None:
            labels = ['Fully Paid (0)', 'Not Fully Paid (1)']
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot heatmap
        ax = sns.heatmap(cm, 
                        annot=annot, 
                        fmt=fmt, 
                        cmap=cmap, 
                        linewidths=0.5, 
                        linecolor='gray',
                        xticklabels=labels,
                        yticklabels=labels,
                        cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
        
        # Add title and labels
        plt.title(title, fontsize=16, pad=20)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        
        # Rotate x labels if needed
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Add total counts in title if not normalized
        if not normalize:
            total = len(y_true)
            correct = (y_true == y_pred).sum()
            accuracy = correct / total * 100
            plt.suptitle(f'Accuracy: {accuracy:.2f}%', y=0.95, fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        return ax 
        

    def build_model(self,input_shape, l1_activation, l2_activation, l3_activation, dropout_rate) -> ks.Model:
        inputs = ks.Input(shape=input_shape)  # Input layer with the number of features
        x = ks.layers.Flatten()(inputs)  # Flatten the input if necessary
        x = ks.layers.Dense(128, activation=l1_activation)(x)
        x = ks.layers.Dropout(dropout_rate)(x)  # Add dropout for regularization
        x = ks.layers.Dense(64, activation=l2_activation)(x)  # Hidden layer
        x = ks.layers.Dropout(dropout_rate)(x)  # Add dropout for regularization
        x = ks.layers.Dense(128, activation=l3_activation)(x)  # Another hidden layer
        x = ks.layers.Dropout(dropout_rate)(x)  # Add dropout for regularization
        x = ks.layers.Dense(32, activation='relu')(x)  # Another hidden layer
        x = ks.layers.Dropout(dropout_rate)(x)  # Add dropout for regularization
        x = ks.layers.Dense(16, activation='relu')(x)  # Another hidden layer
        x = ks.layers.Dropout(dropout_rate)(x)  # Add dropout for regularization
        outputs = ks.layers.Dense(1, activation='sigmoid')(x)  # Output layer for binary classification
        model = ks.Model(inputs=inputs, outputs=outputs)
        return model

    
    def objective(self, trial, build_model):
        #, X_train, y_train, X_test, y_test
        # Define the hyperparameters to optimize
        test_metrics = ['accuracy', 'Precision', 'Recall', 'f1_score']

        learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
        weight_decay = trial.suggest_categorical('weight_decay', [1e-5, 1e-4, 1e-3])
        batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64, 128])
        epochs = 100
        epsilon = trial.suggest_categorical('epsilon', [1e-8, 1e-7, 1e-6])
        l1_activation = trial.suggest_categorical('l1_activation', ['relu', 'tanh', 'sigmoid'])
        l2_activation = trial.suggest_categorical('l2_activation', ['relu', 'tanh', 'sigmoid'])
        l3_activation = trial.suggest_categorical('l3_activation', ['relu', 'tanh', 'sigmoid'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.3, 0.5, step=0.1)

        # Build the model with the suggested hyperparameters
        model = build_model(input_shape=self.X_train.shape[1:], l1_activation=l1_activation, l2_activation=l2_activation, l3_activation=l3_activation, dropout_rate=dropout_rate)

        # Compile the model with the suggested hyperparameters
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay, amsgrad=True, name='AdamW', epsilon=epsilon)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=test_metrics)
        
        # Compute class weights to handle class imbalance
        class_weights_computed = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(self.y_train.flatten()), y=self.y_train.flatten())
        class_weights_dict = dict(enumerate(class_weights_computed))
        # Create two more class weights, based off of the computed weights. One to diverge and one to converge, and add that into the optuna trials
        class_weights_dict_diverge = {0: class_weights_computed[0] * 1.5, 1: class_weights_computed[1] * 0.5}
        class_weights_dict_converge = {0: class_weights_computed[0] * 0.5, 1: class_weights_computed[1] * 1.5}
        class_0_weight = trial.suggest_categorical('class_0_weight', [class_weights_dict[0], class_weights_dict_diverge[0], class_weights_dict_converge[0]])
        class_1_weight = trial.suggest_categorical('class_1_weight', [class_weights_dict[1], class_weights_dict_diverge[1], class_weights_dict_converge[1]])
        class_weights = {0: class_0_weight, 1: class_1_weight}

        # Train the model with the suggested hyperparameters
        history = model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.X_test, self.y_test), verbose=0, class_weight=class_weights)

        # Get the validation f1 score from the training
        f1_score = history.history['val_f1_score'][-1][0]  # Get the last epoch's validation f1 score
        #DEBUG: print out the f1 score type and value for each trial to see how it's progressing
        print(f"Trial {trial.number}: f1_score={f1_score} (type={type(f1_score)})")
        # Cast as float to avoid any issues with optuna expecting a float return value
        f1_score = float(f1_score)
        return f1_score
        


