from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import model_selection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from matplotlib.font_manager import FontProperties

import pickle
import cv2

import numpy as np
import os
import matplotlib.pyplot as plt

from dataset import get_ml_image    

def prepare_train_test_data(img_list, yield_pf, metadata=None, train_col='TRAIN_75', 
                            VI_list=None, suffix_list=None, vi_only=False):
    
    yield_list = list(yield_pf['Yield_Bu_Ac'])
    # yield_list = list(yield_pf['Yield_MT_Ha'])
    train_indices = list(yield_pf[yield_pf[train_col] == 1].index)
    test_indices = list(yield_pf[yield_pf[train_col] == 0].index)

    train_img_list = [img_list[i] for i in train_indices]
    test_img_list = [img_list[i] for i in test_indices]
    
    train_images = get_ml_image(train_img_list, VI_list=VI_list, 
                                suffix_list = suffix_list, is_vi_only=vi_only)
    test_images = get_ml_image(test_img_list, VI_list=VI_list, 
                                suffix_list = suffix_list, is_vi_only=vi_only)
    if metadata:
        train_meta = [metadata[i] for i in train_indices]
        test_meta = [metadata[i] for i in test_indices]
        
        train_images = [train_images, train_meta]
        test_images = [test_images, test_meta]
        
    return train_images, test_images, [yield_list[i] for i in train_indices], [yield_list[i] for i in test_indices]
    
    
'''predict yield using machine learning algorithm'''
def ml_predict_yield(train_images, train_yields, test_images, modelname, out_name, out_path: str, is_Meta = False):
    
    if modelname.upper() == 'LASSO':
        model = Lasso(alpha=0.5)
    elif modelname.upper() == 'LR':
        model = LinearRegression()
    elif modelname.upper() == 'RF':
        model = RandomForestRegressor(n_estimators=120, max_depth=25, random_state=42)
    elif modelname.upper() == 'GB':
        model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    elif modelname.upper() == 'SVR':
        model = LinearSVR()
    elif modelname.upper() == 'XGB':
        model = xgb.XGBRegressor(n_estimators =150, max_depth=25,objective='reg:squarederror', random_state=42)
    # xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    else:
        print('not support')

    trained_model = train_model(model, train_images, train_yields, out_path+out_name+modelname)
        
    
    # Save the model to a file using pickle
    # `model_file_name` is a variable that stores the file name and path where the trained model will
    # be saved using the pickle module. The trained model is saved as a binary file with the extension
    # `.pkl`.
    model_file_name = out_path+out_name+modelname+'.pkl'
    with open(model_file_name, 'wb') as model_file:
        pickle.dump(trained_model, model_file)
        
    return trained_model

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def fit(self, X, y=None):
        #self.mean = X.mean((0,2,3)) 
        #self.std = X.std((0,2,3))
        return self

    def transform(self, X, y=None):
        # return (X-self.mean[None,:,None,None])/self.std[None,:,None,None] 
        return X

'''flatten image into 1d array for machine learning model'''
class FlattenTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.reshape((X.shape[0], -1))

class FlattenMeanTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # use mean value of the image 
        X = np.nanmean(X,(2,3))
        return X.reshape((X.shape[0], -1))


class FlattenHistTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bins=50):
        self.bins = bins
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # use mean value of the image 
        histograms = []
        for image in X:
            image_histograms = []
            for band in range(image.shape[0]):  # Iterate over bands
                hist = cv2.calcHist([image[band,:,:]], [0], None, [self.bins], [0, 1])
                image_histograms.append(hist.flatten())
            histograms.append(np.concatenate(image_histograms))  # Concatenate histograms from all bands
        X = np.nanmean(X,(2,3))
        X = np.concatenate((X , np.array(histograms)),axis=1)    
        # print(X[0].shape)
        return X.reshape((X.shape[0], -1))         

'''train the model by pipeline'''
def train_model(model, train_images, train_yields, save_name:str = None, is_plot:bool = False):
    
    MEAN = np.nanmean(train_images,(0,2,3))
    STD = np.nanstd(train_images, (0,2,3))

    pipe = Pipeline(steps=[("scaler", CustomScaler(MEAN,STD)),
                        # ("flatten", FlattenTransformer()),
                        ("flatten", FlattenMeanTransformer()),
                        ("classifier", model )
                        ])
    pipe.fit(train_images,train_yields)

    # predict train
    pred_train = pipe.predict(train_images)
    
    if is_plot:
        plot_result(train_yields, pred_train, save_name + ' Train')
        save_result(train_yields, pred_train, save_name + ' Train')
    
    return pipe


# def test_model(model, test_images, test_yields, test_indices, irrigate_data, save_name):
    
#     # predict
#     pred_validate = model.predict(test_images)
    
#     save_result(test_yields, pred_validate, save_name )

    
def plot_result(y_test, y_pred, save_name):
    
    basename = os.path.basename(save_name)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    x = np.arange(50,300)
    plt.plot(x, x, color = 'k', ls='--')
    
    plt.scatter( y_test, y_pred, label = f'r-squared = {round(r2,2)}, rmse = {round(rmse, 2)}')
    plt.xlim([50,300])
    plt.ylim([50,300])
    plt.xticks(fontname='Times New Roman', size=10)
    plt.yticks(fontname='Times New Roman', size=10)
    plt.xlabel('True Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)
    plt.ylabel('Predicted Yield (Bu/Ac)', fontname = 'Times New Roman', fontsize = 12)


    plt.title(basename, fontname = 'Times New Roman', fontsize = 14)


    plt.text(55, 288, f'R-squared = {round(r2,2)}', fontname = 'Times New Roman', fontsize = 11)
    plt.text(55, 275, f'RMSE = {round(rmse, 2)} (Bu/Ac)',  fontname = 'Times New Roman', fontsize = 11)
    # yy_pred = model_lasso.predict(X_train)
    # plt.scatter(y_train, yy_pred)
    plt.savefig(save_name + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(save_name, f' r-squared = {r2}, rmse = {rmse}')
    
    
def save_result(y_test, y_pred, save_name):
    
    # Combine arrays horizontally
    # combined_array = np.hstack(( y_pred, y_test))
    combined_array = np.column_stack(( y_test, y_pred))

    # Save combined_array to a CSV file
    # np.savetxt(save_name, combined_array, delimiter=', ', fmt='%d')
    np.savetxt(save_name+'.csv', combined_array, delimiter=' ')

