
# coding: utf-8

# In[36]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from sklearn.preprocessing import StandardScaler

sc_obj = StandardScaler()


# In[37]:


training_1=[]

#with open('/home/calsoft/Documents/machine_learning/training_data.csv','r') as f:
df = pd.read_csv('/home/calsoft/Documents/machine_learning/training_data.csv')

#val_share = 0.2
#val_cnt = int(len(df)*val_share)
#val_df = df.iloc[:val_cnt]
#train_df = df.iloc[val_cnt:]
#train_df = detect_drop_outlier(train_df)
#val_df.shape


# In[38]:


def detect_drop_outlier(data_frame):
    if not isinstance(data_frame,pd.DataFrame):
        raise Exception("Expecting DataFrame object as argument")

    #import pdb;pdb.set_trace()
    for feature in data_frame.columns[:-1]:
        first_quartile = data_frame[f'{feature}'].describe()['25%']
        third_quartile = data_frame[f'{feature}'].describe()['75%']

    # Interquartile range
        iqr = third_quartile - first_quartile
    # Remove outliers
        df = data_frame[(data_frame[f'{feature}'] > (first_quartile - 3 * iqr)) &
                   (data_frame[f'{feature}'] < (third_quartile + 3 * iqr))]
    return df


# In[5]:

def pre_process(df):
    df  = df.drop([" weekday_is_saturday"," weekday_is_sunday"], axis=1)
    #df = df.sample(len(df))
    df = detect_drop_outlier(df)
    x = df[df.columns[:-1]].values.astype(np.float32)
    y = df[df.columns[-1]].values.astype(np.float32).reshape(-1,1)
    #x = sc_obj.fit_transform(df[df.columns[:-1]].values.astype(np.float32))
    #y = sc_obj.fit_transform(df[df.columns[-1]].values.astype(np.float32).reshape(-1,1))

    return df, x, y

#def plotdatawrtcol(data,col):
#    plt.figure(figsize(4,2))
#    plt.style.use('fivethirtyeight')
#    plt.hist(data[col].dropna(),bins = 50,edgecolor = 'k');
#    plt.xlabel('tokens'); plt.ylabel('count'); 
#    plt.title(column);
#    plt.show()

#training = train_df.copy()
#for column in training.columns:
#    plotdatawrtcol(training,column)


# In[6]:


#training = train_df.copy()
#training = detect_drop_outlier(training)

#for column in training.columns:
#    plotdatawrtcol(training,column)
#print(training.columns)

#training = training.drop([' is_weekend'],axis=1)


# In[39]:



#training = detect_drop_outlier(training)

#x_mean = data_x.mean(axis=0)
#y_mean = data_y.mean()#
#x_std= data_x.std(axis=0)

#data_val_x = val_df[val_df.columns[1:-1]].values.astype(np.float32)
#data_val_y = val_df[val_df.columns[-1]].values.astype(np.float32)

#def normalize(x,y):
#    x -= x_mean
#    x /= x_std
#    return x,y
#x = sc_obj.transform(xtest)
#y = sc_obj.fit_transform(data_y.reshape(-1,1))


# In[8]:


#x,y = normalize(data_x,data_y)
#val_x,val_y = normalize(data_val_x,data_val_y)


# In[9]:


#len(val_x)
#val_x.shape


# In[43]:


from keras import Sequential
from keras import models
from keras import layers
from keras import optimizers
from keras import regularizers
from keras.layers import BatchNormalization
#reg=0.01
df, x, y = pre_process(df)
#activation = "sigmoid"
def create_model():
    activation = 'relu'
    model = Sequential()
    model.add(layers.Dense(4, activation=activation, kernel_initializer = "uniform",input_shape=(x.shape[1],)))
    model.add(BatchNormalization())
    model.add(layers.Dense(8, activation=activation))
    model.add(BatchNormalization())
    model.add(layers.Dense(16, activation=activation))
    model.add(BatchNormalization())
#model.add(layers.Dense(8, activation='relu'))
#model.add(BatchNormalization())
#model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation=activation))
    model.add(BatchNormalization())
##model.add(BatchNormalization())
    model.add(layers.Dropout(0.2))
#model.add(layers.Dense(32, activation='relu',))
    model.add(layers.Dense(64, activation=activation))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(64, activation=activation))
    #model.add(BatchNormalization())
   # model.add(layers.Dropout(0.4))
    #model.add(layers.Dense(128, activation=activation))
    #model.add(layers.Dropout(0.4))
    #model.add(BatchNormalization())
    #model.add(layers.Dense(512,activation=activation))
    #model.add(BatchNormalization())
    #model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation=activation,kernel_regularizer='l1'))
    #model.add(layers.Dropout(0.5))
    model.add(BatchNormalization())
    #model.add(layers.Dense(512,activation=activation))
    #model.add(BatchNormalization())
    #model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(64,activation=activation))
    #model.add(BatchNormalization())
    #model.add(layers.Dropout(0.2))
    #model.add(layers.Dense(256,activation=activation))
    #model.add(BatchNormalization())
   # model.add(layers.Dropout(0.4))
    #model.add(layers.Dense(128,activation=activation))
    #model.add(BatchNormalization())
    #model.add(layers.Dropout(0.3))
    #model.add(layers.Dense(64,activation=activation))
    #model.add(BatchNormalization())
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32,activation=activation))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.12))
    model.add(layers.Dense(16,activation=activation))
    model.add(BatchNormalization())
    model.add(layers.Dense(4,activation=activation))
    model.add(BatchNormalization())
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.Nadam(lr=0.001), loss = 'mape', metrics=['accuracy'])
    return model
    '''
    model = Sequential()
    model.add(layers.Dense(32, activation='relu', kernel_initializer = "uniform",input_shape=(x.shape[1],)))
    model.add(layers.Dropout(0.2))
    model.add(BatchNormalization())
    model.add(layers.Dense(32, activation='relu',kernel_initializer = "uniform"))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu',kernel_initializer = "uniform"))
    model.add(BatchNormalization())
    model.add(layers.Dropout(0.2))
#model.add(layers.Dense(8, activation='relu'))
#model.add(BatchNormalization())
#model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu',kernel_initializer = "uniform"))
    model.add(BatchNormalization())
##model.add(BatchNormalization())
    model.add(layers.Dropout(0.2))
#model.add(layers.Dense(32, activation='relu',))
    model.add(layers.Dense(16, activation='relu',kernel_initializer = "uniform"))
    model.add(BatchNormalization())

#model.add(layers.Dense(16, activation='relu',kernel_regularizer=regularizers.l2(reg)))
    model.add(layers.Dense(1))

    model.compile(optimizer=optimizers.adam(lr=0.001), loss = 'mape', metrics=['mae'])
    return model
    '''

# In[56]:

model = create_model()
#estimator = KerasRegressor(build_fn = create_model,batch_size = 64,epochs = 80)
#kfold = KFold(n_splits = 5,random_state=42)
#history = estimator.fit(x,data_y)


history = model.fit(x, y,
                    batch_size= 64,
                    epochs = 80,
#                    validation_data=(val_x,val_y)#
                   )
#from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
#model = create_model()
#estimator = KerasRegressor(build_fn = create_model,batch_size = 64,epochs = 80)
#kfold = KFold(n_splits = 5,random_state=42)
#history = estimator.fit(x,data_y)

#history = model.fit(x, y,
#                    batch_size= 64,
#                    epochs = 80,
#                    validation_data=(val_x,val_y)#
#                   )


# In[239]:


#import matplotlib.pyplot as pp
#%matplotlib inline
#def plothistory(history):
#    pp.figure(figsize=(14,4))
#    pp.subplot(1,2,1)
#    pp.plot(history.history['loss'][10:],label='Training loss')
#    pp.plot(history.history['val_loss'][10:],label='Validation loss')
#    pp.xlabel('epochs')
#    pp.ylabel('loss')
#    pp.legend()
#    pp.title('Loss vs epoch')
    
#    pp.subplot(1,2,2)
#    pp.plot(history.history['mean_absolute_error'][10:],label='mean absolute error')
#    pp.plot(history.history['val_mean_absolute_error'][10:],label='validation mean absolute error')
#    pp.xlabel('epochs')
#    pp.ylabel('mae')
#    pp.legend()
#    pp.title('Mean absolute error vs epoch')
#    pp.show()


# In[240]:


#plothistory(history)


# In[57]:


from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
#print("Saved model to disk")
# In[240]:

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=optimizers.adam(lr=0.001), loss = 'mape', metrics=['accuracy'])

with open('/home/calsoft/Documents/machine_learning/test_data.csv','r') as f:
    df = pd.read_csv(f)


df  = df.drop([" weekday_is_saturday"," weekday_is_sunday"], axis=1)



testing = df.copy()
#testing = detect_drop_outlier(testing)

#testing = sc_obj.transform(testing)

data_x = testing[testing.columns[0:]].values.astype(np.float32)
#data_x = sc_obj.transform(data_x)
result = loaded_model.predict(data_x)

np.set_printoptions(suppress=True)
result = np.array(result)
print(result)
url_id = testing["url_id"]
result = result.ravel()

final_df = pd.DataFrame({"url_id" : url_id, "shares" : result})

final_df.to_csv("~/Desktop/submission.csv", index=False)
#pd.Dataframe(prediction, columns = ['url_id','result]
#pred_val = model.predict(val_x)


# In[50]:


#val_df_pred=val_df.copy()
#val_df_pred['Predicted'] = pred_val.squeeze()
#val_df_pred['mape'] = (val_df_pred[' shares']-pred_val.squeeze()).abs()/val_df_pred[' shares']*100
#val_df_pred.columns = [column.strip() for column in val_df_pred.columns]
#val_df_pred.head()


# In[215]:


#sorted_df =val_df_pred.sort_values('mape')[['shares','Predicted','mape']]
#sorted_df.head(10)


# In[187]:


#sorted_df['mape'][1:].mean()


# In[206]:


#sorted_df.tail(250)


# In[122]:


#pd.DataFrame([val_df_pred.loc[22483],val_df_pred.loc[2530]])


# In[208]:


#plt.figure(figsize(14,2))
#plt.style.use('fivethirtyeight')
#plt.hist(train_df.sort_values(' shares')[' shares'].dropna(),bins = 100,edgecolor = 'k');
#plt.xlabel('shares'); plt.ylabel('count'); 
#plt.title(column);
#plt.show()


# In[197]:


#train_df.columns


# In[211]:


#df['share range'] = pd.cut(df[' shares'], [0, 10, 100,1000,10000,100000,1000000])
#df.groupby(['share range'])['share range'].count()


# In[223]:


#sorted_df['share range'] = pd.cut(sorted_df['shares'], [0, 10, 100,1000,10000,100000,1000000])
#sorted_df.groupby(['share range']).mean()*sorted_df.groupby(['share range']).count()


# In[236]:


#sorted_df.groupby(['share range']).mean()


# In[231]:


#sorted_df.shares.describe()

