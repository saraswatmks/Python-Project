#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 1000)

plt.style.use('fivethirtyeight')

import lightgbm as lgb
from sklearn.model_selection import train_test_split


# In[2]:


names = [
    'message_sent_date',
    'time_sent_local',
    'buyer_sk',
    'item_id',
    'msg_num_in_conv',
    'reply_channel_sk',
    'msg_type',
    'message_text'
]
chats = pd.read_csv("s3://olx-relevance-panamera-west/olx-mas-data/web/chats/000.gz", 
                    names=names, 
                    dtype=str, index_col=False)


# In[3]:


chats.head()


# In[4]:


names= [
    'date_sent_nk',
    'buyer_sk',
    'listing_id',
    'action_sk',
    'reply_channel_sk',
    'action_count'
]

contacts = pd.read_csv("s3://olx-relevance-panamera-west/olx-mas-data/web/contacts/000.gz", 
                       names=names
                       )


# In[5]:


contacts.head()


# In[4]:


names = [
    'listing_id',
    'status',
    'district_id',
    'city_id',
    'seller_id',
    'created_at_first',
    'created_at',
    'valid_to',
    'trackevent',
    'title',
    'description',
    'price',
    'brand',
    'model',
    'fuel_type',
    'mileage',
    'installments',
    'new_or_used',
    'year',
    'registration_city',
    'time_sold_local',
    'ad_update_count'
]

ads_data = pd.read_csv("s3://olx-relevance-panamera-west/olx-mas-data/web/user_ads/000.gz",
                       names=names, dtype=str, escapechar='\\', index_col=False)


# In[5]:


ads_data.head()


# In[55]:


ads_data.shape


# In[56]:


ads_data.drop_duplicates(inplace=True)


# In[57]:


ads_data.shape


# In[58]:


ads_data['created_at_first'] = pd.to_datetime(ads_data['created_at_first'])
ads_data['created_at'] = pd.to_datetime(ads_data['created_at'])
ads_data['valid_to'] = pd.to_datetime(ads_data['valid_to'])
ads_data['listing_id'] = ads_data['listing_id'].astype(int)


# In[59]:


# keep only ads valid until 19th July
ads_data = ads_data[ads_data['valid_to'].le('2019-07-18 23:59:59')]


# In[135]:


.12/.36


# In[134]:


ads_data.groupby('brand').transform('len')


# In[60]:


ads_data.shape


# In[61]:


# remove outliers
ads_data['year'] = pd.to_numeric(ads_data['year'], errors='coerce')
ads_data['price'] = pd.to_numeric(ads_data['price'], errors='coerce', downcast='integer')

percentile_99 = ads_data['price'].quantile(0.99)
ads_data = ads_data[ads_data['price'].lt(percentile_99)]
ads_data = ads_data[ads_data['year'].ge(1960)]


# In[62]:


# set fake_mileage if the mileage has less than 3 letters
ads_data['fake_mileage'] = ads_data['mileage'].str.len().lt(3).astype(int)


# In[63]:


ads_data['std_price_district'] = ads_data.groupby(['district_id','brand','model'])['price'].transform('std')


# In[64]:


names = [
    'created_at_first',
    'time_sold_local',
    'message_sent_date',
    'buyer_sk',
    'item_id',
    'reply_channel_sk',
    'msg_type',
    'msg_count'
]

chat_count = pd.read_csv("s3://olx-relevance-panamera-west/olx-mas-data/web/chat_count/000.gz", 
                       names = names
                       )


# In[65]:


chat_count.head()


# In[66]:


replies = (pd
 .pivot_table(index=['item_id','buyer_sk'], 
              columns=['msg_type'], 
              values='msg_count', 
              data=chat_count, 
              aggfunc='sum',
              fill_value=0)
           .reset_index())
replies.columns.name = None


# In[67]:


replies.head()


# In[68]:


# only keep seller messages
replies = (replies
           .groupby('item_id')
           .aggregate({'buyer_sk': 'nunique', 'seller-buyer': sum})
           .reset_index())
replies.columns = ['listing_id','unique_buyer_chat_sent_count','seller_msgs']


# In[69]:


ads_data = ads_data.merge(replies, how='left', on='listing_id')


# In[70]:


# cumulative unique ads posted by seller until each date
ads_data['seller_ad_count'] = ads_data.groupby('seller_id')['listing_id'].cumcount()


# In[71]:


# add features from contacts
contacts['action_sk'] = contacts['action_sk'].str.split('|').str[1]


# In[72]:


real_contacts = pd.pivot_table(index=['listing_id','buyer_sk'], 
               columns='action_sk', 
               data=contacts, 
               values='action_count', 
               aggfunc='mean', 
               fill_value=0)
real_contacts.columns.name = None
real_contacts = real_contacts.reset_index()


# In[73]:


real_contacts.head()


# In[74]:


# call-sms 0.86 correlation
real_contacts = (real_contacts
           .groupby('listing_id')
           .aggregate({'call': sum, 'sms': sum})
           .reset_index())
real_contacts.columns = ['listing_id','call_count','sms_count']
# real_contacts = real_contacts.drop('chat_count', axis=1) # because this is a same feature in replies data frame


# In[75]:


real_contacts.head()


# In[29]:


contacts['reply_channel_sk'] = contacts['reply_channel_sk'].str.split('|').str[1]


# In[30]:


platform_contacts = pd.pivot_table(index=['listing_id'], 
               columns=['action_sk','reply_channel_sk'], 
               data=contacts, 
               values='action_count', 
               aggfunc='sum', 
               fill_value=0)
#platform_contacts.columns.name = None
#platform_contacts = real_contacts.reset_index()

platform_contacts.columns = ['_'.join(col).strip() for col in platform_contacts.columns.values]
platform_contacts = platform_contacts.reset_index()


# In[76]:


ads_data = (ads_data
            .merge(real_contacts, how='left',on='listing_id')
           )


# In[32]:


ads_data = (ads_data
            .merge(platform_contacts, how='left',on='listing_id')
           )


# In[33]:


cols = ['unique_buyer_chat_sent_count','seller_msgs','call_count','sms_count']
ads_data[cols] = ads_data[cols].fillna(0).astype(int)


# In[37]:


ads_data[['unique_buyer_chat_sent_count',
          'seller_msgs','seller_ad_count',
         'call_count']].corr()


# In[45]:


drop_correlated_cols = ['call_android','chat_android','call_count','sms_android','sms_count']
ads_data = ads_data.drop(drop_correlated_cols, axis=1)


# In[77]:


ads_data.head()


# In[78]:


old_data = ads_data[ads_data.created_at_first < '2019-05-07']
patch_data = ads_data[ads_data.created_at_first >= '2019-05-07']

print("number of ads created before '2019-05-07': {}\nnumber of ads created after '2019-05-07': {}"
      .format(old_data.shape[0], patch_data.shape[0]))


# In[79]:


from sklearn.model_selection import train_test_split

# split data into train and test (not using old ads at this point)
train, test = train_test_split(patch_data, test_size=0.2, shuffle=True, random_state=42)
train_size = train.shape[0] 
test_size = test.shape[0]
total_size = train_size + test_size
print("train size: {} ({:.2f}%), test size: {} ({:.2f}%)"
      .format(train_size, train_size/float(train_size + test_size) * 100.0,
              test_size, test_size/float(train_size + test_size) * 100.0))


# In[80]:


# add old ads for training
train = pd.concat([train, old_data], axis=0) # to avoid appending the data multiple times!

# shuffle data
train = train.sample(frac=1.0, random_state=42)


# In[81]:


# fix description null values
train['description'] = train['description'].fillna("")
test['description'] = test['description'].fillna("")


# In[82]:


ads_data.columns


# In[125]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from xgboost import XGBClassifier
from sklearn.decomposition import TruncatedSVD

def generate_pipeline(**kwargs):
    numeric_features = ['price', 'mileage', 'year', 'ad_update_count', 'fake_mileage', 'unique_buyer_chat_sent_count',
                        'seller_msgs', 'seller_ad_count','std_price_district','call_count'] 

    # categorical_features = ['registration_city', 'brand', 'model', 'petrol', 'installments', 'new_or_used']
    # categorical_transformer = Pipeline([
    #     ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    #     ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    

    features = ColumnTransformer(
        [('numeric', 'passthrough', numeric_features),
    #      ('categorical', categorical_transformer, categorical_features),
         ('title_bow', CountVectorizer(stop_words='english', ngram_range=(1,3), min_df=7, max_df=0.9), 'title'),
         ('descr_bow_svd', Pipeline([("desc_bow", TfidfVectorizer(stop_words=None, ngram_range=(1,1), min_df=7, max_df=1.0)),
                                     ('svd', TruncatedSVD(n_components=50, n_iter=20))]),
          'description'),
         
        ],  
        remainder='drop',
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None
    #     verbose=False,
    )
    
    params = {k.split('__')[-1]:v for k,v in kwargs.items()}
    clf = XGBClassifier(n_estimators=500, **params)
    params_xgb = {
        'clf__learning_rate': np.logspace(-3, -1, 100),
        'clf__max_depth': (5,10,20,40,80),
        ##'clf__reg_alpha': (10,20,30,50),
        ##'clf__reg_lambda': (10,20,30,50),
        'clf__subsample': (.7,.8,.9), 
        'clf__colsample_bytree': (.7,.8,.9)
    }

    model = Pipeline([
        ('features', features),
        ('clf', clf)
    ]) 
    return model


# In[126]:


model = generate_pipeline()

search = RandomizedSearchCV(estimator=model, param_distributions=params_xgb, scoring='precision',
                            cv=3, verbose=3, n_iter=50, n_jobs=-1, random_state=42)


# In[86]:


get_ipython().run_line_magic('time', "search.fit(train, train.status == 'sold')")


# In[87]:


search.best_score_, search.best_params_


# In[88]:


from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred = search.best_estimator_.predict(test)
y_pred_proba = search.best_estimator_.predict_proba(test)[:, 1]
print(roc_auc_score(test.status == 'sold', y_pred_proba))
print(classification_report(test.status == 'sold', y_pred))
print(confusion_matrix(test.status == 'sold', y_pred))


# ### train model with fix parametes

# (0.673352482515424,
#  {'clf__subsample': 0.8,
#   'clf__max_depth': 40,
#   'clf__learning_rate': 0.049770235643321115,
#   'clf__colsample_bytree': 0.9})

# In[127]:


model = generate_pipeline(**search.best_params_)


# In[128]:


model.fit(train, train.status == 'sold')


# In[123]:


# without SVD
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred = model.predict(test)
y_pred_proba = model.predict_proba(test)[:, 1]
print(roc_auc_score(test.status == 'sold', y_pred_proba))
print(classification_report(test.status == 'sold', y_pred))
print(confusion_matrix(test.status == 'sold', y_pred))


# In[129]:


# with SVD
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

y_pred = model.predict(test)
y_pred_proba = model.predict_proba(test)[:, 1]
print(roc_auc_score(test.status == 'sold', y_pred_proba))
print(classification_report(test.status == 'sold', y_pred))
print(confusion_matrix(test.status == 'sold', y_pred))


# In[ ]:





# In[130]:


xgb_model = search.best_estimator_.named_steps['clf']


# In[131]:


plt.figure(figsize=(15, 5))
plt.bar(numeric_features, xgb_model.feature_importances_[:len(numeric_features)])
plt.title('Feature Importances (Numeric Variables Only)')
plt.ylabel('importance');
_ = plt.xticks(rotation=90)


# In[67]:


title_vect = search.best_estimator_.named_steps['features'].named_transformers_['title_bow']
descr_vect = search.best_estimator_.named_steps['features'].named_transformers_['descr_bow']


# In[69]:


title_vect.get_feature_names()


# In[ ]:





# In[6]:


ads_data.head()


# In[42]:


all_items = ads_data.groupby('model')['listing_id'].apply(list).reset_index()


# In[43]:


all_items = all_items.set_index('model')['listing_id'].apply(pd.Series).unstack().reset_index(level=-1)
all_items.columns = ['model','items']


# In[58]:


all_items = all_items[all_items['items'].notna()]


# In[59]:


all_items.head(100)


# In[64]:


audi_data = all_items[all_items['model'].str.startswith('bmw')]
suzuki_data = all_items[all_items['model'].str.startswith('cars-suzuki')]


# In[66]:


suzuki_data.head()


# In[81]:


suzuki_data['key'] = 0


# In[68]:


audi_data['key'] = 0


# In[70]:


audi_pairs = audi_data.merge(audi_data, how='outer', on='key')


# In[73]:


audi_pairs = audi_pairs.query('model_x != model_y')


# In[75]:


audi_pairs = audi_pairs.drop(['model_x','model_y','key'], axis=1)


# In[77]:


audi_pairs['label'] = 1


# In[142]:


suzuki_pairs = (audi_data
.merge(suzuki_data, how='outer', on='key').query('model_x != model_y')
.drop(['model_x','model_y','key'], axis=1))


# In[143]:


suzuki_pairs['label'] = 0


# suzuki_pairs =  suzuki_pairs.sample(frac=0.1).reset_index(drop=True)

# In[144]:


suzuki_pairs['items_x'] = suzuki_pairs['items_x'].astype(int)


# In[145]:


audi_pairs['items_x'] = audi_pairs['items_x'].astype(int)


# In[ ]:


audi_pairs = audi_pairs[["items_x","items_y"]]
audi_pairs.columns = ["item", "positive"]

suzuki_pairs = suzuki_pairs[["items_x","items_y"]]
suzuki_pairs.columns = ["item", "negative"]


# In[153]:



all_pairs = audi_pairs.merge(suzuki_pairs, on="item", how="left").sample(frac=1.0).drop_duplicates(["item", "positive"])


# In[156]:


all_pairs.reset_index(drop=True, inplace=True)


# In[157]:


all_pairs.head()


# In[158]:


items = set(all_pairs['item'].tolist() + all_pairs['positive'].tolist() + all_pairs['negative'].tolist())


# In[165]:


item_attr = ads_data[ads_data['listing_id'].isin(items)].reset_index(drop=True)
item_attr = item_attr[['listing_id','title','district_id']]


# In[222]:


item_attr.head()


# In[225]:


all_pairs.head()


# In[ ]:





# In[257]:


path = "s3://olx-relevance-panamera-west/reco-network/"

names = [
    'listing_id',
    'district_id',
    'city_id',
    'created_at_first',
    'title',
    'description',
    'price',
    'brand',
    'model',
    'fuel_type',
    'mileage',
    'installments',
    'new_or_used',
    'year',
    'registration_city',
]

positive_pairs = pd.read_csv(path+'item_info/000.gz', 
                            names=names, 
                             escapechar='\\', 
                             index_col=False)


# In[258]:


positive_pairs.head()


# In[ ]:





# In[249]:


from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dot, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder

item_encoder = LabelEncoder()
item_encoder.fit(list(items))

item_attr["location_id"] = LabelEncoder().fit_transform(list(item_attr["district_id"]))
item_attr["item_id"] = LabelEncoder().fit_transform(list(item_attr["listing_id"]))
all_pairs["item_id"] = item_encoder.transform(all_pairs["item"])
all_pairs["positive_id"] = item_encoder.transform(all_pairs["positive"])
all_pairs["negative_id"] = item_encoder.transform(all_pairs["negative"])

title_max_len = 5
num_location_id = 1000

title_tokenizer = Tokenizer(num_words=10000)
title_tokenizer.fit_on_texts(item_attr["title"])
title_vocab_size = len(title_tokenizer.word_index) + 1

items_titles_encoded = title_tokenizer.texts_to_sequences(item_attr["title"])
items_titles_encoded = pad_sequences(items_titles_encoded, padding='post', maxlen=title_max_len)

location_encoded = item_attr[["location_id"]].values

title_input = Input(shape=(title_max_len,))
title_embedding = Embedding(title_vocab_size, 64)(title_input)
title_lstm = LSTM(32)(title_embedding)

location_id_input = Input(shape=(1,))
location_embedding = Embedding(num_location_id, 64)(location_id_input)
location_flattened = Flatten()(location_embedding)

merged = concatenate([title_lstm, location_flattened])
merged = Dropout(0.3)(merged)

z = Dense(512, activation="relu")(merged)
z = Dense(64, activation="tanh")(z)

sub_model = Model(inputs=[title_input, location_id_input], outputs=[z])

item_title_enc = Input(shape=(title_max_len,))
location_id_enc = Input(shape=(1,))

pos_item_title_enc = Input(shape=(title_max_len,))
pos_location_id_enc = Input(shape=(1,))

neg_item_title_enc = Input(shape=(title_max_len,))
neg_location_id_enc = Input(shape=(1,))

item_embedding = sub_model([item_title_enc, location_id_enc])
positive_item_embedding = sub_model([pos_item_title_enc, pos_location_id_enc])
negative_item_embedding = sub_model([neg_item_title_enc, neg_location_id_enc])

positive_sim = Dot(axes=-1)([item_embedding, positive_item_embedding])
negative_sim = Dot(axes=-1)([item_embedding, negative_item_embedding])

output = concatenate([positive_sim, negative_sim])

model = Model(inputs = [item_title_enc, location_id_enc,
                        pos_item_title_enc, pos_location_id_enc,
                        neg_item_title_enc, neg_location_id_enc],
              outputs = output)


model.compile(loss="categorical_crossentropy", optimizer="adam")

y_train =  len(all_pairs)*[[1,0]]

model.fit([items_titles_encoded[all_pairs["item_id"]], location_encoded[all_pairs["item_id"]],
           items_titles_encoded[all_pairs["positive_id"]], location_encoded[all_pairs["positive_id"]],
           items_titles_encoded[all_pairs["negative_id"]], location_encoded[all_pairs["negative_id"]]
          ], np.array(y_train), epochs=8)

item_embdedding = sub_model.predict([items_titles_encoded, location_encoded])

all_pairs.head()

np.dot(item_embdedding[293], item_embdedding[750])

np.dot(item_embdedding[293], item_embdedding[743])

all_pairs["pos_score"] = all_pairs.apply(lambda row: np.dot(item_embdedding[row["item_id"]],
                                                            item_embdedding[row["positive_id"]]) , axis=1)
all_pairs["neg_score"] = all_pairs.apply(lambda row: np.dot(item_embdedding[row["item_id"]],
                                                            item_embdedding[row["negative_id"]]) , axis=1)

all_pairs[["pos_score", "neg_score"]].describe()


# In[256]:


item_embdedding.dtype


# In[248]:


all_pairs[["pos_score", "neg_score"]].describe()


# In[233]:


all_pairs.head(10)


# In[251]:


all_pairs.to_csv('all_pairs.csv', index=False)


# In[252]:


item_attr.to_csv('item_attr.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




