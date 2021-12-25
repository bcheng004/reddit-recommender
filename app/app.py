from re import sub
import streamlit as st
from st_aggrid import AgGrid
import os, requests, json, confuse
from azureml.core import Workspace, Dataset
from azureml.core.webservice import AciWebservice
import pandas as pd
import numpy as np

config = confuse.Configuration('StRecApp')
config.set_file('config-st.yaml')
dataset_name = config['azureml']['dataset_name'].get()
aci_service_name = config['azureml']['aciservice']['endpoint_name'].get()
st_page_title = config['streamlit']['page_title'].get()
st_page_icon = config['streamlit']['page_icon'].get()

def get_pred_obj():
    ws = Workspace.from_config()
    subreddit_df = Dataset.get_by_name(ws,name=dataset_name).to_pandas_dataframe()
    top_user_count = subreddit_df['user'].value_counts().sort_values(ascending=False)[:1]
    top_user = list(top_user_count.index)[0]
    unique_items = subreddit_df['subreddit'].unique()
    item_user_x = subreddit_df.loc[(subreddit_df['user']==top_user),'subreddit']
    items_to_pred = np.setdiff1d(unique_items,item_user_x)
    return top_user, items_to_pred

def fetch_top_n_rec(top_user,items_to_pred,n=10):
    ws = Workspace.from_config()
    endpoint_service = AciWebservice(ws,name=aci_service_name)
    recom_list = []
    for iid in items_to_pred:
        pred_df = pd.DataFrame(columns=['user','subreddit'])
        pred_df.loc[0] = [top_user,iid]
        x_pred = json.dumps({'data': pred_df.to_dict(orient='records')})
        pred_response = endpoint_service.run(input_data=x_pred)
        recom_list.append(json.loads(pred_response))
    cleaned_recom_list = [pred['prediction'] for pred in recom_list]
    pred_recom_df = pd.DataFrame(cleaned_recom_list)
    top_n_recom = pred_recom_df.sort_values('est',ascending=False)[:n]
    return top_user, top_n_recom

def app():
    st.set_page_config(
        page_title=st_page_title,
        page_icon=st_page_icon
    )
    st.title(st_page_title)
    data_load_state = st.text('Loading data...')
    top_user, items_to_pred = get_pred_obj()
    data_load_state.text("Done!")
    with st.form("Subreddit Recommender"):
        user_sel = st.selectbox(
            'select user to generate recommendations for:',
            (top_user,))
        top_n = st.slider(
            'select top n recommendations:',
            min_value=1,
            max_value=10,
            value=10,
            step=1
        )
        submit = st.form_submit_button("Submit for recommendations")
        if submit:
            top_n_user, top_n_recom = fetch_top_n_rec(
                top_user=user_sel,
                items_to_pred=items_to_pred,
                n=top_n
            )
            if top_n_user:
                st.success(f"top {top_n} recommendations for {user_sel}")
                AgGrid(
                    top_n_recom,
                    theme="material",
                    fit_columns_on_grid_load=True,
                    editable=False
                )

if __name__ == '__main__':
    app()