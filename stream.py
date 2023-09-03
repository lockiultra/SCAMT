import streamlit as st
import pandas as pd
from DiseasePipeline import DiseasePipeline
import tensorflow.keras as keras
from MPNN import MessagePassing, TransformerEncoderReadout
from generate import get_valid_smiles

@st.cache_resource
def load_dp():
    dp = DiseasePipeline(pd.read_csv("data.csv"))
    for disease in dp.diseases:
        dp.models[disease] = keras.models.load_model(f'./Models/model_{disease}.h5', compile=False, custom_objects={'MessagePassing': MessagePassing, 'TransformerEncoderReadout': TransformerEncoderReadout})
    dp.is_trained = True
    return dp

def get_df():
    if 'result_dict' not in st.session_state:
        res = pd.DataFrame()
    else:
        res = pd.DataFrame(index=list(st.session_state.result_dict.keys()), data=[prob[0] for prob in st.session_state.result_dict.values()])
    st.session_state.result_df = res

def generate():
    st.session_state.smiles = get_valid_smiles()

def predict(smiles):
    st.session_state.result_dict = st.session_state.dp.predict([smiles])
    get_df()

st.session_state.dp = load_dp()
st.title("SCAMT Trainee")
st.text_input("SMILES", key="smiles")
st.button("Generate", on_click=generate)
st.button("Predict", on_click=predict, args=(st.session_state.smiles,))
get_df()
st.text("Predict results")
st.dataframe(st.session_state.result_df)
