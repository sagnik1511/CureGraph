import torch
import streamlit as st
from src.prediction import draw, predict_from_smiles
from src.prediction import validate_smiles_string


st.title("Graph-Cure")



prob = None
res = None
submit = None

model = torch.load("best_model.pth")
model.to(torch.device("cpu"))
model.eval()

page = st.sidebar.selectbox("Select Page", ["Home", "Prediction"])

if page == "Home":
    st.header("Identify molecules that can inhibit HIV")
else:
    st.markdown("Select input molecule.")
    upload_columns = st.columns([2, 1])

    # Smiles input
    smiles_select = upload_columns[0].expander(label="Specify SMILES string")
    smiles_string = smiles_select.text_input("Enter a valid SMILES string.")

    if smiles_string and validate_smiles_string(smiles_string):
        try:
            upload_columns[1].image(draw(smiles_string))
            submit = upload_columns[1].button("Get predictions")
            if submit:
                print(smiles_string)
                with st.spinner(text="Fetching model prediction..."):
                    res, prob = predict_from_smiles(smiles_string, model)
                result_blocks = st.columns([2, 1])
                result_blocks[0].subheader("HIV Inhibition Status")
                if res == 1:
                    result_blocks[1].success("Positive")
                else:
                    result_blocks[1].error("Negative")
                st.markdown("""---""")
                detail_blocks = st.columns([2, 1])
                detail_blocks[0].subheader("Confidence")
                detail_blocks[1].subheader(f"{round(prob, 2)} %")
        except:
            st.error("Enter a valid smiles string")
    elif not smiles_string:
        pass
    else:
        st.error("Enter a valid smiles string")
    st.markdown("""---""")
