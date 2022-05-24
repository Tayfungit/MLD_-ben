from pyparsing import White
import streamlit as st
import pickle
import pandas as pd
from PIL import Image



img = Image.open("auto.jpg")
st.sidebar.image(img, width=300)

st.sidebar.title('Autopreisvorhersage')
html_temp = """
<div style="background-color:#3fadcc;padding:10px">
<h2 style="color:#f7aa05;text-align:center;">Streamlit ML Cloud App </h2>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)


age=st.sidebar.selectbox("Wie alt ist Ihr Auto::",(1,2,3,4,5,6,7,8,9))
hp=st.sidebar.slider("Welche kW hat Ihr Auto?", 60, 200, step=5)
km=st.sidebar.slider("Wie viele km hat Ihr Auto", 0,100000, step=500)
gearing_type=st.sidebar.radio('Getriebetyp wählen',('Automatisch','Manual','Halbautomatisch'))
car_model=st.sidebar.selectbox("Wählen Sie Ihr Fahrzeugmodell aus", ('A1', 'A2', 'A3','Astra','Clio','Corsa','Espace','Insignia'))


# model_name=st.selectbox("Wählen Sie Ihr Modell:",("XGBOOST","Random Forest"))

# if model_name=="XGBOOST":
model=pickle.load(open("xgb_model","rb"))
# 	st.success("Sie haben das Modell {} ausgewählt".format(model_name))
# else :
# 	model=pickle.load(open("rf_model","rb"))
# 	st.success("Sie haben das Modell {} ausgewählt".format(model_name))



my_dict = {
    "Alter": age,
    "kW": hp,
    "km": km,
    "Modell": car_model,
    'Getriebetyp':gearing_type
}

df = pd.DataFrame.from_dict([my_dict])


st.header("Die Konfiguration Ihres Autos ist unten")
st.table(df)

columns= ['age','hp', 'km', 'model_A1', 'model_A2', 'model_A3', 'model_Astra', 'model_Clio', 'model_Corsa', 'model_Espace',
'model_Insignia',
'gearing_type_Automatic',
'gearing_type_Manual',
'gearing_type_Semi-automatic']


df = pd.get_dummies(df).reindex(columns=columns, fill_value=0)

# st.subheader("Drücken Sie Vorhersage, wenn die Konfiguration in Ordnung ist")

html_temp = """
<div style="background-color:#f7aa05;padding:6px">
<h5 style="color:#3fadcc;text-align:center;">Drücken Sie Vorhersage, wenn die Konfiguration in Ordnung ist</h5>
</div>"""
st.markdown(html_temp,unsafe_allow_html=True)

if st.button("Vorhersagen"):
    prediction = model.predict(df)
    st.success("Der geschätzte Preis Ihres Autos beträgt {} €. ".format(int(prediction[0])))
    

