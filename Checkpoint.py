#IMPORTS
import streamlit as st
import pandas as pd

####################################################################################
#SET ALL VARIABLES EQUAL TO "None"
neighborhood,property_type,room_type,bedrooms,bathrooms,max_capacity,utilities,extra,distance_from_center,cancellation_policy= "Select","Select","Select","Select","Select","Select","Select","Select","Select","Select"
start=False

st.set_page_config(page_title="AIrBnB Pricing Tool",page_icon="airlogo.png")
st.image("airlogo.png",width=270)  #CHECKPOINT


st.write("""# Airbnb Price Sugestion Tool""")
st.header("    ")
st.write("This app helps you setting the price of your Listing comparing it with similar ones on our platform!")
st.header('Just answer to a couple of questions ;)')
st.header("    ")  #CHECKPOINT

neighbourhood = st.selectbox(
    'Where is the Neighborhood of your Listing?',
    ("Select",'Bijlmer-Oost', 'Noord-Oost', 'Noord-West', 'Oud-Noord',
       'IJburg - Zeeburgereiland', 'Centrum-West',
       'Oostelijk Havengebied - Indische Buurt', 'Centrum-Oost',
       'Oud-Oost', 'Watergraafsmeer', 'Gaasperdam - Driemond',
       'Westerpark', 'Bijlmer-Centrum', 'De Pijp - Rivierenbuurt', 'Zuid',
       'Buitenveldert - Zuidas', 'De Baarsjes - Oud-West',
       'Bos en Lommer', 'Geuzenveld - Slotermeer', 'Slotervaart',
       'Osdorp', 'De Aker - Nieuw Sloten')
)
#   CHECKPOINT

if neighbourhood != "Select":
    property_type= st.selectbox("What kind of Listing do you have?",("Select","Bed & Breakfast","House","Apartment"))

#CHECKPOINT

if property_type != "Select":
    room_type= st.selectbox("What kind of accomodation are you offering?",("Select",'Private room', 'Entire home/apt', 'Shared room'))

#   CHECKPOINT

if room_type != "Select":
    bedrooms= st.slider("How many Bedrooms does your Listing have?",0,9)

if bedrooms!="Select" and bedrooms!=0:
    bathrooms= st.slider(("How many Bathrooms does your Listing have?"),0,5)

if bathrooms!= "Select" and bathrooms!= 0:
    max_capacity= st.slider("How many people can your Listing accomodate at max?",0,15)


if max_capacity!= "Select" and max_capacity!= 0:
    distance_from_center= st.number_input("Approximately, how much is your Listing distant from the center of the city? (in Km)")
#CHECKPOINT

if distance_from_center!="Select" and distance_from_center != 0.0:
    cancellation_policy= st.radio("What about your Cancellation Policy?",(None,"Flexible","Moderately Flexible","Strict"))
#CHECKPOINT

if cancellation_policy!="Select" and cancellation_policy!=None:
    utilities=st.multiselect("Tell us more about your Listing: select all the utilities it has",("None","Kitchen","Wireless Internet","TV","Washer","Hair Dryer","Air Conditioning"), default=None)

if utilities!="Select" and len(utilities)>0:
    extra=st.multiselect("We are almost finished! Tell us if your listing has some 'Extra' perks...",("None","Gym","Pool","Breakfast","24-Hour Check-in","Elevator in Building","Free Parking on Premises","Pets Allowed"), default=None)


if extra!= "Select" and len(extra)>0:
    start= st.button("Get Your Report!!!")

# CHECKPOINT  , spiegare che l app è finita(DI BASE), ma bisogna processare i dati.

if start==True:

    user_data = pd.DataFrame()

    # GET THE PERCENTAGE OF UTILITIES AND EXTRA SERVICES OFFERED
    if "None" in utilities:
        utilities = 0
    else:
        utilities = len(utilities) / 6

    if "None" in extra:
        extra = 0
    else:
        extra = len(extra) / 7

    user_data = {"max_capacity": [max_capacity],
                 "bathrooms": [bathrooms],
                 "bedrooms": [bedrooms],
                 "cancellation_policy": [cancellation_policy],
                 "Distance_from_center": [distance_from_center],
                 "utilities": [utilities],
                 "extra": [extra],
                 "neighbourhood": [neighbourhood],
                 "property_type": [property_type],
                 "room_type": [room_type]
                 }

    # CREATE USER DATASET
    row = pd.DataFrame(data=user_data)


    #ENCODING
    df = pd.read_csv("Airbnb-cleaned.csv")

    df = df[['neighbourhood', 'property_type', 'room_type']]
    # IMPORT ENCODER
    from sklearn.preprocessing import OneHotEncoder

    # FIT ENCODER ON THE ORIGINAL DATASET TO MAKE IT REMEMBER CATEGORIES
    enc = OneHotEncoder(sparse=False)
    enc.fit(df)
    # ISOLATE CAT VARIABLES AND ENCODE THEM

    row_cat = row[['neighbourhood', 'property_type', 'room_type']]

    row_cat[['Bijlmer-Oost', 'Noord-Oost', 'Noord-West', 'Oud-Noord',
             'IJburg - Zeeburgereiland', 'Centrum-West',
             'Oostelijk Havengebied - Indische Buurt', 'Centrum-Oost',
             'Oud-Oost', 'Watergraafsmeer', 'Gaasperdam - Driemond',
             'Westerpark', 'Bijlmer-Centrum', 'De Pijp - Rivierenbuurt', 'Zuid',
             'Buitenveldert - Zuidas', 'De Baarsjes - Oud-West',
             'Bos en Lommer', 'Geuzenveld - Slotermeer', 'Slotervaart',
             'Osdorp', 'De Aker - Nieuw Sloten',
             'Apartment', 'Bed & Breakfast', 'House',
             'Entire home/apt', 'Private room', 'Shared room']] = enc.transform(
        row_cat[["neighbourhood", "property_type", "room_type"]])

    row_cat.drop(["neighbourhood", "property_type", "room_type"], axis=1, inplace=True)
    row.drop(["neighbourhood", "property_type", "room_type"], axis=1, inplace=True)

    row = pd.concat([row, row_cat], axis=1)

    # ORDINAL ENCODING FOR CANCELLATION POLICY
    row['cancellation_policy'] = row['cancellation_policy'].map({'Strict': 0, 'Moderately Flexible': 1, "Flexible": 2})
    # GET PREDICTIONS

    import pickle
    # model=pickle.load(open("Airbnb-Workshop.sav","rb"))
    from joblib import dump, load

    model = load('Airbnb.joblib')

    pred = model.predict(row)

    st.write("Our model estimates an optimum price x night of " + str(pred[0]) + "$")

    #AGGIUNGILO ALLA FINE
    st.balloons()

