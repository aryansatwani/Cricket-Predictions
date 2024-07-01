import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBRegressor

pipe = pickle.load(open('pipepkl', 'rb'))

teams = ['India',
 'Pakistan',
 'Australia',
 'South Africa',
 'New Zealand',
 'England',
 'Bangladesh',
 'Sri Lanka',
 'Afghanistan',
 'Netherlands']

cities = ['Brisbane', 'Melbourne', 'Perth', 'Sydney', 'Adelaide', 'Canberra',
       'Christchurch', 'Nelson', 'Auckland', 'Hamilton', 'Wellington',
       'London', 'Birmingham', 'Cardiff', 'Mirpur', 'Chittagong',
       'Dharmasala', 'Delhi', 'Chandigarh', 'Ranchi', 'Visakhapatnam',
       'Leeds', 'Southampton', 'Dublin', 'Pune', 'Cuttack', 'Kolkata',
       'Kimberley', 'Paarl', 'Rangiri', 'Colombo', 'Pallekele',
       'Mount Maunganui', 'Dunedin', 'Indore', 'Bengaluru', 'Nagpur',
       'Nottingham', 'Chester-le-Street', 'Manchester', 'Mumbai',
       'Kanpur', 'Dubai', 'Abu Dhabi', 'Sharjah', 'Durban', 'Centurion',
       'Cape Town', 'Johannesburg', 'Port Elizabeth', 'Dharamsala',
       'Dhaka', 'Bristol', 'Taunton', 'Hobart', 'Napier', 'Hyderabad',
       'Bloemfontein', 'Potchefstroom', 'Rajkot', 'Karachi', 'Canterbury',
       'Harare', 'Ahmedabad', 'Vadodara', 'Lahore', 'Rawalpindi',
       'Queenstown', 'Peshawar', 'Multan', 'Bogra', 'Fatullah',
       'Faridabad', 'Margao', 'Jamshedpur', 'St Lucia', 'Trinidad',
       'St Kitts', 'Guyana', 'Antigua', 'Barbados', 'Grenada', 'Jamaica',
       'Jaipur', 'Kuala Lumpur', 'Belfast', 'Bangalore', 'Kochi',
       'Guwahati', 'Gwalior', 'Faisalabad', 'Darwin', 'Chennai',
       'Bulawayo', 'East London', 'Hambantota', 'Benoni']

st.title('Runs Predictor')

import streamlit as st

# Create columns for layout
col1, col2 = st.columns(2)

# Add widgets to the first column
with col1:
    batting_team = st.selectbox('Select batting team', sorted(teams))

    bowling_team = st.selectbox('Select bowling team', sorted(teams))

    city = st.selectbox('Select city', sorted(cities))

# Add widgets to the second column
with col2:
    current_score = st.number_input('Current Score')

    overs = st.number_input('Overs done (works for over > 5)')

    wickets = st.number_input('Wickets left')

# Add widgets in a new row using st.columns()
col3, col4, col5 = st.columns(3)

with col3:
    last_five = st.number_input('Runs scored in last 5 overs')

# Add button in a new column using st.columns()
with col4:
    if st.button('Predict Score'):
        pass

# The fifth column is empty
with col5:
    balls_left=300- (overs * 6)
    wickets_left=10-wickets
    crr=current_score/overs
    
    input_df=pd.DataFrame(
        {
            'batting_team':[batting_team],'bowling_team':[bowling_team],'city':city,'current_score':[current_score],'balls_left':[balls_left],'wickets_left':[wickets],'crr':[crr],'last_five':[last_five]})
    st.text(xgboost.__version__)
    result=pipe.predict(input_df)
    st.header("Predicted Score: " + str(int(result[0])))
