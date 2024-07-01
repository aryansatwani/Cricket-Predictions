import streamlit as st
import pickle
import pandas as pd

cities = ['Brisbane', 'Melbourne', 'Perth', 'Sydney', 'Adelaide', 'Canberra',
       'Christchurch', 'Nelson', 'Auckland', 'Hamilton', 'Wellington',
       'London', 'Birmingham', 'Cardiff', 'Mirpur', 'Chittagong',
       'Dharmasala', 'Delhi', 'Chandigarh', 'Ranchi', 'Visakhapatnam',
       'Leeds', 'Southampton', 'Dublin', 'Pune', 'Cuttack', 'Kolkata',
       'Kimberley', 'Paarl', 'East London', 'Rangiri', 'Colombo',
       'Pallekele', 'Mount Maunganui', 'Dunedin', 'Chennai', 'Indore',
       'Bengaluru', 'Nagpur', 'Nottingham', 'Chester-le-Street',
       'Manchester', 'Mumbai', 'Kanpur', 'Dubai', 'Abu Dhabi', 'Sharjah',
       'Durban', 'Centurion', 'Cape Town', 'Johannesburg',
       'Port Elizabeth', 'Dharamsala', 'Dhaka', 'Bristol', 'Taunton',
       'Hobart', 'Napier', 'Hyderabad', 'Bloemfontein', 'Potchefstroom',
       'Rajkot', 'Karachi', 'Canterbury', 'Harare', 'Ahmedabad',
       'Vadodara', 'Lahore', 'Rawalpindi', 'Queenstown', 'Peshawar',
       'Multan', 'Bogra', 'Fatullah', 'Faridabad', 'Margao', 'Jamshedpur',
       'St Kitts', 'St Lucia', 'Trinidad', 'Guyana', 'Antigua',
       'Barbados', 'Grenada', 'Jamaica', 'Jaipur', 'Kuala Lumpur',
       'Belfast', 'Kochi', 'Guwahati', 'Gwalior', 'Faisalabad', 'Darwin',
       'Bangalore', 'Bulawayo', 'Hambantota', 'Benoni']
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

pipe = pickle.load(open('pipe2.pkl', 'rb'))
st.title('ODI Win Predictor')

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))

with col2:
    bowling_team=st.selectbox('Select the bowling team',sorted(teams))
    
selected_city = st.selectbox('Select host city',sorted(cities))

target=st.number_input('Target')

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed')
with col5:
    wickets_left = st.number_input('Wickets Out')
    
if st.button('Predict Probability'):
    runs_left = target-score
    balls_left=300-(overs*6)
    wickets_left =10-wickets_left
    crr = score/overs
    rrr=(runs_left*6)/balls_left
    
    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets_left':[wickets_left],'first_innings_total':[target],'crr':[crr],'rrr':[rrr]})
    
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + "-" + str(round(win*100))+"%")
    st.header(bowling_team + "-"+str(round(loss*100))+"%")

