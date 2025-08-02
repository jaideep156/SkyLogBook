import streamlit as st
import pandas as pd
import pydeck as pdk
from math import radians, cos, sin, asin, sqrt
import numpy as np
import altair as alt

st.set_page_config(page_title="SkyLogBook", layout="wide")
# st.title("✈️ My Flight Logger")
st.markdown("<h1 style='text-align: center;'>SkyLogBook</h1>", unsafe_allow_html=True)
# --- Load Data ---
df = pd.read_csv("flight_data.csv")

# --- Generate Points (Pins) ---
points = pd.DataFrame({
    'city': df['departure_city'].tolist() + df['destination_city'].tolist(),
    'lat': df['origin_lat'].tolist() + df['dest_lat'].tolist(),
    'lon': df['origin_lon'].tolist() + df['dest_lon'].tolist(),
})

# --- Helper function to create curved paths ---
def great_circle_path(lon1, lat1, lon2, lat2, n_points=30):
    """
    Generate intermediate points between two lat/lon coordinates to simulate a curved path.
    """
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Compute delta
    delta = np.linspace(0, 1, n_points)
    
    # Interpolate using spherical linear interpolation (slerp)
    def slerp(p0, p1, t):
        omega = np.arccos(np.clip(np.dot(p0, p1), -1, 1))
        so = sin(omega)
        return (sin((1 - t) * omega) / so) * p0 + (sin(t * omega) / so) * p1

    # Convert lat/lon to cartesian coordinates
    def to_cartesian(lon, lat):
        return np.array([cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat)])

    start = to_cartesian(lon1, lat1)
    end = to_cartesian(lon2, lat2)
    points = [slerp(start, end, t) for t in delta]

    # Convert back to lat/lon
    def to_latlon(x, y, z):
        lon = np.arctan2(y, x)
        hyp = np.sqrt(x * x + y * y)
        lat = np.arctan2(z, hyp)
        return [np.degrees(lon), np.degrees(lat)]

    return [to_latlon(x, y, z) for x, y, z in points]

# --- Create Curved Paths ---
paths = []
for _, row in df.iterrows():
    arc = great_circle_path(row["origin_lon"], row["origin_lat"], row["dest_lon"], row["dest_lat"])
    paths.append({"path": arc})

# --- PyDeck Layers ---
tile_layer = pdk.Layer("TileLayer", data=None, min_zoom=0, max_zoom=20)

pin_layer = pdk.Layer(
    "ScatterplotLayer",
    data=points,
    get_position='[lon, lat]',
    get_color='[200, 30, 0, 160]',
    get_radius=60000,
    pickable=True
)

# Use PathLayer for curves
path_layer = pdk.Layer(
    "PathLayer",
    data=paths,
    get_path="path",
    get_color=[0, 100, 255, 160],
    width_scale=20,
    width_min_pixels=2
)

# --- Map View ---
view_state = pdk.ViewState(latitude=20, longitude=0, zoom=1.2, min_zoom=0.5, max_zoom=5, pitch=0)

# --- Render Map ---
r = pdk.Deck(
    layers=[tile_layer, path_layer, pin_layer],
    initial_view_state=view_state,
    tooltip={"text": "{city}"}
)

st.pydeck_chart(r)

st.write("# 🇮🇳 🇸🇬 🇺🇸 🇨🇦")

#### Total distance travelled
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r

# Calculate total distance
df["distance_km"] = df.apply(
    lambda row: haversine(row["origin_lon"], row["origin_lat"], row["dest_lon"], row["dest_lat"]), axis=1
)

total_distance = df["distance_km"].sum()
total_flights = len(df)
total_airports = pd.concat([df['departure_city'], df['destination_city']]).nunique()

# --- Most Flown Airline ---
most_flown_airline = df['airline_name'].value_counts().idxmax()
# --Most flown aircraft type--
aircraft_flown = df['aircraft_name'].value_counts().idxmax()

# --- Display Metrics ---
col1, col2, col3, col4,col5 = st.columns(5)
with col1:
    st.metric(label="🛫 Total Flights Taken", value=total_flights)
with col2:
    st.metric(label="🌍 Total Distance Travelled", value=f"{total_distance:,.1f} km")
with col3:
    st.metric(label="✈️ Most Flown Airline", value=f"{most_flown_airline}")
with col4:
    st.metric(label="✈️ Most Flown Aircraft", value=f"{aircraft_flown}")
with col5:
    st.metric(label="🛃 Airports", value=f"{total_airports}")


# --- Top Airlines ---
airline_counts = df['airline_name'].value_counts().reset_index()
airline_counts.columns = ['Airline', 'Flights']
top_airlines = airline_counts

st.subheader("🛫 Top airlines")

# Base bar chart
bars = (
    alt.Chart(top_airlines)
    .mark_bar(color="#ff7f0e")
    .encode(
        x=alt.X("Flights:Q", title=None, axis=None),
        y=alt.Y("Airline:N", sort='-x', title="Airline"),
        tooltip=["Airline", "Flights"]
    )
)

# Text labels on bars
text = bars.mark_text(
    align="left",
    baseline="middle",
    dx=3,  # space from bar
    color="black"
).encode(text="Flights:Q")

# Combine bars + text
airline_bar_chart = (bars + text).properties(height=200, width=300)

st.altair_chart(airline_bar_chart, use_container_width=True)




# --- Data Prep ---
class_counts = df['class'].value_counts().reset_index()
class_counts.columns = ['Class', 'Count']

seat_counts = df['window_middle_aisle'].value_counts().reset_index()
seat_counts.columns = ['Seat Type', 'Count']

# --- Pie Charts with Left Legends (No Titles) ---
class_pie = (
    alt.Chart(class_counts)
    .mark_arc(innerRadius=50)
    .encode(
        theta="Count:Q",
        color=alt.Color("Class:N", legend=alt.Legend(title=None, orient="left")),
        tooltip=["Class", "Count"]
    )
    .properties(width=250, height=250)
)

seat_pie = (
    alt.Chart(seat_counts)
    .mark_arc(innerRadius=50)
    .encode(
        theta="Count:Q",
        color=alt.Color("Seat Type:N", legend=alt.Legend(title=None, orient="left")),
        tooltip=["Seat Type", "Count"]
    )
    .properties(width=250, height=250)
)

# --- Side by Side Layout ---
col1,spacer, col2 = st.columns([1, 0.2, 1]) 

with col1:
    st.subheader("📈 Flight Class")
    st.altair_chart(class_pie, use_container_width=True)

with col2:
    st.subheader("💺 Seat Type")
    st.altair_chart(seat_pie, use_container_width=True)


##########

# Ensure date column is in datetime format
df['date_of_travel'] = pd.to_datetime(df['date_of_travel'],dayfirst= True)

# --- Flights Per Month (Jan to Dec, aggregated across all years) ---
df['month'] = df['date_of_travel'].dt.strftime('%b')
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


flights_per_month = (
    df.groupby('month')
    .size()
    .reindex(month_order)  # Ensures correct month order
    .reset_index(name='Flights')
)

flights_month_chart = (
    alt.Chart(flights_per_month)
    .mark_line(point=True)
    .encode(
        x=alt.X('month:N', title=None, sort=month_order, axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Flights:Q', title='Number of Flights'),
        tooltip=['month', 'Flights']
    )
    .properties(title='Flights per Month', width=500, height=300)
)

# --- Flights Per Weekday ---
df['weekday'] = df['date_of_travel'].dt.day_name()
weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
flights_per_weekday = (
    df.groupby('weekday')
    .size()
    .reindex(weekday_order)
    .reset_index(name='Flights')
)

flights_weekday_chart = (
    alt.Chart(flights_per_weekday)
    .mark_line(point=True)
    .encode(
        x=alt.X('weekday:N', title=None, sort=weekday_order,axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Flights:Q', title='Number of Flights'),
        tooltip=['weekday', 'Flights']
    )
    .properties(title='Flights per Weekday', width=500, height=300)
)

# --- Side by Side Layout in Streamlit ---
col1, spacer, col2 = st.columns([1, 0.2, 1]) 
with col1:
    st.altair_chart(flights_month_chart, use_container_width=True)
with col2:
    st.altair_chart(flights_weekday_chart, use_container_width=True)