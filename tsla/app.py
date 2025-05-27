import streamlit as st
import pandas as pd
import plotly
import plotly.graph_objects as go
import time
import numpy as np
import ast
import google.generativeai as genai
from datetime import datetime

# Configure Gemini API
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]  # You'll need to add this in Streamlit secrets
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('models/gemini-1.5-flash-latest')

# Set modern page configuration
st.set_page_config(
    page_title="TSLA Stock Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
        .stApp {
            background-color: #0e1117;
        }
        .main {
            background-color: #0e1117;
        }
        h1 {
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
            font-size: 2.5rem !important;
        }
        p {
            color: #9ca3af;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .stButton>button {
            background-color: #1f2937;
            color: #ffffff;
            border-radius: 8px;
            border: 1px solid #374151;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #374151;
            border-color: #4b5563;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            color: #ffffff !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            color: #9ca3af !important;
        }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    DATA_PATH = "TSLA_data - Sheet1.csv"
    data = pd.read_csv(DATA_PATH)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    # Convert string lists to actual lists
    data['Support'] = data['Support'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])
    data['Resistance'] = data['Resistance'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) and x != '[]' else [])
    
    # Calculate additional indicators
    data['MA20'] = data['close'].rolling(window=20).mean()
    data['MA50'] = data['close'].rolling(window=50).mean()
    data['volume_ma'] = data['volume'].rolling(window=20).mean()
    data['price_change'] = data['close'].pct_change() * 100
    data['price_change_abs'] = data['close'].diff()
    
    # Calculate support and resistance bands
    data['support_low'] = data['Support'].apply(lambda x: min(x) if len(x) > 0 else None)
    data['support_high'] = data['Support'].apply(lambda x: max(x) if len(x) > 0 else None)
    data['resistance_low'] = data['Resistance'].apply(lambda x: min(x) if len(x) > 0 else None)
    data['resistance_high'] = data['Resistance'].apply(lambda x: max(x) if len(x) > 0 else None)
    
    return data

data = load_data()

# Modern header section
col1, col2, col3 = st.columns([2,6,2])
with col2:
    st.title("TSLA Stock Analysis Dashboard")
    st.markdown("<p style='text-align: center; font-size: 1.2rem;'>Real-time technical analysis with advanced indicators</p>", unsafe_allow_html=True)

# Key metrics section
latest_data = data.iloc[-1]
prev_data = data.iloc[-2]
metrics_cols = st.columns(5)

with metrics_cols[0]:
    price_change = latest_data['close'] - prev_data['close']
    price_change_pct = (price_change / prev_data['close']) * 100
    st.metric(
        "Current Price",
        f"${latest_data['close']:.2f}",
        f"{price_change_pct:+.2f}%",
        delta_color="normal"
    )

with metrics_cols[1]:
    vol_change = (latest_data['volume'] - prev_data['volume']) / prev_data['volume'] * 100
    st.metric(
        "Volume",
        f"{latest_data['volume']:,.0f}",
        f"{vol_change:+.2f}%"
    )

with metrics_cols[2]:
    st.metric(
        "Day Range",
        f"${latest_data['high']:.2f}",
        f"${latest_data['low']:.2f} Low"
    )

with metrics_cols[3]:
    ma20_change = ((latest_data['MA20'] - prev_data['MA20']) / prev_data['MA20']) * 100
    st.metric(
        "20-day MA",
        f"${latest_data['MA20']:.2f}",
        f"{ma20_change:+.2f}%"
    )

with metrics_cols[4]:
    ma50_change = ((latest_data['MA50'] - prev_data['MA50']) / prev_data['MA50']) * 100
    st.metric(
        "50-day MA",
        f"${latest_data['MA50']:.2f}",
        f"{ma50_change:+.2f}%"
    )

# Chart settings
chart_container = st.container()
settings_cols = st.columns([1,1,1,1])

with settings_cols[0]:
    window_size = st.select_slider(
        "Window Size",
        options=[30, 60, 90, 120, 150, 180],
        value=90,
        help="Number of days to display in the chart"
    )

with settings_cols[1]:
    animation_speed = st.select_slider(
        "Animation Speed",
        options=[0.05, 0.1, 0.15, 0.2, 0.25],
        value=0.15,
        help="Speed of the chart animation"
    )

with settings_cols[2]:
    show_ma = st.multiselect(
        "Moving Averages",
        ["20-day MA", "50-day MA"],
        default=["20-day MA", "50-day MA"],
        help="Select which moving averages to display"
    )

with settings_cols[3]:
    show_zones = st.multiselect(
        "Price Zones",
        ["Support", "Resistance"],
        default=["Support", "Resistance"],
        help="Select which price zones to display"
    )

# Create a placeholder for the chart
chart_placeholder = st.empty()

# Function to create the figure
def create_figure(data_slice):
    fig = go.Figure()
    
    # Add support and resistance zones if selected
    if "Support" in show_zones:
        fig.add_trace(go.Scatter(
            x=data_slice['timestamp'].tolist() + data_slice['timestamp'].tolist()[::-1],
            y=data_slice['support_low'].tolist() + data_slice['support_high'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,255,0,0.1)',
            line=dict(color='rgba(0,255,0,0)'),
            name='Support Zone',
            showlegend=True
        ))
    
    if "Resistance" in show_zones:
        fig.add_trace(go.Scatter(
            x=data_slice['timestamp'].tolist() + data_slice['timestamp'].tolist()[::-1],
            y=data_slice['resistance_low'].tolist() + data_slice['resistance_high'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,0,0,0)'),
            name='Resistance Zone',
            showlegend=True
        ))
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data_slice['timestamp'],
        open=data_slice['open'],
        high=data_slice['high'],
        low=data_slice['low'],
        close=data_slice['close'],
        name='TSLA',
        increasing_line_color='#00C805',
        decreasing_line_color='#FF3737',
        increasing_fillcolor='#00C805',
        decreasing_fillcolor='#FF3737'
    ))
    
    # Add Moving Averages if selected
    if "20-day MA" in show_ma:
        fig.add_trace(go.Scatter(
            x=data_slice['timestamp'],
            y=data_slice['MA20'],
            name='20-day MA',
            line=dict(color='#00B7FF', width=1.5),
            opacity=0.7
        ))
    
    if "50-day MA" in show_ma:
        fig.add_trace(go.Scatter(
            x=data_slice['timestamp'],
            y=data_slice['MA50'],
            name='50-day MA',
            line=dict(color='#FFD700', width=1.5),
            opacity=0.7
        ))
    
    # Add direction arrows
    for idx, row in data_slice.iterrows():
        if pd.notna(row['direction']):
            if row['direction'] == 'LONG':
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['low'] * 0.99],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00C805',
                        line=dict(color='#00C805', width=2)
                    ),
                    name='Long Signal',
                    showlegend=idx == data_slice.index[0]
                ))
            elif row['direction'] == 'SHORT':
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['high'] * 1.01],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='#FF3737',
                        line=dict(color='#FF3737', width=2)
                    ),
                    name='Short Signal',
                    showlegend=idx == data_slice.index[0]
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=[row['timestamp']],
                    y=[row['close']],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=8,
                        color='#FFD700',
                        line=dict(color='#FFD700', width=2)
                    ),
                    name='Neutral',
                    showlegend=idx == data_slice.index[0]
                ))
    
    # Volume bars with improved colors
    colors = ['#00C805' if close >= open_ else '#FF3737' 
              for close, open_ in zip(data_slice['close'], data_slice['open'])]
    
    # Normalize volume for better visualization
    max_vol = data_slice['volume'].max()
    normalized_vol = data_slice['volume'] / max_vol * data_slice['close'].min() * 0.3
    
    fig.add_trace(go.Bar(
        x=data_slice['timestamp'],
        y=normalized_vol,
        marker_color=colors,
        opacity=0.5,
        name='Volume',
        yaxis='y2'
    ))
    
    # Add volume MA
    normalized_vol_ma = data_slice['volume_ma'] / max_vol * data_slice['close'].min() * 0.3
    fig.add_trace(go.Scatter(
        x=data_slice['timestamp'],
        y=normalized_vol_ma,
        name='Volume MA',
        line=dict(color='#9370DB', width=1.5),
        opacity=0.7,
        yaxis='y2'
    ))
    
    # Update layout with modern styling
    fig.update_layout(
        template='plotly_dark',
        title=dict(
            text=f'TSLA Stock Price Movement',
            x=0.5,
            y=0.95,
            font=dict(size=20, color='#ffffff', family='Segoe UI')
        ),
        xaxis=dict(
            title='Date',
            rangeslider=dict(visible=False),
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            showline=True,
            linecolor='#2d374d',
            tickfont=dict(color='#ffffff', size=12, family='Segoe UI'),
            title_font=dict(size=14, color='#ffffff', family='Segoe UI'),
            type='date'
        ),
        yaxis=dict(
            title='Price (USD)',
            showgrid=True,
            gridcolor='rgba(128, 128, 128, 0.15)',
            showline=True,
            linecolor='#2d374d',
            tickfont=dict(color='#ffffff', size=14, family='Segoe UI'),
            title_font=dict(size=16, color='#ffffff', family='Segoe UI'),
            side='left',
            tickformat='$,.0f',
            tickmode='auto',
            nticks=8,
            tickangle=0,
            tickprefix='$',
            separatethousands=True,
            showticklabels=True,
            fixedrange=True
        ),
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False,
            tickfont=dict(color='#ffffff', size=14, family='Segoe UI'),
            title_font=dict(size=16, color='#ffffff', family='Segoe UI'),
            showticklabels=True,
            tickformat=',.0f',
            tickangle=0,
            fixedrange=True
        ),
        height=700,
        margin=dict(l=80, r=80, t=80, b=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(14, 17, 23, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            font=dict(color='#ffffff', size=12, family='Segoe UI'),
            borderwidth=1
        ),
        hovermode='x unified',
        hoverlabel=dict(
            bgcolor='rgba(14, 17, 23, 0.8)',
            font=dict(color='#ffffff', family='Segoe UI'),
            bordercolor='rgba(255, 255, 255, 0.2)'
        ),
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        paper_bgcolor='rgba(14, 17, 23, 0.8)'
    )
    
    # Add range buttons with modern styling
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(step="all", label="ALL")
            ]),
            bgcolor='rgba(14, 17, 23, 0.8)',
            activecolor='#374151',
            font=dict(color='#9ca3af')
        )
    )
    
    return fig

def show_chart_and_chatbot():
    # Chart animation loop
    chart_container = st.container()
    chart_placeholder = st.empty()
    import time as _time  # Avoid conflict with already imported time
    start_time = _time.time()
    try:
        for i in range(len(data) - window_size):
            # Stop animation after 10 seconds
            if _time.time() - start_time > 10:
                break
            data_window = data.iloc[i:i + window_size]
            fig = create_figure(data_window)
            with chart_container:
                chart_placeholder.plotly_chart(fig, use_container_width=True)
            _time.sleep(animation_speed)
            # Stop animation if user navigates away
            if st.session_state.get('page') != 'Dashboard':
                break
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.stop()

    # --- AI Chatbot Section ---
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: #ffffff; font-family: Segoe UI;'>AI Trading Assistant</h2>
            <p style='color: #9ca3af; font-family: Segoe UI;'>Ask me anything about TSLA stock analysis and trading strategies</p>
        </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about TSLA stock analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        latest_price = data.iloc[-1]['close']
        price_change = data.iloc[-1]['price_change']
        volume = data.iloc[-1]['volume']
        ma20 = data.iloc[-1]['MA20']
        ma50 = data.iloc[-1]['MA50']
        context = f"""
        Current TSLA Data:
        - Latest Price: ${latest_price:.2f}
        - Price Change: {price_change:.2f}%
        - Volume: {volume:,.0f}
        - 20-day MA: ${ma20:.2f}
        - 50-day MA: ${ma50:.2f}
        User Question: {prompt}
        Please provide a helpful response about TSLA stock analysis, considering the current market data.
        """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(context)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure you have set up your Google API key in Streamlit secrets.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

def show_home():
    st.header("Home Page")
    st.write("Welcome to the home page! Use the sidebar to navigate.")
    user_input = st.text_input("Enter your name:")
    if user_input:
        st.write(f"Hello, {user_input}! 👋")
    number = st.slider("Select a number", 0, 100, 50)
    st.write(f"You selected: {number}")

def show_data_analysis():
    st.header("Data Analysis")
    st.write("TSLA Data:")
    st.dataframe(data)

    # Candlestick chart using Plotly
    fig = go.Figure(data=[
        go.Candlestick(
            x=data['timestamp'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Candlestick'
        )
    ])
    fig.update_layout(
        title='TSLA Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, use_container_width=True)

def show_agentic_ai():
    st.header("Agentic AI")
    st.write("This section is reserved for future Agentic AI features and tools.")

def show_about():
    st.header("About")
    st.write("""
    This is a sample Streamlit application that demonstrates various features:
    - Interactive widgets
    - Data visualization
    - Multi-page navigation
    - Sidebar functionality
    """)

def show_ai_chatbot():
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center;'>
            <h2 style='color: #ffffff; font-family: Segoe UI;'>AI Trading Assistant</h2>
            <p style='color: #9ca3af; font-family: Segoe UI;'>Ask me anything about TSLA stock analysis and trading strategies</p>
        </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about TSLA stock analysis..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        latest_price = data.iloc[-1]['close']
        price_change = data.iloc[-1]['price_change']
        volume = data.iloc[-1]['volume']
        ma20 = data.iloc[-1]['MA20']
        ma50 = data.iloc[-1]['MA50']
        context = f"""
        Current TSLA Data:
        - Latest Price: ${latest_price:.2f}
        - Price Change: {price_change:.2f}%
        - Volume: {volume:,.0f}
        - 20-day MA: ${ma20:.2f}
        - 50-day MA: ${ma50:.2f}
        User Question: {prompt}
        Please provide a helpful response about TSLA stock analysis, considering the current market data.
        """
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content(context)
                    st.markdown(response.text)
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure you have set up your Google API key in Streamlit secrets.")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "AI Chatbot"])
st.session_state['page'] = page

# Main content
if page == "Dashboard":
    show_chart_and_chatbot()
elif page == "AI Chatbot":
    show_ai_chatbot() 