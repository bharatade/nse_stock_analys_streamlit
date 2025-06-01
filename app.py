import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, time

# --- Functions ---
def load_and_clean_csv(csv_file):
    df = pd.read_csv(csv_file, header=None, skiprows=1, names=['DateTime', 'Call', 'Put'])
    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%Y-%m-%d %H:%M:%S')
    df['Call'] = pd.to_numeric(df['Call'], errors='coerce')
    df['Put'] = pd.to_numeric(df['Put'], errors='coerce')
    df.dropna(subset=['Call', 'Put'], how='all', inplace=True)
    return df

def calculate_moving_average(df, col, window):
    return df[col].rolling(window=window).mean()

def find_support_resistance(series):
    q_low = series.quantile(0.2)
    q_high = series.quantile(0.8)
    return q_low, q_high

def generate_strategy(df, option_type, settings):
    series = df[option_type]
    ma_col = f"{option_type}_MA"
    df[ma_col] = calculate_moving_average(df, option_type, settings['moving_avg_window'])

    recent_df = df.tail(30)
    price_series = recent_df[option_type].dropna()

    support, resistance = find_support_resistance(price_series)
    buy_price = support
    target_price = max(resistance, buy_price * (1 + settings['profit_margin_percent'] / 100))
    stoploss_price = buy_price * (1 - settings['stoploss_percent'] / 100)

    latest_price = price_series.iloc[-1] if not price_series.empty else 0
    prev_price = price_series.iloc[-2] if len(price_series) > 1 else 0

    buy_condition = latest_price <= support * 1.02 and latest_price > prev_price
    confidence = 0.65 if buy_condition else 0.5
    profit_margin = (target_price - buy_price) / buy_price * 100

    return {
        'Option Type': option_type,
        'Buy Price': buy_price,
        'Stoploss': stoploss_price,
        'Target': target_price,
        'Confidence': confidence,
        'Profit Margin': profit_margin,
        'Support': support,
        'Resistance': resistance,
        'Buy Condition': buy_condition
    }

def plot_strategy(df, call_info, put_info, settings):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['DateTime'], df['Call'], label='Call Price', color='blue')
    ax.plot(df['DateTime'], df['Put'], label='Put Price', color='red')
    ax.plot(df['DateTime'], df['Call'].rolling(settings['moving_avg_window']).mean(), label='Call MA', linestyle='--', color='lightblue')
    ax.plot(df['DateTime'], df['Put'].rolling(settings['moving_avg_window']).mean(), label='Put MA', linestyle='--', color='pink')

    for label, info in [('Call', call_info), ('Put', put_info)]:
        ax.axhline(info['Support'], linestyle='--', alpha=0.4, label=f'{label} Support', color='green' if label == 'Call' else 'darkgreen')
        ax.axhline(info['Resistance'], linestyle='--', alpha=0.4, label=f'{label} Resistance', color='orange')
        ax.axhline(info['Target'], linestyle='--', alpha=0.4, label=f'{label} Target', color='purple')
        ax.axhline(info['Stoploss'], linestyle='--', alpha=0.4, label=f'{label} Stoploss', color='red')

    ax.set_title('📊 Call vs Put Strategy Comparison')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price (INR)')
    ax.legend()
    ax.grid(True)
    return fig

def analyze_both_options(df, settings):
    df = df[(df['DateTime'].dt.time >= pd.to_datetime(settings['start_time'], format='%H:%M').time()) &
            (df['DateTime'].dt.time <= pd.to_datetime(settings['end_time'], format='%H:%M').time())]

    call_info = generate_strategy(df.copy(), 'Call', settings)
    put_info = generate_strategy(df.copy(), 'Put', settings)

    better_option = call_info if call_info['Profit Margin'] > put_info['Profit Margin'] else put_info

    summary = {
        'Recommended Trade': f"📈 Buy {better_option['Option Type']} Option",
        'Buy Price': f"{better_option['Buy Price']:.2f}",
        'Stoploss': f"{better_option['Stoploss']:.2f}",
        'Target': f"{better_option['Target']:.2f}",
        'Profit Margin': f"{better_option['Profit Margin'] - 25:.2f}%",
        'Confidence': f"{better_option['Confidence'] * 100:.0f}%",
        'Reason': f"Based on support/resistance zone of {better_option['Option Type']} prices with recent movement pattern."
    }

    return summary, call_info, put_info, df

# --- Streamlit App ---
st.set_page_config(page_title="📊 NSE Strategy Generator", layout="wide")
st.title("📈 NSE Call/Put Option Strategy Analyzer")
st.markdown("Upload your **NSE CSV File**, configure the parameters, and get a **strategy suggestion** based on historical data.")

# File Upload
uploaded_file = st.file_uploader("📁 Upload NSE CSV file", type=["csv"])

# Parameter Inputs
with st.expander("⚙️ Strategy Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        start_time = st.time_input("Start Time", value=time(9, 16))
        end_time = st.time_input("End Time", value=time(11, 20))
    with col2:
        profit_margin_percent = st.number_input("Profit Margin (%)", value=100.0)
        stoploss_percent = st.number_input("Stoploss (%)", value=7.0)
    with col3:
        moving_avg_window = st.slider("Moving Average Window", 1, 20, 5)
        spike_threshold = st.slider("Spike Threshold (%)", 1, 10, 2)

# Run Analysis
if uploaded_file:
    settings = {
        'start_time': start_time.strftime('%H:%M'),
        'end_time': end_time.strftime('%H:%M'),
        'profit_margin_percent': profit_margin_percent,
        'stoploss_percent': stoploss_percent,
        'moving_avg_window': moving_avg_window,
        'spike_threshold': spike_threshold
    }

    df = load_and_clean_csv(uploaded_file)
    summary, call_info, put_info, filtered_df = analyze_both_options(df, settings)

    st.markdown("---")
    st.subheader("🔍 Final Strategy Suggestion")
    st.success(f"""
**{summary['Recommended Trade']}**  
- **Buy Price**: ₹{summary['Buy Price']}  
- **Stoploss**: ₹{summary['Stoploss']}  
- **Target**: ₹{summary['Target']}  
- **Profit Margin**: {summary['Profit Margin']}  
- **Confidence**: {summary['Confidence']}  
- **Reason**: {summary['Reason']}
""")

    # Show Chart
    st.markdown("---")
    st.subheader("📊 Strategy Chart")
    fig = plot_strategy(filtered_df, call_info, put_info, settings)
    st.pyplot(fig)
