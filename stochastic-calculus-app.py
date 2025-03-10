import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from PIL import Image
from io import BytesIO
import time

st.set_page_config(
    page_title="Stochastic Calculus Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# (Removed the MathJax script block since we use st.latex)

# Custom CSS to make the app look better
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #0e1117;
    }
    .disclaimer {
        font-size: 0.8em;
        color: #888;
        border-top: 1px solid #ddd;
        padding-top: 10px;
        margin-top: 20px;
    }
    .css-1v3fvcr {
        padding-left: 10px;
        padding-right: 10px;
    }
    .formula-box, .note-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4e8cff;
        overflow-x: auto;
    }
    .note-box {
        background-color: #e7f0fd;
    }
    .cc-license {
        text-align: center;
        margin: 20px 0;
    }
    @media (max-width: 768px) {
        .formula-box {
            padding: 10px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to generate Brownian motion
def generate_brownian_motion(T=1.0, N=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # Start at 0
    t = np.linspace(0, T, N+1)
    return t, W

# Function to generate geometric Brownian motion
def generate_gbm(S0, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        S[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * W[i])
    return t, S

# Function to generate mean-reverting process (Ornstein-Uhlenbeck)
def generate_mean_reverting(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    dt = T / N
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        S[i] = S[i-1] + (nu - mu * S[i-1]) * dt + sigma * (W[i] - W[i-1])
    return t, S

# Function to generate CIR process
def generate_cir(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    dt = T / N
    S = np.zeros(N+1)
    S[0] = S0
    for i in range(1, N+1):
        if S[i-1] < 0:
            S[i-1] = 0  # Ensure non-negativity
        S[i] = S[i-1] + (nu - mu * S[i-1]) * dt + sigma * np.sqrt(max(0, S[i-1])) * (W[i] - W[i-1])
    return t, S

# Function to create interactive plot with Plotly
def plot_stochastic_process(t, S, title, ylabel):
    fig = px.line(x=t, y=S, labels={'x': 'Time', 'y': ylabel})
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title=ylabel,
        template='plotly_white',
        height=500,
    )
    return fig

# Create CC BY-NC license image
def get_cc_image():
    buffer = BytesIO()
    img = Image.new('RGB', (120, 40), (255, 255, 255))
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

# App title
st.title("Elementary Stochastic Calculus Explorer")

# Sidebar navigation - before author info
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to section:",
    ["Introduction", 
     "Markov & Martingale Properties",
     "Brownian Motion", 
     "Stochastic Integration",
     "Stochastic Differential Equations",
     "Itô's Lemma",
     "Common Stochastic Processes",
     "Interactive Simulation"]
)

# Author info after navigation
st.write("By Luís Simões da Cunha")

# Add disclaimer and license information
st.sidebar.markdown("""
---
### About

This interactive application explores the foundations of stochastic calculus and its applications in financial mathematics.

#### Disclaimer
The content provided in this application is for educational purposes only. While efforts have been made to ensure accuracy, no guarantee is made regarding the correctness of information or code examples. This is not financial advice. Users should verify all information independently before use in any financial or investment context.

The author is not a financial advisor and this content should not be considered investment advice.
""")

st.sidebar.markdown("""
---
<div class="cc-license">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</div>
""", unsafe_allow_html=True)

# ----------------------
# Section: Introduction
# ----------------------
if section == "Introduction":
    st.header("Introduction to Stochastic Calculus")
    st.markdown("""
Stochastic calculus is a key mathematical framework for understanding and modeling random processes,
especially in the realm of financial mathematics. It provides the foundation for pricing derivatives,
managing risk, and analyzing market dynamics.

At its core, stochastic calculus deals with the manipulation of random quantities and the solution 
of stochastic differential equations. While the mathematics can appear intimidating, this interactive
application aims to build intuition and practical understanding.
""")
    st.markdown("""
**Why is stochastic calculus important in finance?**  
Financial markets exhibit random behavior that cannot be adequately modeled by deterministic equations.
Stochastic calculus provides the tools to model this randomness and understand how it propagates
through various financial instruments.
""")
    st.header("What You'll Learn")
    st.markdown("""
- The meaning of **Markov** and **martingale** processes  
- **Brownian motion** and its properties  
- **Stochastic integration** and how it differs from regular calculus  
- **Stochastic differential equations** and their interpretation  
- **Itô's lemma** - the cornerstone of stochastic calculus  
- Common stochastic processes used in financial modeling
""")
    st.header("Randomness in Financial Markets")
    col1, col2 = st.columns(2)
    with col1:
        seed = np.random.randint(0, 10000)
        t, bm = generate_brownian_motion(seed=seed)
        fig_bm = plot_stochastic_process(t, bm, "Brownian Motion", "Value")
        st.plotly_chart(fig_bm, use_container_width=True)
    with col2:
        t, gbm = generate_gbm(100, 0.05, 0.2, seed=seed)
        fig_gbm = plot_stochastic_process(t, gbm, "Stock Price Model", "Price")
        st.plotly_chart(fig_gbm, use_container_width=True)
    st.markdown("These plots illustrate the random nature of financial processes. The left plot shows pure randomness (Brownian motion), while the right shows a common model for stock prices that incorporates both random fluctuations and a growth trend.")

# ----------------------------------------
# Section: Markov & Martingale Properties
# ----------------------------------------
elif section == "Markov & Martingale Properties":
    st.header("The Markov Property")
    st.markdown("""
The Markov property is a fundamental concept in stochastic processes. A process has the Markov property if 
the conditional probability distribution of future states depends only on the present state, not on the sequence 
of states that preceded it.

In simple terms, a Markov process has no memory beyond its current state. The future behavior of the process
is entirely determined by where it is now, not how it got there.
""")
    st.markdown("For a stochastic process $S_i$, the Markov property states that:")
    st.latex(r"E[S_i \mid S_1, S_2, \dots, S_{i-1}] = E[S_i \mid S_{i-1}]")
    
    st.header("The Martingale Property")
    st.markdown("""
A martingale is a stochastic process where the expected value of the next observation, given all past 
observations, is equal to the most recent observation.
""")
    st.markdown("For a stochastic process $S_i$, the martingale property states that:")
    st.latex(r"E[S_i \mid S_j,\, j < i] = S_j")
    
    st.markdown("""
In financial terms, a martingale represents a "fair game" where no information available at the present time 
can be used to predict future gains or losses. The martingale property is closely related to the efficient market hypothesis.
""")
    
    st.header("Interactive Example: Coin Flipping")
    st.markdown("""
Consider a coin flipping experiment:
- Each "Head" gives you $1  
- Each "Tail" costs you $1  
- Your total winnings have both the Markov and martingale properties
""")
    num_flips = st.slider("Number of coin flips", 5, 100, 20)
    if st.button("Flip coins"):
        flips = np.random.choice([-1, 1], size=num_flips)
        cumulative_winnings = np.cumsum(flips)
        cumulative_winnings = np.insert(cumulative_winnings, 0, 0)
        df = pd.DataFrame({
            'Flip': range(num_flips + 1),
            'Winnings': cumulative_winnings
        })
        fig = px.line(df, x='Flip', y='Winnings', markers=True)
        fig.update_layout(
            title="Coin Flipping Experiment",
            xaxis_title="Number of Flips",
            yaxis_title="Cumulative Winnings",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
**Observations:**
- The expected winnings at any future time is just the current amount (martingale property)  
- Only the current winnings matter for future expectations, not how you got there (Markov property)  
- The path is unpredictable, illustrating the random nature of the process
""")

# -------------------
# Section: Brownian Motion
# -------------------
elif section == "Brownian Motion":
    st.header("Brownian Motion")
    st.markdown("""
Brownian motion (also called a Wiener process) is one of the most fundamental stochastic processes. 
It is a continuous-time process with three defining properties:
1. It starts at zero: 
""")
    st.latex(r"W(0) = 0")
    st.markdown("""
2. It has independent increments.  
3. The increments follow a normal distribution with mean 0 and variance equal to the time difference.
""")
    st.markdown("Additionally, the paths of $W(t)$ are continuous but nowhere differentiable.")
    
    st.header("Interactive Brownian Motion Simulation")
    col1, col2 = st.columns([1, 3])
    with col1:
        T = st.slider("Time horizon", 0.1, 5.0, 1.0, 0.1)
        N = st.slider("Number of steps", 100, 5000, 1000, 100)
        num_paths = st.slider("Number of paths", 1, 10, 3)
        seed = st.number_input("Random seed (optional)", value=None, min_value=0, max_value=10000, step=1)
    with col2:
        fig = go.Figure()
        for i in range(num_paths):
            np_seed = seed + i if seed is not None else None
            t, W = generate_brownian_motion(T, N, np_seed)
            fig.add_trace(go.Scatter(x=t, y=W, mode='lines', name=f'Path {i+1}'))
        fig.update_layout(
            title='Brownian Motion Paths',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
**Key observations about Brownian motion:**
- The paths are continuous but very erratic  
- Over time, the paths tend to spread out (variance increases with time)  
- Each path is unpredictable and can deviate significantly from others  
- Despite the randomness, Brownian motion has well-defined statistical properties
""")
    
    st.header("Quadratic Variation")
    st.markdown("An important property of Brownian motion is its quadratic variation. For a partition of the interval [0,t] into n subintervals, the sum of squared increments converges to t as the partition gets finer:")
    st.latex(r"\sum_{j=1}^{n} \left(W(t_j) - W(t_{j-1})\right)^2 \to t \quad \text{as } n \to \infty")
    st.markdown("This property is crucial for developing stochastic calculus as it leads to the rule $dW^2 = dt$, which is fundamental to Itô's lemma.")

# --------------------------
# Section: Stochastic Integration
# --------------------------
elif section == "Stochastic Integration":
    st.header("Stochastic Integration")
    st.markdown("""
Stochastic integration extends the concept of integration to include random processes. The most common form is the Itô integral, which integrates a deterministic function against a Brownian motion.
""")
    st.markdown("The Itô integral is defined as:")
    st.latex(r"\int_0^t f(\tau) \, dW(\tau) = \lim_{n \to \infty} \sum_{j=1}^{n} f(t_{j-1})\left(W(t_j) - W(t_{j-1})\right)")
    st.markdown("""
Key differences from ordinary integration include:
1. The integrand is evaluated at the left endpoint (non-anticipatory property).  
2. The integral’s properties differ from Riemann integration.  
3. Standard calculus rules (like the chain rule) are modified.
""")
    st.header("Non-Anticipatory Property")
    st.markdown("""
A crucial aspect of stochastic integration is that it is non-anticipatory. This means when evaluating the integrand, no information about future values of the Brownian motion is used. In finance, this ensures that current actions (e.g., portfolio choices) do not depend on future price movements.
""")
    st.header("Visualizing Stochastic Integration")
    t, W = generate_brownian_motion(1.0, 1000, seed=42)
    f = np.sin(5 * t)
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=("Function $f(t) = \sin(5t)$", "Brownian Motion $W(t)$"))
    fig.add_trace(go.Scatter(x=t, y=f, mode='lines', name='f(t)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=W, mode='lines', name='W(t)'), row=2, col=1)
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
The stochastic integral 
$$\int_0^t f(\tau) \, dW(\tau)$$ 
can be interpreted as accumulating the product of the function value and the increments of Brownian motion. Note that the function is evaluated at the beginning of each time interval, before the Brownian increment is known.
""")
    st.markdown("""
**Financial interpretation:**  
If $W(t)$ represents the randomness in a stock price and $f(t)$ represents an investment strategy, then 
$$\int_0^t f(\tau) \, dW(\tau)$$ 
represents cumulative gains or losses from that strategy.
""")

# -------------------------------
# Section: Stochastic Differential Equations
# -------------------------------
elif section == "Stochastic Differential Equations":
    st.header("Stochastic Differential Equations")
    st.markdown("""
Stochastic Differential Equations (SDEs) are differential equations in which one or more terms are stochastic processes.
They form the mathematical framework for modeling systems with random components.
""")
    st.markdown("A general form of an SDE is:")
    st.latex(r"dS = a(S,t) \, dt + b(S,t) \, dW")
    st.markdown("""
where:
- $S$ is the variable being modeled (e.g. stock price)
- $t$ is time  
- $a(S,t)$ is the drift (deterministic part)  
- $b(S,t)$ is the diffusion (random part)  
- $W$ is a Brownian motion
""")
    st.header("Interpreting SDEs")
    st.markdown("The equivalent integral form is:")
    st.latex(r"S(t) = S(0) + \int_0^t a(S(\tau),\tau) \, d\tau + \int_0^t b(S(\tau),\tau) \, dW(\tau)")
    st.markdown("""
This shows that $S(t)$ consists of:
1. The initial value $S(0)$  
2. A deterministic integral (cumulative drift)  
3. A stochastic integral (cumulative random effects)
""")
    st.header("Comparing Different SDEs")
    col1, col2 = st.columns(2)
    with col1:
        sde_type = st.selectbox(
            "Select SDE type",
            ["Geometric Brownian Motion", "Mean-Reverting (Vasicek)", "Cox-Ingersoll-Ross (CIR)"]
        )
    with col2:
        seed = st.number_input("Random seed", value=42, min_value=1, max_value=10000, step=1)
    
    if sde_type == "Geometric Brownian Motion":
        st.markdown("Geometric Brownian Motion (GBM):")
        st.latex(r"dS = \mu S \, dt + \sigma S \, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 1.0, 100.0, 50.0, 1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
        with col2:
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
            num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig = go.Figure()
        for i in range(num_paths):
            t, S = generate_gbm(S0, mu, sigma, seed=seed+i)
            fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name=f'Path {i+1}'))
        fig.update_layout(
            title='Geometric Brownian Motion',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Key properties of GBM include:")
        st.markdown("- Used to model stock prices and other financial assets")
        st.markdown("- Always positive")
        st.markdown("- Percentage changes are normally distributed")
        st.markdown("Closed-form solution:")
        st.latex(r"S(t) = S(0)e^{\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t)}")
        
    elif sde_type == "Mean-Reverting (Vasicek)":
        st.markdown("Vasicek Model (Mean-Reverting):")
        st.latex(r"dS = (\nu - \mu S) \, dt + \sigma \, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean (ν)", 0.01, 5.0, 1.0, 0.01)
        with col2:
            mu = st.slider("Reversion rate (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig = go.Figure()
        for i in range(num_paths):
            t, S = generate_mean_reverting(S0, nu, mu, sigma, seed=seed+i)
            fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name=f'Path {i+1}'))
        fig.add_hline(y=nu/mu, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        fig.update_layout(
            title='Vasicek Mean-Reverting Process',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Key properties:")
        st.markdown("- Reverts to a long-term mean (ν/μ)")
        st.markdown("- Can become negative")
        st.markdown("- Speed of reversion is proportional to the distance from the mean")
        
    elif sde_type == "Cox-Ingersoll-Ross (CIR)":
        st.markdown("Cox-Ingersoll-Ross (CIR) Model:")
        st.latex(r"dS = (\nu - \mu S) \, dt + \sigma \sqrt{S} \, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean factor (ν)", 0.01, 5.0, 1.0, 0.01)
        with col2:
            mu = st.slider("Reversion rate (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig = go.Figure()
        for i in range(num_paths):
            t, S = generate_cir(S0, nu, mu, sigma, seed=seed+i)
            fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name=f'Path {i+1}'))
        fig.add_hline(y=nu/mu, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        fig.update_layout(
            title='Cox-Ingersoll-Ross Process',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("Key properties:")
        st.markdown("- Always non-negative")
        st.markdown("- Reverts to a long-term mean (ν/μ)")
        st.markdown("- Under certain conditions, the process never reaches zero")
    
    st.subheader("Comparing Different Stochastic Processes")
    if st.button("Generate Comparison"):
        seed = 42
        t_bm, W = generate_brownian_motion(1.0, 1000, seed)
        S_bm = 1.0 + 0.1 * np.linspace(0,1,1001) + 0.2 * W
        t_gbm, S_gbm = generate_gbm(1.0, 0.1, 0.2, 1.0, 1000, seed)
        t_vas, S_vas = generate_mean_reverting(1.0, 0.1, 0.5, 0.2, 1.0, 1000, seed)
        t_cir, S_cir = generate_cir(1.0, 0.1, 0.5, 0.2, 1.0, 1000, seed)
        fig = make_subplots(rows=2, cols=2, 
                            subplot_titles=("Brownian Motion with Drift", "Geometric Brownian Motion", 
                                            "Vasicek Process", "CIR Process"))
        fig.add_trace(go.Scatter(x=t_bm, y=S_bm, mode='lines'), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_gbm, y=S_gbm, mode='lines'), row=1, col=2)
        fig.add_trace(go.Scatter(x=t_vas, y=S_vas, mode='lines'), row=2, col=1)
        fig.add_trace(go.Scatter(x=t_cir, y=S_cir, mode='lines'), row=2, col=2)
        fig.add_hline(y=0.1/0.5, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0.1/0.5, line_dash="dash", line_color="red", row=2, col=2)
        fig.update_layout(
            height=800,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
**Observations:**
- Brownian motion can become negative  
- GBM remains positive and tends to grow exponentially  
- Vasicek and CIR both revert to their long-term mean  
- CIR cannot go negative, unlike Vasicek  
- The choice of process should match the properties of the financial quantity being modeled
""")

elif section == "Common Stochastic Processes":
    st.header("Common Stochastic Processes in Finance")

    st.markdown("""
    Financial markets are modeled using various stochastic processes,
    each with unique properties that make them suitable for different
    assets or market conditions. Below are some of the most commonly used:
    """)

    # Choose which process to explore
    process = st.selectbox(
        "Select a stochastic process to explore:",
        [
            "Brownian Motion with Drift",
            "Geometric Brownian Motion",
            "Mean-Reverting (Vasicek)",
            "Cox-Ingersoll-Ross (CIR)"
        ]
    )

    # ---- BROWNIAN MOTION WITH DRIFT ----
    if process == "Brownian Motion with Drift":
        st.subheader("Brownian Motion with Drift")
        st.latex(r"dS = \mu \, dt + \sigma \, dW")

        st.markdown("""
        **Interpretation**  
        This is a simple extension of standard Brownian motion with an added drift term \\(\mu\\). 
        While it can become negative (unsuitable for strictly non-negative quantities like stock prices), 
        it is useful for modeling processes that can span the entire real line.
        """)

        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Drift (μ) for BM", -0.5, 0.5, 0.1, 0.05)
        with col2:
            sigma = st.slider("Volatility (σ) for BM", 0.01, 1.0, 0.3, 0.05)

        # Generate & plot Brownian motion with drift
        t, W = generate_brownian_motion(T=1.0, N=1000, seed=42)
        S = mu * t + sigma * W  # Starting from 0 for simplicity

        fig = px.line(x=t, y=S)
        fig.update_layout(
            title="Brownian Motion with Drift",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key properties**  
        - Can be both positive and negative  
        - Has a linear trend determined by the drift \\(\mu\\)  
        - Variance grows linearly with time  
        - A simple closed-form: \\(S(t) = S(0) + \mu t + \sigma W(t)\\)
        """)

    # ---- GEOMETRIC BROWNIAN MOTION ----
    elif process == "Geometric Brownian Motion":
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"dS = \mu S \, dt + \sigma S \, dW")

        st.markdown("""
        **Interpretation**  
        GBM is the most widely used model for stock prices because the process remains positive 
        and percentage changes follow a normal distribution. It is the foundation of the 
        Black–Scholes–Merton option pricing model.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.slider("Initial value (S₀) for GBM", 10.0, 200.0, 100.0, 10.0)
        with col2:
            mu = st.slider("Drift (μ) for GBM", -0.5, 0.5, 0.05, 0.05)
        with col3:
            sigma = st.slider("Volatility (σ) for GBM", 0.05, 0.5, 0.2, 0.05)

        t, S = generate_gbm(S0, mu, sigma, T=1.0, N=1000, seed=42)

        # Plot the GBM path
        fig = px.line(x=t, y=S)
        fig.update_layout(
            title="Geometric Brownian Motion",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # Also plot the log of the process
        fig_log = px.line(x=t, y=np.log(S))
        fig_log.update_layout(
            title="Log of Geometric Brownian Motion",
            xaxis_title="Time",
            yaxis_title="Log Value",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_log, use_container_width=True)

        st.markdown(r"""
        **Key properties**  
        - Always positive (suitable for asset prices)  
        - Percentage changes (log-returns) are normally distributed  
        - Has log-normal distribution at any fixed time  
        - A closed-form solution: 
        \[
        S(t) = S(0)\,\exp\Bigl((\mu - \tfrac{\sigma^2}{2})\,t + \sigma\,W(t)\Bigr)
        \]
        """)

    # ---- VASICEK (MEAN-REVERTING) ----
    elif process == "Mean-Reverting (Vasicek)":
        st.subheader("Mean-Reverting (Vasicek) Process")
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma\, dW")

        st.markdown("""
        **Interpretation**  
        The Vasicek model is a classic mean-reverting process often used to model interest rates. 
        The drift term \\((\\nu - \\mu S)\\) pulls the process toward the long-term mean \\(\\nu / \\mu\\).
        """)

        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀) for Vasicek", 0.01, 2.0, 1.0, 0.01)
            nu = st.slider("Long-term mean (ν)", 0.1, 2.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion speed (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma_ = st.slider("Volatility (σ)", 0.05, 0.5, 0.2, 0.05)

        t, S_vas = generate_mean_reverting(S0, nu, mu_, sigma_, T=1.0, N=1000, seed=42)

        fig = px.line(x=t, y=S_vas)
        fig.add_hline(y=nu/mu_, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        fig.update_layout(
            title="Vasicek Mean-Reverting Process",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(r"""
        **Key properties**  
        - Tends to revert to long-term mean \( \nu / \mu \)  
        - Can go negative (limitation for interest rates)  
        - Reversion speed determines how quickly it returns to the mean
        """)

    # ---- CIR (MEAN-REVERTING, NON-NEGATIVE) ----
    elif process == "Cox-Ingersoll-Ross (CIR)":
        st.subheader("Cox-Ingersoll-Ross (CIR) Model")
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma \sqrt{S}\, dW")

        st.markdown("""
        **Interpretation**  
        The CIR model improves upon Vasicek by ensuring the process remains non-negative, making it more realistic for
        modeling interest rates and other quantities that cannot go below zero. As \\(S\\) approaches zero, volatility also
        shrinks, preventing negative values.
        """)

        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 0.01, 2.0, 1.0, 0.01)
            nu = st.slider("Long-term mean factor (ν)", 0.1, 2.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion speed (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma_ = st.slider("Volatility (σ)", 0.05, 0.5, 0.2, 0.05)

        t, S_cir = generate_cir(S0, nu, mu_, sigma_, T=1.0, N=1000, seed=42)

        fig = px.line(x=t, y=S_cir)
        fig.add_hline(y=nu/mu_, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        fig.update_layout(
            title="Cox-Ingersoll-Ross Process",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(r"""
        **Key properties**  
        - Always non-negative (volatility goes to zero near 0)  
        - Mean-reverts to \( \nu / \mu \)  
        - If \(2\,\nu\,\mu \ge \sigma^2\), the process never hits zero  
        """)

    # ---- COMPARISON BUTTON ----
    st.subheader("Comparing Different Stochastic Processes")
    if st.button("Generate Comparison"):
        # Use the same random seed for consistent comparisons
        seed = 42

        # Brownian motion with drift
        t_bm, W_bm = generate_brownian_motion(T=1.0, N=1000, seed=seed)
        S_bm = 1.0 + 0.1 * t_bm + 0.2 * W_bm  # Example parameters

        # Geometric Brownian Motion
        t_gbm, S_gbm = generate_gbm(S0=1.0, mu=0.1, sigma=0.2, T=1.0, N=1000, seed=seed)

        # Vasicek
        t_vas, S_vas = generate_mean_reverting(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)

        # CIR
        t_cir, S_cir = generate_cir(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)

        fig_comp = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Brownian Motion w/ Drift",
                "Geometric Brownian Motion",
                "Vasicek Process",
                "CIR Process"
            )
        )
        # Plot each process
        fig_comp.add_trace(go.Scatter(x=t_bm, y=S_bm, mode='lines', name="BM w/ drift"), row=1, col=1)
        fig_comp.add_trace(go.Scatter(x=t_gbm, y=S_gbm, mode='lines', name="GBM"), row=1, col=2)
        fig_comp.add_trace(go.Scatter(x=t_vas, y=S_vas, mode='lines', name="Vasicek"), row=2, col=1)
        fig_comp.add_trace(go.Scatter(x=t_cir, y=S_cir, mode='lines', name="CIR"), row=2, col=2)

        # Add mean lines for mean-reverting processes
        fig_comp.add_hline(
            y=0.1 / 0.5, line_dash="dash", line_color="red",
            row=2, col=1, annotation_text="Long-term mean"
        )
        fig_comp.add_hline(
            y=0.1 / 0.5, line_dash="dash", line_color="red",
            row=2, col=2, annotation_text="Long-term mean"
        )

        fig_comp.update_layout(
            height=800,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        st.markdown("""
        **Observations**  
        - Brownian motion with drift can become negative.  
        - GBM remains positive and tends to grow exponentially.  
        - Vasicek and CIR both revert to their long-term mean, but CIR stays non-negative.  
        - Choose the process that matches the real-world behavior of the quantity being modeled.
        """)




# ---------------------------
# Section: Interactive Simulation
# ---------------------------
elif section == "Interactive Simulation":
    st.header("Interactive Stochastic Process Simulator")
    st.markdown("""
This simulator allows you to generate and visualize various stochastic processes with your chosen parameters.
Observe how parameter changes affect the process behavior.
""")
    process_type = st.selectbox(
        "Select process type",
        ["Geometric Brownian Motion (GBM)", "Mean-Reverting (Vasicek)", "Cox-Ingersoll-Ross (CIR)"]
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        S0 = st.number_input("Initial value (S₀)", value=100.0, min_value=0.1, step=1.0)
        T = st.number_input("Time horizon", value=1.0, min_value=0.1, step=0.1)
    with col2:
        N = st.number_input("Number of steps", value=1000, min_value=100, step=100)
        num_paths = st.number_input("Number of paths", value=5, min_value=1, max_value=20)
    with col3:
        if process_type == "Geometric Brownian Motion (GBM)":
            mu = st.number_input("Drift (μ)", value=0.05, min_value=-0.5, max_value=0.5, step=0.01)
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
        else:
            nu = st.number_input("Long-term mean factor (ν)", value=0.05, min_value=0.01, max_value=0.2, step=0.01)
            mu = st.number_input("Reversion speed (μ)", value=0.5, min_value=0.1, max_value=5.0, step=0.1)
            sigma = st.number_input("Volatility (σ)", value=0.2, min_value=0.01, max_value=1.0, step=0.01)
    seed = st.number_input("Random seed (optional)", value=42, min_value=1, step=1)
    if st.button("Generate Simulation"):
        fig = go.Figure()
        for i in range(num_paths):
            if process_type == "Geometric Brownian Motion (GBM)":
                t, S = generate_gbm(S0, mu, sigma, T, N, seed + i)
            elif process_type == "Mean-Reverting (Vasicek)":
                t, S = generate_mean_reverting(S0, nu, mu, sigma, T, N, seed + i)
            elif process_type == "Cox-Ingersoll-Ross (CIR)":
                t, S = generate_cir(S0, nu, mu, sigma, T, N, seed + i)
            fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name=f'Path {i+1}'))
        if process_type != "Geometric Brownian Motion (GBM)":
            fig.add_hline(y=nu/mu, line_dash="dash", line_color="red", 
                          annotation=dict(text="Long-term mean"))
        fig.update_layout(
            title=f'{process_type} Simulation',
            xaxis_title='Time',
            yaxis_title='Value',
            height=600,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
        if num_paths > 1:
            all_paths = np.zeros((num_paths, N+1))
            for i in range(num_paths):
                if process_type == "Geometric Brownian Motion (GBM)":
                    _, path = generate_gbm(S0, mu, sigma, T, N, seed + i)
                elif process_type == "Mean-Reverting (Vasicek)":
                    _, path = generate_mean_reverting(S0, nu, mu, sigma, T, N, seed + i)
                elif process_type == "Cox-Ingersoll-Ross (CIR)":
                    _, path = generate_cir(S0, nu, mu, sigma, T, N, seed + i)
                all_paths[i] = path
            mean_path = np.mean(all_paths, axis=0)
            std_path = np.std(all_paths, axis=0)
            min_path = np.min(all_paths, axis=0)
            max_path = np.max(all_paths, axis=0)
            fig_stats = go.Figure()
            fig_stats.add_trace(go.Scatter(x=t, y=mean_path, mode='lines', name='Mean',
                                           line=dict(color='blue', width=2)))
            fig_stats.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([mean_path + std_path, (mean_path - std_path)[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,255,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Mean ± 1 Std Dev'
            ))
            fig_stats.add_trace(go.Scatter(x=t, y=min_path, mode='lines', name='Min',
                                           line=dict(color='red', width=1, dash='dash')))
            fig_stats.add_trace(go.Scatter(x=t, y=max_path, mode='lines', name='Max',
                                           line=dict(color='green', width=1, dash='dash')))
            fig_stats.update_layout(
                title='Statistics Across Paths',
                xaxis_title='Time',
                yaxis_title='Value',
                height=500,
                template='plotly_white'
            )
            st.plotly_chart(fig_stats, use_container_width=True)
            final_values = all_paths[:, -1]
            fig_hist = px.histogram(final_values, nbins=20, 
                                    labels={'value': 'Final Value', 'count': 'Frequency'},
                                    title='Distribution of Final Values')
            fig_hist.update_layout(template='plotly_white')
            st.plotly_chart(fig_hist, use_container_width=True)
            st.subheader("Summary Statistics of Final Values")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Standard Deviation', 'Minimum', 'Maximum'],
                'Value': [
                    f"{np.mean(final_values):.4f}",
                    f"{np.median(final_values):.4f}",
                    f"{np.std(final_values):.4f}",
                    f"{np.min(final_values):.4f}",
                    f"{np.max(final_values):.4f}"
                ]
            })
            st.table(stats_df)
    st.markdown("""
**Understanding the simulation:**
- Each path represents a possible future trajectory.
- Variability between paths shows uncertainty in future outcomes.
- Aggregated statistics give insight into expected behavior and risk.
- Parameter choices significantly affect process behavior.
""")
    st.markdown("""
**Financial Applications:**
- Monte Carlo methods for option pricing  
- Risk management and Value-at-Risk (VaR) calculations  
- Asset allocation and portfolio optimization  
- Stress testing financial models
""")

# -----------------
# Section: Itô's Lemma
# -----------------
elif section == "Itô's Lemma":
    st.header("Itô's Lemma")
    st.markdown("""
Itô's lemma is the cornerstone of stochastic calculus. It is the stochastic analogue of the chain rule, telling us how to differentiate functions of stochastic processes.
""")
    st.markdown("For a function $F(S,t)$ where $S$ follows the SDE $dS = a(S,t)\, dt + b(S,t)\, dW$, Itô's lemma states:")
    st.latex(r"""
dF = \left(\frac{\partial F}{\partial t} + a\frac{\partial F}{\partial S} + \frac{1}{2}b^2\frac{\partial^2 F}{\partial S^2}\right)dt + b\frac{\partial F}{\partial S} \, dW
""")
    st.markdown("""
The key difference from ordinary calculus is the second derivative term, which arises due to the quadratic variation of Brownian motion (i.e. $dW^2 = dt$).
""")
    st.header("Intuition Behind Itô's Lemma")
    st.markdown("""
A rule of thumb for applying Itô's lemma:
1. Taylor expand the function.
2. Keep terms up to second order in $dS$.
3. Replace $dW^2$ with $dt$.
""")
    st.markdown("""
This approach explains why the extra term appears in the stochastic setting.
""")
    st.header("Example: Applying Itô's Lemma")
    st.markdown("Consider a stock price following geometric Brownian motion:")
    st.latex(r"dS = \mu S \, dt + \sigma S \, dW")
    st.markdown("Now, apply Itô's lemma to find the SDE for $F(S)=\log(S)$. First, compute the derivatives:")
    st.latex(r"""
\begin{aligned}
\frac{\partial F}{\partial S} &= \frac{1}{S},\\[6pt]
\frac{\partial^2 F}{\partial S^2} &= -\frac{1}{S^2},\\[6pt]
\frac{\partial F}{\partial t} &= 0.
\end{aligned}
""")
    st.markdown("Then, applying Itô's lemma gives:")
    st.latex(r"""
\begin{aligned}
d(\log S) &= \left(\mu S\cdot\frac{1}{S} + \frac{1}{2}\sigma^2 S^2\cdot\left(-\frac{1}{S^2}\right)\right)dt + \sigma S\cdot\frac{1}{S}dW\\[6pt]
&= \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma dW.
\end{aligned}
""")
    st.markdown("""
This result shows that while $S$ follows a geometric Brownian motion, $\log(S)$ follows a Brownian motion with drift.
""")
    st.markdown("""
**Important implication:**  
Log-returns follow a normal distribution—a common assumption in financial modeling.
""")
    st.header("Visual Demonstration")
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    t, S = generate_gbm(S0, mu, sigma, seed=42)
    log_S = np.log(S)
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=('Stock Price $S(t)$', 'Log Stock Price $\log(S(t))$'))
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='S(t)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=log_S, mode='lines', name='log(S(t))'), row=2, col=1)
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Notice that the logarithm transformation makes the process appear more linear with added noise, consistent with the derived SDE for $\log(S)$.")
    st.header("Itô's Lemma in Multiple Dimensions")
    st.markdown("""
Itô's lemma extends to functions of multiple stochastic variables. For a function $F(S_1, S_2, \dots, t)$ where each $S_i$ satisfies 
$$dS_i = a_i\, dt + b_i\, dW_i,$$ 
and where $dW_i\, dW_j = \rho_{ij}\, dt$, we have:
""")
    st.latex(r"""
dF = \frac{\partial F}{\partial t}dt + \sum_i \frac{\partial F}{\partial S_i}dS_i + \frac{1}{2}\sum_i\sum_j \rho_{ij} b_i b_j \frac{\partial^2 F}{\partial S_i \partial S_j}dt.
""")
    st.markdown("This multidimensional version is crucial for pricing options on baskets or modeling multiple correlated risk factors.")

