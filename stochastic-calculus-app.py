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
    .katex {
        font-size: 1.1em;
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
    .formula-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4e8cff;
    }
    .note-box {
        background-color: #e7f0fd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4e8cff;
    }
    .cc-license {
        text-align: center;
        margin: 20px 0;
    }
    .st-emotion-cache-16txtl3 h1, .st-emotion-cache-16txtl3 h2, .st-emotion-cache-16txtl3 h3 {
        padding-top: 20px;
        padding-bottom: 10px;
    }
    @media (max-width: 768px) {
        .katex {
            font-size: 1em;
        }
    }
</style>
""", unsafe_allow_html=True)

# Function to generate Brownian motion
def generate_brownian_motion(T=1.0, N=1000, seed=None):
    if seed is not None:
        np.random.seed(seed)
    dt = T/N
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # Start at 0
    t = np.linspace(0, T, N+1)
    return t, W

# Function to generate geometric Brownian motion
def generate_gbm(S0, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    dt = T/N
    t = np.linspace(0, T, N+1)
    S = np.zeros(N+1)
    S[0] = S0
    
    # Using the exact solution for GBM
    for i in range(1, N+1):
        S[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * W[i])
    
    return t, S

# Function to generate mean-reverting process (Ornstein-Uhlenbeck)
def generate_mean_reverting(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    dt = T/N
    S = np.zeros(N+1)
    S[0] = S0
    
    for i in range(1, N+1):
        S[i] = S[i-1] + (nu - mu * S[i-1]) * dt + sigma * (W[i] - W[i-1])
    
    return t, S

# Function to generate CIR process
def generate_cir(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    t, W = generate_brownian_motion(T, N, seed)
    dt = T/N
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
    # Create a simple CC-BY-NC badge
    cc_text = "CC BY-NC"
    # Create a bytes buffer for the image
    buffer = BytesIO()
    # Create a simple white background image
    img = Image.new('RGB', (120, 40), (255, 255, 255))
    # Save the image to the buffer
    img.save(buffer, format='PNG')
    # Get the base64 encoded image
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

# App title and author info
st.title("Elementary Stochastic Calculus Explorer")
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

# Add CC-BY-NC license at the bottom
st.sidebar.markdown("""
---
<div class="cc-license">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
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

# Introduction
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
    <div class="note-box">
    <b>Why is stochastic calculus important in finance?</b><br>
    Financial markets exhibit random behavior that cannot be adequately modeled by deterministic equations.
    Stochastic calculus provides the tools to model this randomness and understand how it propagates
    through various financial instruments.
    </div>
    """, unsafe_allow_html=True)
    
    st.header("What You'll Learn")
    
    st.markdown("""
    * The meaning of **Markov** and **martingale** processes
    * **Brownian motion** and its properties
    * **Stochastic integration** and how it differs from regular calculus
    * **Stochastic differential equations** and their interpretation
    * **Itô's lemma** - the cornerstone of stochastic calculus
    * Common stochastic processes used in financial modeling
    """)
    
    # Display random paths to illustrate stochasticity
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
    
    st.markdown("""
    These plots illustrate the random nature of financial processes. The left plot shows pure randomness (Brownian motion),
    while the right shows a common model for stock prices that incorporates both random fluctuations and a growth trend.
    """)

# Markov & Martingale Properties
elif section == "Markov & Martingale Properties":
    st.header("The Markov Property")
    
    st.markdown("""
    The Markov property is a fundamental concept in stochastic processes. A process has the Markov property if 
    the conditional probability distribution of future states depends only on the present state, not on the sequence 
    of states that preceded it.
    
    In simple terms, a Markov process has no memory beyond its current state. The future behavior of the process
    is entirely determined by where it is now, not how it got there.
    """)
    
    st.markdown("""
    <div class="formula-box">
    For a stochastic process $S_i$, the Markov property states that:<br>
    $E[S_i|S_1, S_2, ..., S_{i-1}] = E[S_i|S_{i-1}]$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This property is incredibly important in financial modeling. Most financial models we use have the 
    Markov property, which greatly simplifies their analysis.
    """)
    
    st.header("The Martingale Property")
    
    st.markdown("""
    A martingale is a stochastic process where the expected value of the next observation, given all past 
    observations, is equal to the most recent observation.
    """)
    
    st.markdown("""
    <div class="formula-box">
    For a stochastic process $S_i$, the martingale property states that:<br>
    $E[S_i|S_j, j < i] = S_j$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    In financial terms, a martingale represents a "fair game" where no information available at the present time 
    can be used to predict future gains or losses. The martingale property is closely related to 
    efficient market hypotheses in finance.
    """)
    
    # Interactive example
    st.header("Interactive Example: Coin Flipping")
    
    st.markdown("""
    Consider a coin flipping experiment:
    - Each "Head" gives you $1
    - Each "Tail" costs you $1
    - Your total winnings have both the Markov and martingale properties
    """)
    
    num_flips = st.slider("Number of coin flips", 5, 100, 20)
    if st.button("Flip coins"):
        flips = np.random.choice([-1, 1], size=num_flips)  # -1 for tails, 1 for heads
        cumulative_winnings = np.cumsum(flips)
        cumulative_winnings = np.insert(cumulative_winnings, 0, 0)  # Start at 0
        
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

# Brownian Motion
elif section == "Brownian Motion":
    st.header("Brownian Motion")
    
    st.markdown("""
    Brownian motion (also called a Wiener process) is one of the most fundamental stochastic processes. 
    It is a continuous-time stochastic process with three defining properties:
    
    1. It starts at zero: $W(0) = 0$
    2. It has independent increments
    3. The increments follow a normal distribution with mean 0 and variance equal to the time difference
    
    Brownian motion is the limit of a random walk as the time steps become infinitesimally small, and it 
    serves as the building block for more complex stochastic models.
    """)
    
    st.markdown("""
    <div class="formula-box">
    The key properties of Brownian motion $W(t)$ are:
    <ul>
        <li>$W(0) = 0$</li>
        <li>$W(t) - W(s) \sim N(0, t - s)$ for $0 \leq s < t$</li>
        <li>For non-overlapping time intervals, the increments are independent</li>
        <li>The paths of $W(t)$ are continuous but nowhere differentiable</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Interactive Brownian motion simulation
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
            if seed is not None:
                np_seed = seed + i
            else:
                np_seed = None
                
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
    
    # Quadratic variation
    st.header("Quadratic Variation")
    
    st.markdown("""
    An important property of Brownian motion is its quadratic variation. For a partition of the interval [0,t] into n subintervals,
    the sum of squared increments converges to t as the partition gets finer:
    """)
    
    st.markdown("""
    <div class="formula-box">
    $\sum_{j=1}^{n} (W(t_j) - W(t_{j-1}))^2 \to t$ as $n \to \infty$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This property is crucial for developing stochastic calculus, as it leads to the rule $dW^2 = dt$, which is 
    fundamental to Itô's lemma.
    """)

# Stochastic Integration
elif section == "Stochastic Integration":
    st.header("Stochastic Integration")
    
    st.markdown("""
    Stochastic integration extends the concept of integration to include random processes. The most common 
    form is the Itô integral, which integrates a deterministic function against a Brownian motion.
    """)
    
    st.markdown("""
    <div class="formula-box">
    The Itô integral is defined as:<br>
    $\int_0^t f(\tau) dW(\tau) = \lim_{n \to \infty} \sum_{j=1}^{n} f(t_{j-1})(W(t_j) - W(t_{j-1}))$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    There are several key differences between stochastic integration and ordinary calculus:
    
    1. The integrand is evaluated at the left endpoint of each subinterval (non-anticipatory property)
    2. The integral has different properties from Riemann integration
    3. Standard calculus rules (like the chain rule) need to be modified
    """)
    
    st.header("Non-Anticipatory Property")
    
    st.markdown("""
    A crucial aspect of stochastic integration is that it is non-anticipatory. This means that when evaluating 
    the integrand at each point, we don't use any information about future values of the Brownian motion.
    
    In financial terms, this ensures that our actions today (such as choosing a portfolio) don't depend on 
    future price movements that we couldn't possibly know.
    """)
    
    # Visual example
    st.header("Visualizing Stochastic Integration")
    
    t, W = generate_brownian_motion(1.0, 1000, seed=42)
    
    # Create a function to integrate
    f = np.sin(5*t)
    
    fig = make_subplots(rows=2, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=('Function f(t) = sin(5t)', 'Brownian Motion W(t)'))
    
    fig.add_trace(go.Scatter(x=t, y=f, mode='lines', name='f(t)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=W, mode='lines', name='W(t)'), row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    The stochastic integral $\int_0^t f(\tau) dW(\tau)$ can be thought of as accumulating the product of the 
    function value and the increments of Brownian motion. The crucial point is that we evaluate the function at 
    the beginning of each time interval, before we know the Brownian increment.
    """)
    
    st.markdown("""
    <div class="note-box">
    <b>Financial interpretation:</b><br>
    If $W(t)$ represents the randomness in a stock price, and $f(t)$ represents our investment strategy,
    then $\int_0^t f(\tau) dW(\tau)$ represents our cumulative gains or losses from this strategy.
    The non-anticipatory property ensures that our strategy doesn't peek into the future.
    </div>
    """, unsafe_allow_html=True)

# Stochastic Differential Equations
elif section == "Stochastic Differential Equations":
    st.header("Stochastic Differential Equations")
    
    st.markdown("""
    Stochastic Differential Equations (SDEs) are differential equations where one or more terms is a stochastic process.
    They are the mathematical framework for modeling systems with random components.
    """)
    
    st.markdown("""
    <div class="formula-box">
    A general form of an SDE is:<br>
    $dS = a(S,t)dt + b(S,t)dW$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    where:
    - $S$ is the variable being modeled (e.g., stock price)
    - $t$ is time
    - $a(S,t)$ is the "drift" function (deterministic component)
    - $b(S,t)$ is the "diffusion" function (random component)
    - $W$ is a Brownian motion
    
    The drift term represents the expected change in the variable, while the diffusion term represents the random fluctuations.
    """)
    
    st.header("Interpreting Stochastic Differential Equations")
    
    st.markdown("""
    SDEs are often written in differential form, but their precise meaning comes from the equivalent integral form:
    
    <div class="formula-box">
    $S(t) = S(0) + \int_0^t a(S(\tau),\tau)d\tau + \int_0^t b(S(\tau),\tau)dW(\tau)$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This integral equation shows that $S(t)$ consists of:
    1. The initial value $S(0)$
    2. A deterministic integral representing the cumulative drift
    3. A stochastic integral representing the cumulative random effects
    """)
    
    # Visual comparison of different SDEs
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
        st.markdown("""
        <div class="formula-box">
        Geometric Brownian Motion (GBM):<br>
        $dS = \mu S dt + \sigma S dW$
        </div>
        """, unsafe_allow_html=True)
        
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
        
        st.markdown("""
        **Properties of Geometric Brownian Motion:**
        
        - Used to model stock prices and other financial assets
        - Cannot become negative (suitable for prices)
        - The percentage changes follow a normal distribution
        - Has a closed-form solution: $S(t) = S(0)e^{(\mu - \frac{\sigma^2}{2})t + \sigma W(t)}$
        """)
        
    elif sde_type == "Mean-Reverting (Vasicek)":
        st.markdown("""
        <div class="formula-box">
        Vasicek Model (Mean-Reverting):<br>
        $dS = (\nu - \mu S) dt + \sigma dW$
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Add horizontal line for the long-term mean
        fig.add_hline(y=nu/mu, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        
        fig.update_layout(
            title='Vasicek Mean-Reverting Process',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Properties of the Vasicek Model:**
        
        - Used to model interest rates and other mean-reverting processes
        - Tends to revert to a long-term mean value (ν/μ)
        - Can become negative (a limitation for interest rates)
        - The rate of reversion is proportional to the distance from the mean
        """)
        
    elif sde_type == "Cox-Ingersoll-Ross (CIR)":
        st.markdown("""
        <div class="formula-box">
        Cox-Ingersoll-Ross (CIR) Model:<br>
        $dS = (\nu - \mu S) dt + \sigma \sqrt{S} dW$
        </div>
        """, unsafe_allow_html=True)
        
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
        
        # Add horizontal line for the long-term mean
        fig.add_hline(y=nu/mu, line_dash="dash", line_color="red", annotation_text="Long-term mean")
        
        fig.update_layout(
            title='Cox-Ingersoll-Ross Process',
            xaxis_title='Time',
            yaxis_title='Value',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Properties of the CIR Model:**
        
        - Used for interest rates and other positive mean-reverting processes
        - Cannot become negative (volatility decreases as the process approaches zero)
        - If 2νσ² ≥ 1, the process never reaches zero
        - Combines mean reversion with state-dependent volatility
        """)

# Itô's Lemma
elif section == "Itô's Lemma":
    st.header("Itô's Lemma")
    
    st.markdown("""
    Itô's lemma is the cornerstone of stochastic calculus. It's the stochastic version of the chain rule from ordinary calculus,
    telling us how to differentiate functions of stochastic processes.
    """)
    
    st.markdown("""
    <div class="formula-box">
    For a function $F(S,t)$ where $S$ follows the SDE $dS = a(S,t)dt + b(S,t)dW$, Itô's lemma states:<br>
    $dF = \left(\frac{\partial F}{\partial t} + a\frac{\partial F}{\partial S} + \frac{1}{2}b^2\frac{\partial^2 F}{\partial S^2}\right)dt + b\frac{\partial F}{\partial S}dW$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    The key difference from ordinary calculus is the second derivative term $\frac{1}{2}b^2\frac{\partial^2 F}{\partial S^2}dt$. 
    This extra term appears because of the quadratic variation property of Brownian motion, which gives us $dW^2 = dt$.
    """)
    
    st.header("Intuition Behind Itô's Lemma")
    
    st.markdown("""
    While Itô's lemma may look complicated, there's a simple "rule of thumb" that can help build intuition:
    
    1. Perform a Taylor series expansion of the function
    2. Keep terms up to second order in $dS$
    3. Replace $dW^2$ with $dt$
    
    This works because the quadratic variation of Brownian motion gives us $dW^2 = dt$ in the limit.
    """)
    
    st.markdown("""
    <div class="note-box">
    <b>Why is this important?</b><br>
    Itô's lemma allows us to derive SDEs for functions of stochastic processes. For instance, if we know 
    the SDE for a stock price $S$, we can derive the SDE for an option whose price depends on $S$. This 
    is the foundation of option pricing theories like Black-Scholes.
    </div>
    """, unsafe_allow_html=True)
    
    # Example of using Itô's lemma
    st.header("Example: Applying Itô's Lemma")
    
    st.markdown("""
    Consider a stock price following geometric Brownian motion:
    
    <div class="formula-box">
    $dS = \mu S dt + \sigma S dW$
    </div>
    
    Let's apply Itô's lemma to find the SDE for $F(S) = \log(S)$.
    
    <div class="formula-box">
    $\frac{\partial F}{\partial S} = \frac{1}{S}$<br>
    $\frac{\partial^2 F}{\partial S^2} = -\frac{1}{S^2}$<br>
    $\frac{\partial F}{\partial t} = 0$
    </div>
    
    Applying Itô's lemma:
    
    <div class="formula-box">
    $d(\log S) = \left(0 + \mu S \cdot \frac{1}{S} + \frac{1}{2}\sigma^2 S^2 \cdot (-\frac{1}{S^2})\right)dt + \sigma S \cdot \frac{1}{S}dW$<br>
    $d(\log S) = \left(\mu - \frac{1}{2}\sigma^2\right)dt + \sigma dW$
    </div>
    
    This shows that while $S$ follows a geometric Brownian motion, $\log(S)$ follows a Brownian motion with drift.
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="note-box">
    <b>Important implication:</b><br>
    This result shows that log-returns follow a normal distribution, which is a common assumption in financial models.
    </div>
    """, unsafe_allow_html=True)
    
    # Visual demonstration of Itô's lemma
    st.header("Visual Demonstration")
    
    st.markdown("""
    Let's visualize how Itô's lemma works by generating a stock price path and its logarithm:
    """)
    
    S0 = 100.0  # Initial stock price
    mu = 0.05   # Drift
    sigma = 0.2  # Volatility
    
    t, S = generate_gbm(S0, mu, sigma, seed=42)
    log_S = np.log(S)
    
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       subplot_titles=('Stock Price S(t)', 'Log Stock Price log(S(t))'))
    
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='S(t)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=log_S, mode='lines', name='log(S(t))'), row=2, col=1)
    
    fig.update_layout(height=600, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    Notice how the logarithm transformation makes the process look more like a straight line with noise,
    which is consistent with our derived SDE for $\log(S)$.
    """)
    
    # Multi-dimensional Itô's lemma
    st.header("Itô's Lemma in Multiple Dimensions")
    
    st.markdown("""
    Itô's lemma extends to functions of multiple stochastic variables. For a function $F(S_1, S_2, ..., t)$ 
    of multiple stochastic processes, we need to account for their correlations.
    
    <div class="formula-box">
    If $dS_i = a_i dt + b_i dW_i$ and $dW_i dW_j = \rho_{ij} dt$, then:<br>
    $dF = \frac{\partial F}{\partial t}dt + \sum_i \frac{\partial F}{\partial S_i}dS_i + \frac{1}{2}\sum_i\sum_j \rho_{ij}b_i b_j\frac{\partial^2 F}{\partial S_i \partial S_j}dt$
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    This multi-dimensional version is crucial for pricing options on multiple assets (like basket options) or
    when modeling multiple correlated risk factors.
    """)
    