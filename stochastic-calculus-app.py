# =============================================================================
# Complete Streamlit Application for Stochastic Calculus Exploration
# =============================================================================

# --- Module Imports ---------------------------------------------------------
import streamlit as st  # Must import Streamlit before using its functions!
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import base64
from PIL import Image
from io import BytesIO
import time

# --- Page Configuration -----------------------------------------------------
st.set_page_config(
    page_title="Stochastic Calculus Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling -----------------------------------------------------
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

# =============================================================================
# Helper Functions: Generating Stochastic Processes
# =============================================================================

def generate_brownian_motion(T=1.0, N=1000, seed=None):
    """
    Generates a standard Brownian motion (Wiener process).

    Parameters:
    - T: Total time.
    - N: Number of time steps.
    - seed: Random seed for reproducibility.

    Returns:
    - t: Time vector.
    - W: Brownian motion sample path.
    """
    if seed is not None:
        np.random.seed(seed)
    dt = T / N
    dW = np.random.normal(0, np.sqrt(dt), N)
    W = np.cumsum(dW)
    W = np.insert(W, 0, 0)  # Start the process at 0
    t = np.linspace(0, T, N + 1)
    return t, W

def generate_gbm(S0, mu, sigma, T=1.0, N=1000, seed=None):
    """
    Generates a Geometric Brownian Motion (GBM) using its closed-form solution.
    
    The closed-form expression is:
      S(t) = S0 * exp((mu - sigma^2/2) * t + sigma * W(t))
    
    Parameters:
    - S0: Initial stock price.
    - mu: Drift term.
    - sigma: Volatility.
    - T: Time horizon.
    - N: Number of steps.
    - seed: Random seed.

    Returns:
    - t: Time vector.
    - S: GBM sample path.
    """
    t, W = generate_brownian_motion(T, N, seed)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S0 * np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * W[i])
    return t, S

def generate_mean_reverting(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    """
    Generates a mean-reverting process using Euler–Maruyama discretization.
    
    The SDE is of the form:
      dS = (nu - mu*S) dt + sigma dW
    
    Parameters:
    - S0: Initial value.
    - nu: Long-term mean parameter.
    - mu: Speed of reversion.
    - sigma: Volatility.
    - T: Time horizon.
    - N: Number of steps.
    - seed: Random seed.
    
    Returns:
    - t: Time vector.
    - S: Sample path.
    """
    t, W = generate_brownian_motion(T, N, seed)
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        S[i] = S[i - 1] + (nu - mu * S[i - 1]) * dt + sigma * (W[i] - W[i - 1])
    return t, S

def generate_cir(S0, nu, mu, sigma, T=1.0, N=1000, seed=None):
    """
    Generates a Cox-Ingersoll-Ross (CIR) process, which is used in interest rate modeling.
    
    The SDE is:
      dS = (nu - mu*S) dt + sigma * sqrt(S) dW
    
    A safeguard is used to ensure non-negativity (since sqrt is only defined for non-negative values).
    
    Parameters:
    - S0: Initial value.
    - nu: Long-term mean factor.
    - mu: Speed of reversion.
    - sigma: Volatility.
    - T: Time horizon.
    - N: Number of steps.
    - seed: Random seed.
    
    Returns:
    - t: Time vector.
    - S: Sample path (remains non-negative).
    """
    t, W = generate_brownian_motion(T, N, seed)
    dt = T / N
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(1, N + 1):
        # Enforce non-negativity: if the previous value is negative, reset it to 0.
        if S[i - 1] < 0:
            S[i - 1] = 0
        S[i] = S[i - 1] + (nu - mu * S[i - 1]) * dt + sigma * np.sqrt(max(0, S[i - 1])) * (W[i] - W[i - 1])
    return t, S

# =============================================================================
# Helper Functions: Plotting with Matplotlib
# =============================================================================

def plot_stochastic_process_matplotlib(t, S, title, ylabel):
    """
    Creates a line plot of a stochastic process.
    
    Parameters:
    - t: Time vector.
    - S: Process values.
    - title: Plot title.
    - ylabel: Label for the y-axis.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, S)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig

def plot_comparison(t_list, S_list, titles, main_title):
    """
    Plots a 2x2 grid to compare four different stochastic processes.
    
    Parameters:
    - t_list: List of time vectors for each process.
    - S_list: List of process paths.
    - titles: List of titles for each subplot.
    - main_title: Overall title for the figure.
    
    Returns:
    - fig: Matplotlib figure object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i in range(4):
        axes[i].plot(t_list[i], S_list[i])
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Value")
    fig.suptitle(main_title)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def get_cc_image():
    """
    Generates a placeholder image (as Base64 string) for Creative Commons licensing.
    
    Returns:
    - A data URL string for embedding an image.
    """
    buffer = BytesIO()
    img = Image.new('RGB', (120, 40), (255, 255, 255))
    img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

# =============================================================================
# App Title and Sidebar Navigation Setup
# =============================================================================

st.title("Elementary Stochastic Calculus Explorer")
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
st.write("By Luís Simões da Cunha")
st.sidebar.markdown("""
---
### About

This interactive application explores the foundations of stochastic calculus and its applications in financial mathematics.

#### Disclaimer
The content provided here is for educational purposes only. No guarantee is made regarding the correctness of the information or code examples.
""")
st.sidebar.markdown("""
---
<div class="cc-license">
<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">
<img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">CC BY-NC 4.0 License</a>.
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Section: Introduction
# =============================================================================

if section == "Introduction":
    st.header("Introduction to Stochastic Calculus")
    st.markdown("""
Stochastic calculus is a key mathematical framework for understanding and modeling random processes,
especially in financial mathematics. It provides the foundation for pricing derivatives, managing risk, 
and analyzing market dynamics.
    """)
    st.markdown("""
**Why is stochastic calculus important in finance?**  
Financial markets exhibit randomness that cannot be modeled by deterministic equations. Stochastic calculus 
provides the tools to model this randomness and its propagation through financial instruments.
    """)
    st.header("Randomness in Financial Markets")
    col1, col2 = st.columns(2)
    with col1:
        seed = np.random.randint(0, 10000)
        t, bm = generate_brownian_motion(seed=seed)
        fig_bm = plot_stochastic_process_matplotlib(t, bm, "Brownian Motion", "Value")
        st.pyplot(fig_bm)
    with col2:
        t, gbm = generate_gbm(100, 0.05, 0.2, seed=seed)
        fig_gbm = plot_stochastic_process_matplotlib(t, gbm, "Stock Price Model", "Price")
        st.pyplot(fig_gbm)
    st.markdown("The left plot shows pure randomness (Brownian motion) and the right shows a stock price model incorporating growth.")

# =============================================================================
# Section: Markov & Martingale Properties
# =============================================================================

elif section == "Markov & Martingale Properties":
    st.header("The Markov Property")
    st.markdown("For a stochastic process $S_i$, the Markov property states that:")
    st.latex(r"E[S_i \mid S_1, S_2, \dots, S_{i-1}] = E[S_i \mid S_{i-1}]")
    st.header("The Martingale Property")
    st.markdown("For a stochastic process $S_i$, the martingale property states that:")
    st.latex(r"E[S_i \mid S_j,\, j < i] = S_j")
    st.markdown("""
In finance, a martingale represents a "fair game" where current information cannot predict future gains or losses.
    """)
    st.header("Interactive Example: Coin Flipping")
    st.markdown("""
Consider a coin flipping experiment:
- Each "Head" gives you \$1  
- Each "Tail" costs you \$1  
- The cumulative winnings follow both the Markov and martingale properties.
    """)
    num_flips = st.slider("Number of coin flips", 5, 100, 20)
    if st.button("Flip coins"):
        flips = np.random.choice([-1, 1], size=num_flips)
        cumulative_winnings = np.cumsum(flips)
        cumulative_winnings = np.insert(cumulative_winnings, 0, 0)
        df = pd.DataFrame({"Flip": range(num_flips + 1), "Winnings": cumulative_winnings})
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["Flip"], df["Winnings"], marker="o")
        ax.set_title("Coin Flipping Experiment")
        ax.set_xlabel("Number of Flips")
        ax.set_ylabel("Cumulative Winnings")
        st.pyplot(fig)
        st.markdown("Observations: The expected winnings remain equal to the current amount, illustrating the martingale property.")

# =============================================================================
# Section: Brownian Motion
# =============================================================================

elif section == "Brownian Motion":
    st.header("Brownian Motion")
    st.markdown("""
Brownian motion (or Wiener process) is a continuous-time process with three key properties:
1. It starts at zero: 
    """)
    st.latex(r"W(0) = 0")
    st.markdown("""
2. It has independent increments.  
3. The increments follow a normal distribution with mean 0 and variance equal to the time difference.
    """)
    st.markdown("Furthermore, the paths of $W(t)$ are continuous but nowhere differentiable.")
    st.header("Interactive Brownian Motion Simulation")
    col1, col2 = st.columns([1, 3])
    with col1:
        T = st.slider("Time horizon", 0.1, 5.0, 1.0, 0.1)
        N = st.slider("Number of steps", 100, 5000, 1000, 100)
        num_paths = st.slider("Number of paths", 1, 10, 3)
        seed = st.number_input("Random seed (optional)", value=None, min_value=0, max_value=10000, step=1)
    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_paths):
            np_seed = seed + i if seed is not None else None
            t, W = generate_brownian_motion(T, N, np_seed)
            ax.plot(t, W, label=f"Path {i+1}")
        ax.set_title("Brownian Motion Paths")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if num_paths > 1:
            ax.legend()
        st.pyplot(fig)
    st.markdown("Key observations: The paths are continuous yet erratic, with variance increasing over time.")
    st.header("Quadratic Variation")
    st.latex(r"\sum_{j=1}^{n} \left(W(t_j) - W(t_{j-1})\right)^2 \to t \quad \text{as } n \to \infty")
    st.markdown("This property is fundamental in stochastic calculus, leading to the rule $dW^2 = dt$.")

# =============================================================================
# Section: Stochastic Integration
# =============================================================================

elif section == "Stochastic Integration":
    st.header("Stochastic Integration")
    st.markdown("""
Stochastic integration extends integration to include random processes. The Itô integral is defined as:
    """)
    st.latex(r"\int_0^t f(\tau) \, dW(\tau) = \lim_{n \to \infty} \sum_{j=1}^{n} f(t_{j-1}) \left(W(t_j)-W(t_{j-1})\right)")
    st.markdown("""
Key differences from classical integration include evaluation at the left endpoint (non-anticipatory property) and modified calculus rules.
    """)
    st.header("Visualizing Stochastic Integration")
    t, W = generate_brownian_motion(1.0, 1000, seed=42)
    f = np.sin(5 * t)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, f)
    axs[0].set_title("Function $f(t) = \sin(5t)$")
    axs[0].set_ylabel("f(t)")
    axs[1].plot(t, W)
    axs[1].set_title("Brownian Motion $W(t)$")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("W(t)")
    st.pyplot(fig)
    st.markdown("""
The stochastic integral $$\int_0^t f(\tau) \, dW(\tau)$$ 
can be interpreted as the accumulation of products of function values and Brownian increments.
    """)

# =============================================================================
# Section: Stochastic Differential Equations
# =============================================================================

elif section == "Stochastic Differential Equations":
    st.header("Stochastic Differential Equations")
    st.markdown("""
SDEs are differential equations that include one or more stochastic processes. A general SDE is:
    """)
    st.latex(r"dS = a(S,t) \, dt + b(S,t) \, dW")
    st.markdown("""
where:
- $S$ is the modeled variable (e.g. stock price)
- $t$ is time  
- $a(S,t)$ is the drift  
- $b(S,t)$ is the diffusion  
- $W$ is Brownian motion
    """)
    st.markdown("The integral form is:")
    st.latex(r"S(t) = S(0) + \int_0^t a(S(\tau),\tau) \, d\tau + \int_0^t b(S(\tau),\tau) \, dW(\tau)")
    sde_type = st.selectbox("Select SDE type", ["Geometric Brownian Motion", "Mean-Reverting (Vasicek)", "Cox-Ingersoll-Ross (CIR)"])
    seed = st.number_input("Random seed", value=42, min_value=1, max_value=10000, step=1)
    if sde_type == "Geometric Brownian Motion":
        st.latex(r"dS = \mu S \, dt + \sigma S \, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 1.0, 100.0, 50.0, 1.0)
            mu = st.slider("Drift (μ)", -0.5, 0.5, 0.05, 0.01)
        with col2:
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
            num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_paths):
            t, S = generate_gbm(S0, mu, sigma, seed=seed + i)
            ax.plot(t, S, label=f"Path {i+1}")
        ax.set_title("Geometric Brownian Motion")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if num_paths > 1:
            ax.legend()
        st.pyplot(fig)
        st.markdown("Closed-form solution:")
        st.latex(r"S(t) = S(0) \, e^{\left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t)}")
    elif sde_type == "Mean-Reverting (Vasicek)":
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma\, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean (ν)", 0.1, 5.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion speed (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_paths):
            t, S = generate_mean_reverting(S0, nu, mu_, sigma, seed=seed + i)
            ax.plot(t, S, label=f"Path {i+1}")
        ax.axhline(y=nu/mu_, color="red", linestyle="--", label="Long-term mean")
        ax.set_title("Vasicek Mean-Reverting Process")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if num_paths > 1:
            ax.legend()
        st.pyplot(fig)
    elif sde_type == "Cox-Ingersoll-Ross (CIR)":
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma \sqrt{S}\, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀)", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean factor (ν)", 0.1, 5.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion rate (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.01, 1.0, 0.2, 0.01)
        num_paths = st.slider("Number of paths", 1, 5, 3, 1)
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_paths):
            t, S = generate_cir(S0, nu, mu_, sigma, seed=seed + i)
            ax.plot(t, S, label=f"Path {i+1}")
        ax.axhline(y=nu/mu_, color="red", linestyle="--", label="Long-term mean")
        ax.set_title("Cox-Ingersoll-Ross Process")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if num_paths > 1:
            ax.legend()
        st.pyplot(fig)
    st.subheader("Comparing Different Stochastic Processes")
    if st.button("Generate Comparison"):
        seed = 42
        t_bm, W_bm = generate_brownian_motion(T=1.0, N=1000, seed=seed)
        S_bm = 1.0 + 0.1 * np.linspace(0, 1, 1001) + 0.2 * W_bm
        t_gbm, S_gbm = generate_gbm(S0=1.0, mu=0.1, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_vas, S_vas = generate_mean_reverting(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_cir, S_cir = generate_cir(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_list = [t_bm, t_gbm, t_vas, t_cir]
        S_list = [S_bm, S_gbm, S_vas, S_cir]
        titles = ["BM w/ Drift", "GBM", "Vasicek", "CIR"]
        fig_comp = plot_comparison(t_list, S_list, titles, "Comparison of Stochastic Processes")
        st.pyplot(fig_comp)
        st.markdown("Observations: Brownian motion with drift can be negative, GBM remains positive, and while both Vasicek and CIR revert to a long-term mean, only CIR stays non-negative.")

# =============================================================================
# Section: Common Stochastic Processes
# =============================================================================

elif section == "Common Stochastic Processes":
    st.header("Common Stochastic Processes in Finance")
    st.markdown("""
Financial markets are modeled using various stochastic processes,
each with unique properties that suit different assets or market conditions.
    """)
    process = st.selectbox(
        "Select a stochastic process to explore:",
        ["Brownian Motion with Drift", "Geometric Brownian Motion", "Mean-Reverting (Vasicek)", "Cox-Ingersoll-Ross (CIR)"]
    )
    if process == "Brownian Motion with Drift":
        st.subheader("Brownian Motion with Drift")
        st.latex(r"dS = \mu \, dt + \sigma \, dW")
        col1, col2 = st.columns(2)
        with col1:
            mu = st.slider("Drift (μ) for BM", -0.5, 0.5, 0.1, 0.05)
        with col2:
            sigma = st.slider("Volatility (σ) for BM", 0.01, 1.0, 0.3, 0.05)
        t, W = generate_brownian_motion(T=1.0, N=1000, seed=42)
        S = mu * t + sigma * W
        fig = plot_stochastic_process_matplotlib(t, S, "Brownian Motion with Drift", "Value")
        st.pyplot(fig)
        st.markdown("Key properties: Can be both positive and negative; linear trend; variance increases linearly with time.")
    elif process == "Geometric Brownian Motion":
        st.subheader("Geometric Brownian Motion (GBM)")
        st.latex(r"dS = \mu S \, dt + \sigma S \, dW")
        col1, col2, col3 = st.columns(3)
        with col1:
            S0 = st.slider("Initial value (S₀) for GBM", 10.0, 200.0, 100.0, 10.0)
        with col2:
            mu = st.slider("Drift (μ) for GBM", -0.5, 0.5, 0.05, 0.05)
        with col3:
            sigma = st.slider("Volatility (σ) for GBM", 0.05, 0.5, 0.2, 0.05)
        t, S = generate_gbm(S0, mu, sigma, T=1.0, N=1000, seed=42)
        fig = plot_stochastic_process_matplotlib(t, S, "Geometric Brownian Motion", "Value")
        st.pyplot(fig)
        fig_log, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t, np.log(S))
        ax.set_title("Log of Geometric Brownian Motion")
        ax.set_xlabel("Time")
        ax.set_ylabel("Log Value")
        st.pyplot(fig_log)
        st.markdown(r"""
**Key properties:**  
- Always positive  
- Log-returns are normally distributed  
- Closed-form solution: 
\[
S(t) = S(0)\,\exp\Bigl((\mu - \tfrac{\sigma^2}{2})\,t + \sigma\,W(t)\Bigr)
\]
        """)
    elif process == "Mean-Reverting (Vasicek)":
        st.subheader("Mean-Reverting (Vasicek) Process")
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma\, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀) for Vasicek", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean (ν)", 0.1, 5.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion speed (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.05, 0.5, 0.2, 0.05)
        t, S = generate_mean_reverting(S0, nu, mu_, sigma, T=1.0, N=1000, seed=42)
        fig = plot_stochastic_process_matplotlib(t, S, "Vasicek Process", "Value")
        st.pyplot(fig)
    elif process == "Cox-Ingersoll-Ross (CIR)":
        st.subheader("Cox-Ingersoll-Ross (CIR) Model")
        st.latex(r"dS = (\nu - \mu S)\, dt + \sigma \sqrt{S}\, dW")
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.slider("Initial value (S₀) for CIR", 0.01, 5.0, 1.0, 0.01)
            nu = st.slider("Long-term mean factor (ν)", 0.1, 5.0, 1.0, 0.1)
        with col2:
            mu_ = st.slider("Reversion speed (μ)", 0.1, 5.0, 1.0, 0.1)
            sigma = st.slider("Volatility (σ)", 0.05, 0.5, 0.2, 0.05)
        t, S = generate_cir(S0, nu, mu_, sigma, T=1.0, N=1000, seed=42)
        fig = plot_stochastic_process_matplotlib(t, S, "CIR Process", "Value")
        st.pyplot(fig)
    st.subheader("Comparing Different Stochastic Processes")
    if st.button("Generate Comparison"):
        seed = 42
        t_bm, W_bm = generate_brownian_motion(T=1.0, N=1000, seed=seed)
        S_bm = 1.0 + 0.1 * np.linspace(0, 1, 1001) + 0.2 * W_bm
        t_gbm, S_gbm = generate_gbm(S0=1.0, mu=0.1, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_vas, S_vas = generate_mean_reverting(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_cir, S_cir = generate_cir(S0=1.0, nu=0.1, mu=0.5, sigma=0.2, T=1.0, N=1000, seed=seed)
        t_list = [t_bm, t_gbm, t_vas, t_cir]
        S_list = [S_bm, S_gbm, S_vas, S_cir]
        titles = ["BM w/ Drift", "GBM", "Vasicek", "CIR"]
        fig_comp = plot_comparison(t_list, S_list, titles, "Comparison of Stochastic Processes")
        st.pyplot(fig_comp)
        st.markdown("Observations: GBM remains positive, whereas BM with drift can go negative. Vasicek and CIR revert to a long-term mean, but CIR stays non-negative.")

# =============================================================================
# Section: Interactive Simulation
# =============================================================================

elif section == "Interactive Simulation":
    st.header("Interactive Stochastic Process Simulator")
    st.markdown("""
This simulator lets you generate and visualize various stochastic processes with your chosen parameters.
Adjust parameters to see how the process behavior changes.
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
        fig, ax = plt.subplots(figsize=(10, 5))
        for i in range(num_paths):
            if process_type == "Geometric Brownian Motion (GBM)":
                t, S = generate_gbm(S0, mu, sigma, T, N, seed + i)
            elif process_type == "Mean-Reverting (Vasicek)":
                t, S = generate_mean_reverting(S0, nu, mu, sigma, T, N, seed + i)
            elif process_type == "Cox-Ingersoll-Ross (CIR)":
                t, S = generate_cir(S0, nu, mu, sigma, T, N, seed + i)
            ax.plot(t, S, label=f"Path {i+1}")
        if process_type != "Geometric Brownian Motion (GBM)":
            ax.axhline(y=nu/mu, color="red", linestyle="--", label="Long-term mean")
        ax.set_title(f"{process_type} Simulation")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")
        if num_paths > 1:
            ax.legend()
        st.pyplot(fig)
        
        # Calculate and display statistics across multiple paths
        if num_paths > 1:
            all_paths = np.zeros((num_paths, N + 1))
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
            fig_stats, ax_stats = plt.subplots(figsize=(10, 5))
            ax_stats.plot(t, mean_path, label="Mean", linewidth=2)
            ax_stats.fill_between(t, mean_path + std_path, mean_path - std_path, alpha=0.2, label="Mean ± 1 Std Dev")
            ax_stats.plot(t, min_path, label="Min", linestyle="--")
            ax_stats.plot(t, max_path, label="Max", linestyle="--")
            ax_stats.set_title("Statistics Across Paths")
            ax_stats.set_xlabel("Time")
            ax_stats.set_ylabel("Value")
            ax_stats.legend()
            st.pyplot(fig_stats)
            final_values = all_paths[:, -1]
            fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
            ax_hist.hist(final_values, bins=20, edgecolor="black")
            ax_hist.set_title("Distribution of Final Values")
            ax_hist.set_xlabel("Final Value")
            ax_hist.set_ylabel("Frequency")
            st.pyplot(fig_hist)
            st.subheader("Summary Statistics of Final Values")
            stats_df = pd.DataFrame({
                "Statistic": ["Mean", "Median", "Standard Deviation", "Minimum", "Maximum"],
                "Value": [
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
- Variability between paths shows uncertainty in outcomes.
- Aggregated statistics provide insight into expected behavior and risk.
- Parameter choices significantly affect process behavior.
    """)
    st.markdown("""
**Financial Applications:**
- Monte Carlo methods for option pricing  
- Risk management (e.g., Value-at-Risk)  
- Portfolio optimization  
- Stress testing models
    """)

# =============================================================================
# Section: Itô's Lemma
# =============================================================================

elif section == "Itô's Lemma":
    st.header("Itô's Lemma")
    st.markdown("""
Itô's lemma is the cornerstone of stochastic calculus. It is the stochastic version of the chain rule,
telling us how to differentiate functions of stochastic processes.
    """)
    st.markdown("For a function $F(S,t)$ where $S$ follows the SDE $dS = a(S,t)\, dt + b(S,t)\, dW$, Itô's lemma states:")
    st.latex(r"""
dF = \left(\frac{\partial F}{\partial t} + a\frac{\partial F}{\partial S} + \frac{1}{2}b^2\frac{\partial^2 F}{\partial S^2}\right)dt + b\frac{\partial F}{\partial S} \, dW
    """)
    st.markdown("""
The key difference from ordinary calculus is the extra second derivative term, due to the quadratic variation of Brownian motion ($dW^2 = dt$).
    """)
    st.header("Intuition Behind Itô's Lemma")
    st.markdown("""
A simple rule of thumb:
1. Taylor expand the function.
2. Keep terms up to second order in $dS$.
3. Replace $dW^2$ with $dt$.
    """)
    st.header("Example: Applying Itô's Lemma")
    st.markdown("Consider a stock price following geometric Brownian motion:")
    st.latex(r"dS = \mu S \, dt + \sigma S \, dW")
    st.markdown("Now, to find the SDE for $F(S)=\log(S)$, compute the derivatives:")
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
    st.header("Visual Demonstration")
    S0 = 100.0
    mu = 0.05
    sigma = 0.2
    t, S = generate_gbm(S0, mu, sigma, seed=42)
    log_S = np.log(S)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t, S)
    axs[0].set_title("Stock Price $S(t)$")
    axs[0].set_ylabel("S(t)")
    axs[1].plot(t, log_S)
    axs[1].set_title("Log Stock Price $\\log(S(t))$")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("log(S(t))")
    st.pyplot(fig)
    st.markdown("The logarithmic transformation makes the process appear more linear with noise, as predicted by Itô's lemma.")
    st.header("Itô's Lemma in Multiple Dimensions")
    st.markdown("""
Itô's lemma extends to functions of multiple stochastic variables. For a function $F(S_1,S_2,\dots,t)$, if each
$S_i$ satisfies 
$$dS_i = a_i\, dt + b_i\, dW_i,$$ 
with $dW_i\, dW_j = \\rho_{ij}\, dt$, then:
    """)
    st.latex(r"""
dF = \frac{\partial F}{\partial t}dt + \sum_i \frac{\partial F}{\partial S_i}dS_i + \frac{1}{2}\sum_i\sum_j \rho_{ij}b_i b_j\frac{\partial^2 F}{\partial S_i \partial S_j}dt.
    """)
    st.markdown("This multidimensional version is vital for pricing options on baskets or modeling multiple correlated risks.")
