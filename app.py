import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import datetime

# ==============================================================================
# 0. CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Opções: Simulador Quantitativo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. FUNÇÕES MATEMÁTICAS - BLACK-SCHOLES E GREGAS
# ==============================================================================
def d1(S, K, T, r, sigma):
    # Tratamento para T muito pequeno para evitar divisão por zero
    T = max(T, 1e-5)
    return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def d2(S, K, T, r, sigma):
    T = max(T, 1e-5)
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

def bs_price(S, K, T, r, sigma, option_type='call'):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
        return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    return 0.0

def bs_greeks(S, K, T, r, sigma, option_type='call'):
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    T_max = max(T, 1e-5)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d_1)
    else:
        delta = norm.cdf(d_1) - 1
        
    # Gamma (igual para call e put)
    gamma = norm.pdf(d_1) / (S * sigma * np.sqrt(T_max))
    
    # Vega (igual para call e put, dividido por 100 para impacto de 1%)
    vega = (S * norm.pdf(d_1) * np.sqrt(T_max)) / 100
    
    # Theta (dividido por 252 para theta diário)
    term1 = -(S * norm.pdf(d_1) * sigma) / (2 * np.sqrt(T_max))
    if option_type == 'call':
        theta = (term1 - r * K * np.exp(-r * T) * norm.cdf(d_2)) / 252
    else:
        theta = (term1 + r * K * np.exp(-r * T) * norm.cdf(-d_2)) / 252
        
    return delta, gamma, theta, vega

# ==============================================================================
# 2. SIMULAÇÃO MONTE CARLO (GBM)
# ==============================================================================
def simulate_gbm(S0, r, sigma, days=21, shocks=False):
    dt = 1 / 252
    prices = np.zeros(days + 1)
    prices[0] = S0
    
    for t in range(1, days + 1):
        # Adiciona choques aleatórios de volatilidade se ativado
        current_sigma = sigma * np.random.uniform(0.8, 1.5) if shocks else sigma
        shock = np.random.normal(0, 1)
        prices[t] = prices[t-1] * np.exp((r - 0.5 * current_sigma**2) * dt + current_sigma * np.sqrt(dt) * shock)
        
    return prices

# ==============================================================================
# 3. LÓGICA DAS ESTRATÉGIAS
# ==============================================================================
def build_strategy(S, strategy_name, otm_pct=0.10, condor_width=0.05):
    legs = []
    if strategy_name == "Pozinho (Long OTM Strangle)":
        # Compra Call OTM, Compra Put OTM
        legs.append({'type': 'call', 'strike': S * (1 + otm_pct), 'qty': 1})
        legs.append({'type': 'put', 'strike': S * (1 - otm_pct), 'qty': 1})
        
    elif strategy_name == "Iron Condor":
        # Vende Put, Compra Put (Asa), Vende Call, Compra Call (Asa)
        legs.append({'type': 'put', 'strike': S * (1 - condor_width), 'qty': -1})
        legs.append({'type': 'put', 'strike': S * (1 - condor_width * 2), 'qty': 1})
        legs.append({'type': 'call', 'strike': S * (1 + condor_width), 'qty': -1})
        legs.append({'type': 'call', 'strike': S * (1 + condor_width * 2), 'qty': 1})
        
    elif strategy_name == "Trava de Alta com Call":
        # Compra Call ATM, Vende Call OTM
        legs.append({'type': 'call', 'strike': S, 'qty': 1})
        legs.append({'type': 'call', 'strike': S * (1 + condor_width), 'qty': -1})
        
    return legs

def calculate_payoff(ST, legs):
    total_payoff = 0
    for leg in legs:
        if leg['type'] == 'call':
            payoff = max(ST - leg['strike'], 0) * leg['qty']
        else:
            payoff = max(leg['strike'] - ST, 0) * leg['qty']
        total_payoff += payoff
    return total_payoff

def calc_strategy_greeks_and_cost(S, legs, T, r, sigma):
    total_cost = 0
    net_delta = 0
    net_gamma = 0
    net_theta = 0
    net_vega = 0
    
    for leg in legs:
        price = bs_price(S, leg['strike'], T, r, sigma, leg['type'])
        total_cost += price * leg['qty']
        
        delta, gamma, theta, vega = bs_greeks(S, leg['strike'], T, r, sigma, leg['type'])
        net_delta += delta * leg['qty']
        net_gamma += gamma * leg['qty']
        net_theta += theta * leg['qty']
        net_vega += vega * leg['qty']
        
    return total_cost, net_delta, net_gamma, net_theta, net_vega

# ==============================================================================
# 4. INTERFACE DO USUÁRIO (SIDEBAR)
# ==============================================================================
with st.sidebar:
    st.header("⚙️ Parâmetros do Modelo")
    
    S0 = st.number_input("Preço do Ativo (R$)", value=30.0, step=1.0)
    sigma_pct = st.slider("Volatilidade Implícita (%)", min_value=10.0, max_value=150.0, value=35.0, step=1.0)
    r_pct = st.slider("Taxa Selic/Livre de Risco (%)", min_value=1.0, max_value=20.0, value=10.5, step=0.1)
    
    st.markdown("---")
    st.header("🛠️ Configuração da Operação")
    strategy_choice = st.selectbox("Estratégia de Opções", ["Pozinho (Long OTM Strangle)", "Iron Condor", "Trava de Alta com Call"])
    
    if strategy_choice == "Pozinho (Long OTM Strangle)":
        dist_pct = st.slider("Distância OTM (%)", 1.0, 30.0, 15.0) / 100
        condor_width = 0.05
    elif strategy_choice == "Iron Condor":
        condor_width = st.slider("Largura das Asas (%)", 1.0, 15.0, 5.0) / 100
        dist_pct = 0.10
    else:
        condor_width = st.slider("Distância da Trava (%)", 1.0, 15.0, 5.0) / 100
        dist_pct = 0.10
        
    months_to_sim = st.number_input("Período (Meses para Backtest)", min_value=1, max_value=60, value=12)
    contracts = st.number_input("Tamanho da Mão (Qtd)", min_value=100, max_value=10000, value=1000, step=100)
    
    sigma = sigma_pct / 100
    r = r_pct / 100
    T_month = 21 / 252 # 1 mês util

# ==============================================================================
# 5. ESTRUTURA DAS ABAS
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Simulador & Backtest", 
    "📊 Painel de Gregas", 
    "🔍 Screener Quantitativo"
])

# ==============================================================================
# ABA 1: SIMULADOR E BACKTEST
# ==============================================================================
with tab1:
    st.title("Simulador & Backtest de Estratégias")
    
    # Variáveis para armazenar histórico
    history = []
    capital_curve = [0]
    total_pnl = 0
    all_prices = [S0]
    
    # Simulação Mês a Mês
    np.random.seed(42) # Seed para reprodutibilidade
    current_S = S0
    
    for m in range(1, int(months_to_sim) + 1):
        legs = build_strategy(current_S, strategy_choice, dist_pct, condor_width)
        cost, _, _, _, _ = calc_strategy_greeks_and_cost(current_S, legs, T_month, r, sigma)
        
        # Simula a trajetória do mês (21 dias)
        month_prices = simulate_gbm(current_S, r, sigma, days=21, shocks=(strategy_choice=="Pozinho (Long OTM Strangle)"))
        ST = month_prices[-1]
        all_prices.extend(month_prices[1:])
        
        # Payoff no vencimento
        gross_payoff = calculate_payoff(ST, legs)
        net_pnl = (gross_payoff - cost) * contracts
        total_pnl += net_pnl
        capital_curve.append(total_pnl)
        
        history.append({
            'Mês': m,
            'Preço Inicial': round(current_S, 2),
            'Custo Estrutura (R$)': round(cost * contracts, 2),
            'Preço Vencimento': round(ST, 2),
            'Retorno Bruto (R$)': round(gross_payoff * contracts, 2),
            'PnL Líquido (R$)': round(net_pnl, 2),
            'Acumulado (R$)': round(total_pnl, 2)
        })
        
        current_S = ST
        
    df_history = pd.DataFrame(history)
    
    # Métricas de Desempenho
    win_trades = len(df_history[df_history['PnL Líquido (R$)'] > 0])
    win_rate = (win_trades / months_to_sim) * 100
    
    gross_profits = df_history[df_history['PnL Líquido (R$)'] > 0]['PnL Líquido (R$)'].sum()
    gross_losses = abs(df_history[df_history['PnL Líquido (R$)'] < 0]['PnL Líquido (R$)'].sum())
    profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
    
    peak = 0
    max_dd = 0
    for val in capital_curve:
        if val > peak: peak = val
        dd = peak - val
        if dd > max_dd: max_dd = dd

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resultado Acumulado", f"R$ {total_pnl:,.2f}", delta_color="normal" if total_pnl >=0 else "inverse")
    col2.metric("Win Rate", f"{win_rate:.1f} %")
    col3.metric("Fator de Lucro", f"{profit_factor:.2f}")
    col4.metric("Drawdown Máximo", f"R$ -{max_dd:,.2f}")
    
    st.markdown("---")
    
    # Gráficos
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Trajetória do Ativo-Objeto")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(all_prices, color='blue', alpha=0.7)
        ax1.set_title("Movimento Browniano Geométrico")
        ax1.set_xlabel("Dias Úteis")
        ax1.set_ylabel("Preço (R$)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)
        
    with col_chart2:
        st.subheader("Curva de Capital (PnL Acumulado)")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        color = 'green' if total_pnl >= 0 else 'red'
        ax2.fill_between(range(len(capital_curve)), capital_curve, 0, color=color, alpha=0.3)
        ax2.plot(capital_curve, color=color, linewidth=2)
        ax2.set_title("Evolução do Patrimônio")
        ax2.set_xlabel("Meses")
        ax2.set_ylabel("PnL (R$)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig2)

    st.subheader("Extrato Mensal de Operações")
    st.dataframe(df_history.style.map(lambda x: 'color: green' if x > 0 else 'color: red' if x < 0 else '', subset=['PnL Líquido (R$)']), use_container_width=True)

# ==============================================================================
# ABA 2: PAINEL DE GREGAS LÍQUIDAS
# ==============================================================================
with tab2:
    st.title("Painel de Gerenciamento de Gregas")
    st.markdown("Análise da exposição estrutural da carteira no momento da montagem (T=0) para a estratégia selecionada.")
    
    # Calcula as gregas iniciais para a estratégia escolhida com os parâmetros atuais
    base_legs = build_strategy(S0, strategy_choice, dist_pct, condor_width)
    _, net_delta, net_gamma, net_theta, net_vega = calc_strategy_greeks_and_cost(S0, base_legs, T_month, r, sigma)
    
    # Multiplica pelo tamanho da mão
    pos_delta = net_delta * contracts
    pos_gamma = net_gamma * contracts
    pos_theta = net_theta * contracts
    pos_vega = net_vega * contracts
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Δ Delta Líquido", f"{pos_delta:,.2f}", help="Exposição direcional equivalente em ações")
    c2.metric("Γ Gamma Líquido", f"{pos_gamma:,.4f}", help="Taxa de variação do Delta")
    c3.metric("Θ Theta (R$/Dia)", f"R$ {pos_theta:,.2f}", help="Decaimento temporal financeiro por dia")
    c4.metric("V Vega (1% IV)", f"R$ {pos_vega:,.2f}", help="Exposição a cada 1% de aumento na Volatilidade Implícita")
    
    st.markdown("---")
    
    # Simulação da evolução das gregas ao longo de 21 dias sem mexer no preço (apenas passagem do tempo)
    days_to_expiry = np.linspace(21, 1, 21)
    greeks_evo = {'Delta': [], 'Gamma': [], 'Theta': [], 'Vega': []}
    
    for d in days_to_expiry:
        t_rem = d / 252
        _, d_val, g_val, t_val, v_val = calc_strategy_greeks_and_cost(S0, base_legs, t_rem, r, sigma)
        greeks_evo['Delta'].append(d_val * contracts)
        greeks_evo['Gamma'].append(g_val * contracts)
        greeks_evo['Theta'].append(t_val * contracts)
        greeks_evo['Vega'].append(v_val * contracts)
        
    fig3, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    axes[0, 0].bar(21 - days_to_expiry, greeks_evo['Delta'], color='blue', alpha=0.7)
    axes[0, 0].set_title("Evolução do Delta (T mantendo S0)")
    
    axes[0, 1].bar(21 - days_to_expiry, greeks_evo['Gamma'], color='purple', alpha=0.7)
    axes[0, 1].set_title("Evolução do Gamma")
    
    axes[1, 0].bar(21 - days_to_expiry, greeks_evo['Theta'], color='orange', alpha=0.7)
    axes[1, 0].set_title("Evolução do Theta (R$/Dia)")
    
    axes[1, 1].bar(21 - days_to_expiry, greeks_evo['Vega'], color='cyan', alpha=0.7)
    axes[1, 1].set_title("Evolução do Vega")
    
    for ax in axes.flat:
        ax.set_xlabel("Dias Decorridos")
        ax.grid(True, linestyle='--', alpha=0.4)
        
    plt.tight_layout()
    st.pyplot(fig3)

# ==============================================================================
# ABA 3: SCREENER QUANTITATIVO DE ATIVOS
# ==============================================================================
with tab3:
    st.title("Filtro Quantitativo de Ativos (Screener)")
    st.markdown("Varredura simulada do mercado em tempo real aplicando regras quantitativas.")
    
    tickers = ["PETR4", "VALE3", "BOVA11", "ITUB4", "BBDC4", "MGLU3", "WEGE3", "RENT3"]
    
    # Geração de dados sintéticos para simular mercado real
    np.random.seed(datetime.datetime.now().microsecond)
    screener_data = []
    
    for t in tickers:
        iv_rank = np.random.randint(5, 95)
        hv = np.random.uniform(15.0, 70.0)
        vol_diario = np.random.uniform(50.0, 1500.0)
        bid_ask = np.random.uniform(0.01, 0.15)
        
        # Lógica de Recomendação Automática
        if iv_rank > 65:
            rec = "Iron Condor (Venda de Volatilidade)"
            color = "🔴"
        elif iv_rank < 30:
            rec = "Pozinho (Long Strangle)"
            color = "🟢"
        else:
            rec = "Trava de Alta / Baixa"
            color = "🟡"
            
        screener_data.append({
            'Ativo': t,
            'Volume (M R$)': round(vol_diario, 1),
            'Volatilidade Hist. (%)': round(hv, 1),
            'IV Rank (%)': iv_rank,
            'Spread Bid-Ask (R$)': round(bid_ask, 2),
            'Estratégia Recomendada': f"{color} {rec}"
        })
        
    df_screener = pd.DataFrame(screener_data)
    
    # Formatação condicional da tabela
    def highlight_iv(val):
        if isinstance(val, int):
            if val > 65:
                return 'background-color: rgba(255, 0, 0, 0.2)'
            elif val < 30:
                return 'background-color: rgba(0, 255, 0, 0.2)'
        return ''

    st.dataframe(
        df_screener.style.map(highlight_iv, subset=['IV Rank (%)']),
        use_container_width=True,
        hide_index=True
    )
    
    st.info("💡 **Dica de Engenharia Financeira:** O IV Rank compara a Volatilidade Implícita atual com o range histórico de 1 ano. Momentos de IV Rank alto favorecem estratégias de crédito (Vega negativo), enquanto IV Rank baixo favorece a compra de gama/vega (Pozinho).")