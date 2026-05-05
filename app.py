import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict, List

import database as db

# --------------------------
# Inicialização do Banco
# --------------------------
db.init_db()

# --------------------------
# Constantes / Configurações
# --------------------------
EXCHANGE_SUFFIX = {
    'brazil': '.SA',
    'united states': '',
    'canada': '.TO',
    'united kingdom': '.L',
    'germany': '.DE'
}

SUPPORTED_COUNTRIES = list(EXCHANGE_SUFFIX.keys())

BENCHMARKS = {
    'Ibovespa (Brasil)': {'ticker': '^BVSP', 'country': 'brazil'},
    'S&P 500 (EUA)': {'ticker': '^GSPC', 'country': 'united states'}
}

TOP_ASSETS = {
    'Brasil (B3)': ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3', 'WEGE3', 'BBAS3', 'RENT3', 'MGLU3'],
    'EUA (NYSE/NASDAQ)': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'JNJ'],
    'Criptoativos': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD']
}

# --------------------------
# Utilitários
# --------------------------
def _now() -> datetime:
    return datetime.now()

def _as_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _apply_country_suffix(ticker: str, country: Optional[str]) -> str:
    if not country:
        return ticker
    suffix = EXCHANGE_SUFFIX.get(country.lower(), '')
    return ticker + suffix if suffix and not ticker.endswith(suffix) else ticker

# --------------------------
# yfinance Wrappers / Cache
# --------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_crypto_info_from_symbol(symbol: str) -> Optional[str]:
    """Busca o símbolo negociável de cripto (ex: BTC -> BTC-USD)."""
    if not symbol:
        return None
    try:
        res = yf.search(symbol, exchange="CRYPTO", filter="all")
        if isinstance(res, pd.DataFrame) and not res.empty and 'symbol' in res.columns:
            return str(res['symbol'].iloc[0])
    except Exception:
        return None
    return None

def _history_for(query: str, start, end, interval: str):
    """Pequeno wrapper para facilitar testes/alterações."""
    try:
        return yf.Ticker(query).history(start=start, end=end, interval=interval, actions=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_data(portfolio_assets: List[dict], benchmark_info: dict) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    """Baixa preços históricos (1 ano) e preços atuais (último close) para a lista de ativos."""
    if not portfolio_assets:
        return {}, None

    now_dt = _now()
    start_dt = now_dt - timedelta(days=365)
    end_dt = now_dt

    current_prices: Dict[str, float] = {}
    hist_list: List[pd.Series] = []

    for asset in portfolio_assets:
        ticker_raw = (asset.get('Ticker') or '').upper()
        ttype = asset.get('Type', 'Ação')
        if not ticker_raw:
            continue

        if ttype == 'Criptoativo':
            query = get_crypto_info_from_symbol(ticker_raw)
            if not query:
                st.warning(f"Cripto não mapeada: {ticker_raw}")
                continue
        else:
            query = _apply_country_suffix(ticker_raw, asset.get('Country'))

        hist = _history_for(query, start_dt, end_dt, interval="1d")
        if isinstance(hist, pd.DataFrame) and not hist.empty:
            s = hist['Close'].rename(ticker_raw)
            s.index = pd.to_datetime(s.index)
            # handle timezone
            if s.index.tz is not None:
                s.index = s.index.tz_localize(None)
            hist_list.append(s)
            current_prices[ticker_raw] = _as_float(hist['Close'].iloc[-1])

    # benchmark
    if benchmark_info and benchmark_info.get('ticker'):
        bench_tkr = benchmark_info.get('ticker')
        bench_query = _apply_country_suffix(bench_tkr, benchmark_info.get('country'))
        hist_b = _history_for(bench_query, start_dt, end_dt, interval="1d")
        if isinstance(hist_b, pd.DataFrame) and not hist_b.empty:
            s_b = hist_b['Close'].rename(bench_tkr)
            s_b.index = pd.to_datetime(s_b.index)
            if s_b.index.tz is not None:
                s_b.index = s_b.index.tz_localize(None)
            hist_list.append(s_b)

    if not hist_list:
        return {}, None

    # Group by index date to avoid duplicates when concating if there are multiple times same day
    hist_list = [s.groupby(s.index.date).last() for s in hist_list]
    
    hist_df = pd.concat(hist_list, axis=1).sort_index().ffill().bfill()
    hist_df.index = pd.to_datetime(hist_df.index)
    return current_prices, hist_df

def get_price_at_datetime(ticker: str, dt: datetime, asset_type: str = 'Ação', country: Optional[str] = None) -> Optional[float]:
    """Retorna preço aproximado na data/hora (1m se recente, fallback daily)."""
    if not ticker or not isinstance(dt, datetime):
        return None
    try:
        if asset_type == 'Criptoativo':
            query = get_crypto_info_from_symbol(ticker)
            if not query:
                return None
        else:
            query = _apply_country_suffix(ticker, country)

        now_dt = _now()
        # intraday disponível tipicamente ~7 dias
        if (now_dt - dt) <= timedelta(days=7):
            start_dt = dt - timedelta(minutes=15)
            end_dt = dt + timedelta(minutes=15)
            df = _history_for(query, start_dt, end_dt, "1m")
            if not df.empty:
                idx = pd.to_datetime(df.index)
                pos = idx.get_indexer([dt], method="nearest")[0]
                return _as_float(df['Close'].iloc[pos], None)
        # diário fallback
        start_day = dt.date()
        end_day = start_day + timedelta(days=1)
        df_day = _history_for(query, start_day, end_day, "1d")
        if not df_day.empty:
            return _as_float(df_day['Close'].iloc[-1], None)
    except Exception:
        return None
    return None

# --------------------------
# Gerenciamento da Carteira
# --------------------------
class PortfolioManager:
    def __init__(self):
        self.portfolio = db.load_portfolio()

    def reload(self):
        self.portfolio = db.load_portfolio()

    def validate_asset_basics(self, asset_info: dict) -> Tuple[bool, Optional[str]]:
        if not asset_info.get('Nome') or not (asset_info.get('Ticker') or '').strip():
            return False, "Nome e Ticker são obrigatórios."
        if _as_float(asset_info.get('Quantidade'), np.nan) <= 0:
            return False, "Quantidade deve ser maior que 0."
        return True, None

    def provider_validate(self, asset_info: dict) -> Tuple[bool, Optional[str]]:
        ttype = asset_info.get('Type')
        ticker = (asset_info.get('Ticker') or '').upper()
        try:
            if ttype == 'Ação':
                country = asset_info.get('Country')
                if not country:
                    return False, "País obrigatório para Ação."
                query = _apply_country_suffix(ticker, country)
                df = _history_for(query, start=_now() - timedelta(days=7), end=_now(), interval="1d")
                if df.empty:
                    return False, "Ticker não retornou dados (verifique sufixo/país)."
            elif ttype == 'Criptoativo':
                mapped = get_crypto_info_from_symbol(ticker)
                if not mapped:
                    return False, "Símbolo cripto não encontrado."
            else:
                return False, f"Tipo de ativo inválido: {ttype}"
        except Exception as e:
            return False, str(e)
        return True, None

    def add_asset(self, asset_info: dict) -> bool:
        ok, msg = self.validate_asset_basics(asset_info)
        if not ok:
            st.error(msg)
            return False

        # tenta preencher preço se não informado ou inválido
        if _as_float(asset_info.get('Preço Compra'), np.nan) <= 0:
            try:
                dt_str = str(asset_info.get('Data Compra'))
                dt_obj = date.fromisoformat(dt_str) if isinstance(asset_info.get('Data Compra'), str) else asset_info.get('Data Compra')
                
                hr_str = str(asset_info.get('Hora Compra'))
                hr_obj = time.fromisoformat(hr_str) if isinstance(asset_info.get('Hora Compra'), str) else asset_info.get('Hora Compra')

                dt_purchase = datetime.combine(dt_obj, hr_obj)
                fetched = get_price_at_datetime(asset_info['Ticker'], dt_purchase, asset_info.get('Type', 'Ação'), asset_info.get('Country'))
                if fetched is not None:
                    asset_info['Preço Compra'] = fetched
            except Exception:
                pass

        ok, msg = self.provider_validate(asset_info)
        if not ok:
            st.error(msg)
            return False

        db.add_asset(asset_info)
        self.reload()
        st.success(f"{asset_info.get('Type')} '{asset_info.get('Nome')}' adicionada.")
        return True

    def remove_asset(self, ticker_to_remove: str):
        t = (ticker_to_remove or '').upper()
        db.remove_asset(t)
        self.reload()
        st.success("Ativo removido.")

    def get_market_data(self, benchmark_info):
        if not self.portfolio:
            return {}, None
        return fetch_market_data(self.portfolio, benchmark_info)

    def calculate_risk_and_performance(self, benchmark_info):
        current_prices, hist = self.get_market_data(benchmark_info)
        if not current_prices or hist is None or hist.empty:
            return None, None, None, None

        valid = [a for a in self.portfolio if a.get('Ticker') in current_prices]
        if not valid:
            return None, None, None, hist

        rows = []
        invested_total = 0.0
        current_total = 0.0
        
        # Agrupar ativos para pesos da carteira
        portfolio_weights = {}

        for a in valid:
            tkr = a['Ticker']
            price_now = _as_float(current_prices.get(tkr), np.nan)
            price_buy = _as_float(a.get('Preço Compra'), np.nan)
            qty = _as_float(a.get('Quantidade'), np.nan)
            invested = price_buy * qty if np.isfinite(price_buy) and np.isfinite(qty) else np.nan
            now_val = price_now * qty if np.isfinite(price_now) and np.isfinite(qty) else np.nan
            pct = (now_val / invested - 1) * 100 if invested and invested > 0 else np.nan
            if np.isfinite(invested):
                invested_total += invested
            if np.isfinite(now_val):
                current_total += now_val
                if tkr not in portfolio_weights:
                    portfolio_weights[tkr] = 0
                portfolio_weights[tkr] += now_val
                
            rows.append({
                "Nome": a['Nome'],
                "Ticker": tkr,
                "Tipo": a.get('Type', 'Ação'),
                "Preço Compra": price_buy,
                "Preço Atual": price_now,
                "Valor Investido": invested,
                "Valor Atual": now_val,
                "Resultado (%)": pct
            })

        df = pd.DataFrame(rows)
        df = _ensure_numeric(df, ["Preço Compra", "Preço Atual", "Valor Investido", "Valor Atual", "Resultado (%)"])

        summary = {
            "Custo Total da Carteira": float(invested_total),
            "Valor Total Atual": float(current_total),
            "Resultado Consolidado": float(current_total - invested_total),
            "Resultado Consolidado (%)": (float(current_total / invested_total) - 1) * 100 if invested_total > 0 else 0.0
        }

        # Calcular métricas de risco
        risk = {"Volatilidade Anualizada": "N/A", "Beta da Carteira": "N/A", "VaR Histórico (95%, 1 dia)": "N/A"}
        
        try:
            # Retornos diários dos ativos
            daily_returns = hist.pct_change().dropna()
            
            # Pesos atuais de cada ativo na carteira
            weights = np.array([portfolio_weights.get(tkr, 0) for tkr in portfolio_weights.keys()])
            if current_total > 0:
                weights = weights / current_total
            
            # Intersecção entre os pesos calculados e os dados históricos
            tickers_in_hist = [tkr for tkr in portfolio_weights.keys() if tkr in daily_returns.columns]
            
            if len(tickers_in_hist) > 0:
                hist_assets = daily_returns[tickers_in_hist]
                w = np.array([portfolio_weights[tkr] for tkr in tickers_in_hist])
                w = w / np.sum(w)
                
                # Retorno diário da carteira
                port_daily_ret = hist_assets.dot(w)
                
                # Volatilidade (Anualizada)
                volatility = port_daily_ret.std() * np.sqrt(252)
                risk["Volatilidade Anualizada"] = f"{volatility * 100:.2f}%"
                
                # VaR Histórico (95%)
                var_95 = np.percentile(port_daily_ret, 5)
                risk["VaR Histórico (95%, 1 dia)"] = f"{var_95 * 100:.2f}%"
                
                # Beta
                bench_tkr = benchmark_info.get('ticker')
                if bench_tkr and bench_tkr in daily_returns.columns:
                    bench_ret = daily_returns[bench_tkr]
                    cov_matrix = np.cov(port_daily_ret, bench_ret)
                    if cov_matrix[1, 1] > 0:
                        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                        risk["Beta da Carteira"] = f"{beta:.2f}"
        except Exception as e:
            print(f"Erro calculando risco: {e}")

        return df, summary, risk, hist

# --------------------------
# UI - Estilos
# --------------------------
def inject_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border: 1px solid #334155;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --------------------------
# UI - Sidebar Helpers
# --------------------------
def processar_adicao_ativo():
    manager = PortfolioManager()
    
    # Tratamento para tickers sugeridos
    ticker_val = st.session_state.add_asset_ticker_sugerido
    if ticker_val == "Outro":
        ticker_val = st.session_state.add_asset_ticker_customizado

    asset_info = {
        'Type': st.session_state.add_asset_type,
        'Nome': st.session_state.add_asset_name,
        'Ticker': (ticker_val or "").upper(),
        'Country': st.session_state.get('add_asset_country'),
        'Data Compra': st.session_state.add_asset_date,
        'Hora Compra': st.session_state.add_asset_time,
        'Preço Compra': st.session_state.add_asset_price,
        'Quantidade': st.session_state.add_asset_quantity
    }
    
    if manager.add_asset(asset_info):
        st.session_state.add_asset_name = ""
        st.session_state.add_asset_ticker_customizado = ""
        st.session_state.add_asset_quantity = 0.0
        st.session_state.add_asset_price = 0.0
        st.cache_data.clear()

def build_sidebar():
    st.sidebar.header("Gerenciar Carteira")
    benchmark_name = st.sidebar.selectbox("Benchmark de Referência", options=list(BENCHMARKS.keys()))
    st.session_state.selected_benchmark = BENCHMARKS[benchmark_name]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Adicionar Novo Ativo")
    
    tipo_ativo = st.sidebar.selectbox("Tipo de Ativo", options=['Ação', 'Criptoativo'], key="add_asset_type")
    
    if tipo_ativo == 'Ação':
        regiao = st.sidebar.selectbox("Região", options=['Brasil (B3)', 'EUA (NYSE/NASDAQ)'], key="add_asset_region")
        opcoes_tickers = TOP_ASSETS[regiao] + ["Outro"]
        if regiao == 'Brasil (B3)':
            st.session_state.add_asset_country = 'brazil'
        else:
            st.session_state.add_asset_country = 'united states'
    else:
        opcoes_tickers = TOP_ASSETS['Criptoativos'] + ["Outro"]
        
    ticker_sugerido = st.sidebar.selectbox("Ticker / Símbolo", options=opcoes_tickers, key="add_asset_ticker_sugerido")
    
    if ticker_sugerido == "Outro":
        st.sidebar.text_input("Digite o Ticker Manualmente", key="add_asset_ticker_customizado")
        if tipo_ativo == 'Ação':
            st.sidebar.selectbox("País da Bolsa (Outro)", options=SUPPORTED_COUNTRIES, key="add_asset_country_custom")
            st.session_state.add_asset_country = st.session_state.add_asset_country_custom

    st.sidebar.text_input("Nome Amigável (ex: Petrobras, Apple)", key="add_asset_name")
    
    st.sidebar.date_input("Data de Compra", value=_now().date(), key="add_asset_date")
    st.sidebar.time_input("Hora da Compra (opcional)", value=_now().time(), key="add_asset_time")
    st.sidebar.number_input("Preço Unitário (Deixe 0 para auto-preencher)", min_value=0.0, format="%.6f", key="add_asset_price")
    st.sidebar.number_input("Quantidade de Cotas", min_value=0.000001, format="%.6f", key="add_asset_quantity")
    
    st.sidebar.button("➕ Adicionar Ativo", on_click=processar_adicao_ativo, use_container_width=True)

    manager = PortfolioManager()
    if manager.portfolio:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Remover Ativo")
        # Unique tickers
        tickers = list(set([f"{a['Nome']} ({a['Ticker']})" for a in manager.portfolio]))
        sel = st.sidebar.selectbox("Selecione o Ativo para remover", options=tickers)
        if st.sidebar.button("🗑️ Remover Ativo Selecionado", use_container_width=True):
            t = sel.split('(')[-1].replace(')', '').strip()
            manager.remove_asset(t)
            st.cache_data.clear()
            st.rerun()

# --------------------------
# UI - Main
# --------------------------
def render_metric(label, value, is_currency=True, is_percent=False):
    prefix = "R$ " if is_currency else ""
    suffix = "%" if is_percent else ""
    
    if isinstance(value, (int, float)):
        color_class = ""
        if is_percent or label.startswith("Resultado"):
            if value > 0: color_class = "positive"
            elif value < 0: color_class = "negative"
            
        if is_percent:
            formatted_value = f"{value:.2f}{suffix}"
        else:
            formatted_value = f"{prefix}{value:,.2f}"
    else:
        color_class = ""
        formatted_value = value

    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value {color_class}">{formatted_value}</div>
        </div>
    """, unsafe_allow_html=True)

def build_main():
    st.title("📊 Painel da Carteira de Investimentos")
    inject_custom_css()
    
    manager = PortfolioManager()
    
    if not manager.portfolio:
        st.info("👋 Bem-vindo! Sua carteira está vazia. Utilize a barra lateral à esquerda para adicionar seus primeiros ativos e começar a analisar.")
        return

    with st.spinner("Buscando dados de mercado e calculando riscos..."):
        perf_df, summary, risk_metrics, hist = manager.calculate_risk_and_performance(st.session_state.get('selected_benchmark', BENCHMARKS['Ibovespa (Brasil)']))

    if isinstance(perf_df, pd.DataFrame) and not perf_df.empty:
        
        # Tabs for better organization
        tab_overview, tab_risk, tab_assets = st.tabs(["👁️ Visão Geral", "📈 Risco e Performance", "💼 Meus Ativos"])
        
        with tab_overview:
            st.markdown("### Resumo Financeiro")
            col1, col2, col3, col4 = st.columns(4)
            with col1: render_metric("Total Investido", summary['Custo Total da Carteira'])
            with col2: render_metric("Patrimônio Atual", summary['Valor Total Atual'])
            with col3: render_metric("Lucro/Prejuízo", summary['Resultado Consolidado'])
            with col4: render_metric("Rentabilidade", summary['Resultado Consolidado (%)'], is_currency=False, is_percent=True)
            
            st.markdown("---")
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                st.markdown("#### Composição da Carteira")
                fig_pie = px.pie(perf_df.dropna(subset=["Valor Atual"]), names='Nome', values='Valor Atual', hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Teal)
                fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc")
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with col_chart2:
                if hist is not None and not hist.empty:
                    st.markdown("#### Evolução Normalizada (Último Ano)")
                    # Normalizando para base 100
                    hist_norm = (hist / hist.iloc[0]) * 100
                    
                    fig_line = go.Figure()
                    
                    # Plot benchmark se existir
                    bench_tkr = st.session_state.get('selected_benchmark', {}).get('ticker')
                    
                    # Plotar ativos top 5 por valor
                    top_assets = perf_df.nlargest(5, 'Valor Atual')['Ticker'].tolist()
                    
                    for col in hist_norm.columns:
                        if col == bench_tkr:
                            fig_line.add_trace(go.Scatter(x=hist_norm.index, y=hist_norm[col], mode='lines', name=f"Benchmark ({col})", line=dict(color='white', width=2, dash='dash')))
                        elif col in top_assets:
                            fig_line.add_trace(go.Scatter(x=hist_norm.index, y=hist_norm[col], mode='lines', name=col))
                            
                    fig_line.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#f8fafc", showlegend=True)
                    st.plotly_chart(fig_line, use_container_width=True)

        with tab_risk:
            st.markdown("### Métricas de Risco")
            st.write("Análise quantitativa baseada nos últimos 12 meses de negociação.")
            
            r_col1, r_col2, r_col3 = st.columns(3)
            with r_col1: render_metric("Volatilidade Anual", risk_metrics["Volatilidade Anualizada"], is_currency=False)
            with r_col2: render_metric("Beta vs Benchmark", risk_metrics["Beta da Carteira"], is_currency=False)
            with r_col3: render_metric("Value at Risk (95%)", risk_metrics["VaR Histórico (95%, 1 dia)"], is_currency=False)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.info("""
            **Entendendo as métricas:**
            - **Volatilidade:** Mede a variação dos preços. Quanto maior, maior o sobe-e-desce da carteira.
            - **Beta:** Mede a sensibilidade em relação ao mercado (Benchmark). Beta > 1 significa que a carteira é mais volátil que o mercado.
            - **Value at Risk (VaR):** A perda máxima esperada em 1 dia, com 95% de confiança.
            """)

        with tab_assets:
            st.markdown("### Posições Detalhadas")
            
            styled_df = perf_df.style.format({
                "Preço Compra": "R$ {:,.2f}", 
                "Preço Atual": "R$ {:,.2f}",
                "Valor Investido": "R$ {:,.2f}", 
                "Valor Atual": "R$ {:,.2f}",
                "Resultado (%)": "{:+.2f}%"
            }).applymap(lambda x: 'color: #10b981' if x > 0 else ('color: #ef4444' if x < 0 else ''), subset=['Resultado (%)'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)

    else:
        st.warning("Aguardando adição de ativos válidos ou dados disponíveis para gerar o relatório.")

# --------------------------
# Inicialização
# --------------------------
st.set_page_config(page_title="Painel de Investimentos", page_icon="📈", layout="wide")

# manter chaves do formulário para evitar KeyError
for k, v in {
    "add_asset_type": "Ação",
    "add_asset_name": "",
    "add_asset_ticker_sugerido": "",
    "add_asset_ticker_customizado": "",
    "add_asset_country": SUPPORTED_COUNTRIES[0],
    "add_asset_date": _now().date(),
    "add_asset_time": _now().time(),
    "add_asset_price": 0.0,
    "add_asset_quantity": 0.0,
    "selected_benchmark": BENCHMARKS[list(BENCHMARKS.keys())[0]]
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

build_sidebar()
build_main()