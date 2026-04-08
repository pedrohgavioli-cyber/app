import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date, time
import streamlit as st
import plotly.express as px
from typing import Optional, Tuple, Dict, List

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
            hist_list.append(s_b)

    if not hist_list:
        return {}, None

    hist_df = pd.concat(hist_list, axis=1).sort_index().ffill().bfill()
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
    def __init__(self, portfolio_data: List[dict]):
        self.portfolio = portfolio_data

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
                dt_purchase = datetime.combine(
                    asset_info.get('Data Compra') if isinstance(asset_info.get('Data Compra'), date) else date.fromisoformat(str(asset_info.get('Data Compra'))),
                    asset_info.get('Hora Compra') if isinstance(asset_info.get('Hora Compra'), time) else time(0, 0)
                )
                fetched = get_price_at_datetime(asset_info['Ticker'], dt_purchase, asset_info.get('Type', 'Ação'), asset_info.get('Country'))
                if fetched is not None:
                    asset_info['Preço Compra'] = fetched
            except Exception:
                pass

        ok, msg = self.provider_validate(asset_info)
        if not ok:
            st.error(msg)
            return False

        self.portfolio.append(asset_info)
        st.success(f"{asset_info.get('Type')} '{asset_info.get('Nome')}' adicionada.")
        return True

    def remove_asset(self, ticker_to_remove: str):
        orig = len(self.portfolio)
        t = (ticker_to_remove or '').upper()
        self.portfolio = [a for a in self.portfolio if (a.get('Ticker') or '').upper() != t]
        if len(self.portfolio) < orig:
            st.success("Ativo removido.")
        else:
            st.info("Nada a remover.")

    def get_market_data(self, benchmark_info):
        if not self.portfolio:
            return {}, None
        return fetch_market_data(self.portfolio, benchmark_info)

    def calculate_risk_and_performance(self, benchmark_info):
        current_prices, hist = self.get_market_data(benchmark_info)
        if not current_prices or hist is None:
            return None, None, None, None

        valid = [a for a in self.portfolio if a.get('Ticker') in current_prices]
        if not valid:
            return None, None, None, hist

        rows = []
        invested_total = 0.0
        current_total = 0.0
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

        risk = {"Volatilidade Anualizada": "N/A", "Beta da Carteira": "N/A", "VaR Histórico (95%, 1 dia)": "N/A"}
        # cálculos de risco podem ser adicionados aqui usando 'hist' e 'df'
        return df, summary, risk, hist

# --------------------------
# UI - Sidebar Helpers
# --------------------------
def processar_adicao_ativo():
    asset_info = {
        'Type': st.session_state.add_asset_type,
        'Nome': st.session_state.add_asset_name,
        'Ticker': (st.session_state.add_asset_ticker or "").upper(),
        'Country': st.session_state.get('add_asset_country'),
        'Data Compra': st.session_state.add_asset_date,
        'Hora Compra': st.session_state.add_asset_time,
        'Preço Compra': st.session_state.add_asset_price,
        'Quantidade': st.session_state.add_asset_quantity
    }
    manager = PortfolioManager(st.session_state.portfolio)
    if manager.add_asset(asset_info):
        st.session_state.add_asset_name = ""
        st.session_state.add_asset_ticker = ""
        st.session_state.add_asset_quantity = 0.0
        st.session_state.add_asset_price = 0.0
        st.cache_data.clear()

def build_sidebar():
    st.sidebar.header("Gerenciar Carteira")
    benchmark_name = st.sidebar.selectbox("Selecione o Benchmark para Ações", options=list(BENCHMARKS.keys()))
    st.session_state.selected_benchmark = BENCHMARKS[benchmark_name]

    st.sidebar.subheader("Adicionar Novo Ativo")
    st.sidebar.selectbox("Tipo de Ativo", options=['Ação', 'Criptoativo'], key="add_asset_type")
    st.sidebar.text_input("Nome do Ativo (ex: Petrobras, Bitcoin)", key="add_asset_name")
    st.sidebar.text_input("Ticker/Símbolo (ex: PETR4, BTC)", key="add_asset_ticker")
    if st.session_state.add_asset_type == 'Ação':
        st.sidebar.selectbox("País da Bolsa", options=SUPPORTED_COUNTRIES, key="add_asset_country")
    st.sidebar.date_input("Data de Compra", value=_now().date(), key="add_asset_date")
    st.sidebar.time_input("Hora da Compra (opcional)", value=_now().time(), key="add_asset_time")
    st.sidebar.number_input("Preço de Compra (deixe 0 para importar automaticamente)", min_value=0.0, format="%.6f", key="add_asset_price")
    st.sidebar.number_input("Quantidade", min_value=0.000001, format="%.6f", key="add_asset_quantity")
    st.sidebar.button("Adicionar Ativo", on_click=processar_adicao_ativo, use_container_width=True)

    if st.session_state.portfolio:
        st.sidebar.subheader("Remover Ativo")
        tickers = [f"{a['Nome']} ({a['Ticker']})" for a in st.session_state.portfolio]
        sel = st.sidebar.selectbox("Selecione o Ativo para remover", options=tickers)
        if st.sidebar.button("Remover Ativo Selecionado", use_container_width=True):
            t = sel.split('(')[-1].replace(')', '').strip()
            PortfolioManager(st.session_state.portfolio).remove_asset(t)
            st.cache_data.clear()
            st.rerun()

# --------------------------
# UI - Main
# --------------------------
def build_main():
    st.title("Analisador de Carteira de Investimentos")
    if not st.session_state.portfolio:
        st.info("Sua carteira está vazia. Adicione um ativo na barra lateral para começar.")
        return

    manager = PortfolioManager(st.session_state.portfolio)
    with st.spinner("Buscando dados de mercado..."):
        perf_df, summary, risk_metrics, hist = manager.calculate_risk_and_performance(st.session_state.get('selected_benchmark', {}))

    if isinstance(perf_df, pd.DataFrame) and not perf_df.empty:
        st.header("Relatório de Desempenho")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Custo Total", f"{summary['Custo Total da Carteira']:.2f}")
        col2.metric("Valor Atual", f"{summary['Valor Total Atual']:.2f}")
        col3.metric("Resultado (R$)", f"{summary['Resultado Consolidado']:,.2f}")
        col4.metric("Resultado (%)", f"{summary['Resultado Consolidado (%)']:.2f}%")

        fig_pie = px.pie(perf_df.dropna(subset=["Valor Atual"]), names='Nome', values='Valor Atual', title='Composição por Valor Atual')
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Desempenho Detalhado dos Ativos")
        st.dataframe(perf_df.style.format({
            "Preço Compra": "{:,.6f}", "Preço Atual": "{:,.6f}",
            "Valor Investido": "{:,.2f}", "Valor Atual": "{:,.2f}",
            "Resultado (%)": "{:+.2f}%"
        }), use_container_width=True)

    else:
        st.warning("Aguardando adição de ativos válidos ou dados disponíveis para gerar o relatório.")

# --------------------------
# Inicialização
# --------------------------
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

# manter chaves do formulário para evitar KeyError
for k, v in {
    "add_asset_type": "Ação",
    "add_asset_name": "",
    "add_asset_ticker": "",
    "add_asset_country": SUPPORTED_COUNTRIES[0],
    "add_asset_date": _now().date(),
    "add_asset_time": _now().time(),
    "add_asset_price": 0.0,
    "add_asset_quantity": 0.0,
    "selected_benchmark": BENCHMARKS[list(BENCHMARKS.keys())[0]]
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.set_page_config(page_title="Analisador de Carteira Global", layout="wide")
build_sidebar()
build_main()