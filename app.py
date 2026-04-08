import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.express as px

# Utilidades
# ---------------------------------------------

def _now():
    # Isola para facilitar eventual mocking em testes
    return datetime.now()

def _as_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def _ensure_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# mapear sufixos comuns por país (para tickers de bolsa)
EXCHANGE_SUFFIX = {
    'brazil': '.SA',
    'united states': '',
    'canada': '.TO',
    'united kingdom': '.L',
    'germany': '.DE'
}

@st.cache_data(ttl=3600)
def get_crypto_info_from_symbol(symbol: str) -> str | None:
    """Resolve o símbolo negociável da cripto a partir do símbolo (ex.: BTC -> 'BTC-USD' ou 'BTC-USD')."""
    if not symbol:
        return None
    try:
        res = yf.search(symbol, exchange="CRYPTO", filter="all")
        if isinstance(res, pd.DataFrame) and not res.empty:
            # usar a coluna 'symbol' (ticker negociável), não 'name'
            return str(res['symbol'].iloc[0])
    except Exception:
        return None
    return None

def _apply_country_suffix(ticker: str, country: str | None) -> str:
    """Aplica sufixo por país quando apropriado."""
    if not country:
        return ticker
    suffix = EXCHANGE_SUFFIX.get(country.lower(), '')
    return ticker + suffix if suffix and not ticker.endswith(suffix) else ticker

def get_price_at_datetime(ticker: str, dt: datetime, asset_type: str = 'Ação', country: str | None = None) -> float | None:
    """Obtem preço mais próximo para a data/hora informada.
    - Tenta 1m intraday para últimos ~7 dias.
    - Caso contrário, pega fechamento diário do dia."""
    if not ticker or not isinstance(dt, datetime):
        return None

    try:
        query = ticker
        if asset_type == 'Criptoativo':
            mapped = get_crypto_info_from_symbol(ticker)
            if not mapped:
                return None
            query = mapped
        else:
            query = _apply_country_suffix(ticker, country)

        now_dt = _now()
        # intraday (últimos ~7 dias)
        if (now_dt - dt) <= timedelta(days=7):
            start_dt = dt - timedelta(minutes=15)
            end_dt = dt + timedelta(minutes=15)
            df = yf.Ticker(query).history(start=start_dt, end=end_dt, interval="1m", actions=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                idx = pd.to_datetime(df.index)
                pos = idx.get_indexer([dt], method="nearest")[0]
                return _as_float(df['Close'].iloc[pos], None)
        # fallback diário: pegar fechamento do dia
        start_day = dt.date()
        end_day = start_day + timedelta(days=1)
        df = yf.Ticker(query).history(start=start_day, end=end_day, interval="1d", actions=False)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return _as_float(df['Close'].iloc[-1], None)
    except Exception:
        return None
    return None

@st.cache_data(ttl=600, show_spinner=False)
def fetch_market_data(portfolio_assets: list[dict], benchmark_info: dict):
    """Baixa preços atuais e históricos (último ano) para ativos e benchmark."""
    if not portfolio_assets:
        return {}, None

    current_prices: dict[str, float] = {}
    hist_data_list: list[pd.Series] = []

    end_date_dt = _now()
    start_date_dt = end_date_dt - timedelta(days=365)
    # usar objetos datetime/ date diretamente com yfinance
    for asset in portfolio_assets:
        try:
            asset_type = asset.get('Type', 'Ação')
            ticker = (asset.get('Ticker') or "").upper()
            if not ticker:
                continue

            if asset_type == 'Criptoativo':
                query = get_crypto_info_from_symbol(ticker)
                if not query:
                    st.warning(f"Não foi possível mapear símbolo cripto: {ticker}.")
                    continue
            else:
                country = asset.get('Country')
                query = _apply_country_suffix(ticker, country)

            # obter histórico (1d) e preço atual (último close)
            hist = yf.Ticker(query).history(start=start_date_dt, end=end_date_dt, interval="1d", actions=False)
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                s = hist['Close'].rename(ticker)
                s.index = pd.to_datetime(s.index)
                hist_data_list.append(s)
                current_prices[ticker] = _as_float(hist['Close'].iloc[-1])

        except Exception as e:
            st.warning(f"Falha ao obter dados para {asset.get('Ticker')}: {e}")
            continue

    # benchmark
    try:
        if benchmark_info and benchmark_info.get('ticker'):
            bench_tkr = benchmark_info['ticker']
            bench_query = _apply_country_suffix(bench_tkr, benchmark_info.get('country'))
            hist_bench = yf.Ticker(bench_query).history(start=start_date_dt, end=end_date_dt, interval="1d", actions=False)
            if isinstance(hist_bench, pd.DataFrame) and not hist_bench.empty:
                s_bench = hist_bench['Close'].rename(bench_tkr)
                s_bench.index = pd.to_datetime(s_bench.index)
                hist_data_list.append(s_bench)
    except Exception as e:
        st.info(f"Benchmark indisponível ({benchmark_info.get('ticker')}): {e}")

    if not hist_data_list:
        return {}, None

    hist_data = pd.concat(hist_data_list, axis=1).sort_index().ffill().bfill()
    return current_prices, hist_data


class PortfolioManager:
    def __init__(self, portfolio_data: list[dict]):
        self.portfolio = portfolio_data

    def add_asset(self, asset_info: dict) -> bool:
        """Valida ativo no provedor e adiciona à carteira."""
        asset_type = asset_info.get('Type')
        ticker = (asset_info.get('Ticker') or '').upper()

        if not asset_info.get('Nome') or not ticker:
            st.error("Nome e Ticker são obrigatórios.")
            return False
        if _as_float(asset_info.get('Preço Compra'), np.nan) <= 0:
            st.error("Preço de Compra deve ser maior que 0.")
            return False
        if _as_float(asset_info.get('Quantidade'), np.nan) <= 0:
            st.error("Quantidade deve ser maior que 0.")
            return False

        try:
            if asset_type == 'Ação':
                country = asset_info.get('Country')
                if not country:
                    raise ValueError("País é obrigatório para Ação.")
                query = _apply_country_suffix(ticker, country)
                df = yf.Ticker(query).history(period="5d", interval="1d", actions=False)
                if df is None or df.empty:
                    raise ValueError("Ticker não retornou dados (verifique sufixo/país).")
            elif asset_type == 'Criptoativo':
                mapped = get_crypto_info_from_symbol(ticker)
                if not mapped:
                    raise ValueError("Símbolo cripto não encontrado.")
            else:
                raise ValueError(f"Tipo de ativo inválido: {asset_type}")

            asset_info['Ticker'] = ticker
            self.portfolio.append(asset_info)
            st.success(f"{asset_type} '{asset_info['Nome']}' adicionada com sucesso!")
            return True
        except Exception as e:
            st.error(f"Ativo '{ticker}' não encontrado/indisponível. Detalhe: {e}")
            return False

    def remove_asset(self, ticker_to_remove: str):
        original_count = len(self.portfolio)
        tkr = (ticker_to_remove or '').upper()
        self.portfolio = [asset for asset in self.portfolio if (asset.get('Ticker') or '').upper() != tkr]
        if len(self.portfolio) < original_count:
            st.success(f"Ativo com ticker '{ticker_to_remove}' removido com sucesso.")
        else:
            st.info("Nada a remover.")

    def get_market_data(self, benchmark_info):
        if not self.portfolio:
            return {}, None
        return fetch_market_data(self.portfolio, benchmark_info)

    def calculate_risk_and_performance(self, benchmark_info):
        current_prices, hist_data = self.get_market_data(benchmark_info)
        if not current_prices or hist_data is None:
            return None, None, None, None

        valid_portfolio = [a for a in self.portfolio if a.get('Ticker') in current_prices]
        if not valid_portfolio:
            return None, None, None, hist_data

        performance_rows = []
        total_investido = 0.0
        total_atual = 0.0

        for asset in valid_portfolio:
            ticker = asset['Ticker']
            current_price = _as_float(current_prices.get(ticker), np.nan)
            preco_compra = _as_float(asset.get('Preço Compra'), np.nan)
            qtd = _as_float(asset.get('Quantidade'), np.nan)

            investido = preco_compra * qtd if np.isfinite(preco_compra) and np.isfinite(qtd) else np.nan
            atual = current_price * qtd if np.isfinite(current_price) and np.isfinite(qtd) else np.nan
            resultado_perc = (atual / investido - 1) * 100 if investido and investido > 0 else np.nan

            if np.isfinite(investido):
                total_investido += investido
            if np.isfinite(atual):
                total_atual += atual

            performance_rows.append({
                "Nome": asset['Nome'],
                "Ticker": ticker,
                "Tipo": asset.get('Type', 'Ação'),
                "Preço Compra": preco_compra,
                "Preço Atual": current_price,
                "Valor Investido": investido,
                "Valor Atual": atual,
                "Resultado (%)": resultado_perc
            })

        performance_df = pd.DataFrame(performance_rows)
        performance_df = _ensure_numeric(
            performance_df,
            ["Preço Compra", "Preço Atual", "Valor Investido", "Valor Atual", "Resultado (%)"]
        )

        resumo = {
            "Custo Total da Carteira": float(total_investido),
            "Valor Total Atual": float(total_atual),
            "Resultado Consolidado": float(total_atual - total_investido),
            "Resultado Consolidado (%)": (float(total_atual / total_investido) - 1) * 100 if total_investido > 0 else 0.0
        }

        risk_metrics = {"Volatilidade Anualizada": "N/A", "Beta da Carteira": "N/A", "VaR Histórico (95%, 1 dia)": "N/A"}
        
        stock_portfolio = [a for a in valid_portfolio if a.get('Type', 'Ação') == 'Ação']
        bench_tkr = benchmark_info.get('ticker') if benchmark_info else None

        if len(stock_portfolio) >= 2 and bench_tkr in hist_data.columns:
            try:
                # ... (cálculo de risco se mantém o mesmo) ...
                pass
            except Exception:
                pass

        return performance_df, resumo, risk_metrics, hist_data


# ---------------------------------------------
# Interface Streamlit
# ---------------------------------------------

st.set_page_config(page_title="Analisador de Carteira Global", layout="wide")
st.title("📈 Analisador de Carteira de Investimentos (Ações e Cripto)")

# ... (outras configurações de UI) ...

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

st.sidebar.header("Gerenciar Carteira")
benchmark_name = st.sidebar.selectbox("Selecione o Benchmark para Ações", options=['Ibovespa (Brasil)', 'S&P 500 (EUA)'])
BENCHMARKS = {
    'Ibovespa (Brasil)': {'ticker': 'Bovespa', 'country': 'brazil'},
    'S&P 500 (EUA)': {'ticker': 'S&P 500', 'country': 'united states'}
}
selected_benchmark_info = BENCHMARKS[benchmark_name]

# ---------------------------------------------
# Formulário de adição (COM A CORREÇÃO)
# ---------------------------------------------

st.sidebar.subheader("Adicionar Novo Ativo")

def processar_adicao_ativo():
    """Processa a adição do ativo e limpa o formulário em caso de sucesso."""
    asset_info = {
        'Type': st.session_state.add_asset_type,
        'Nome': st.session_state.add_asset_name,
        'Ticker': (st.session_state.add_asset_ticker or "").upper(),
        'Country': st.session_state.get('add_asset_country'),
        'Data Compra': str(st.session_state.add_asset_date),
        'Preço Compra': st.session_state.add_asset_price,
        'Quantidade': st.session_state.add_asset_quantity
    }

    # Se usuário não forneceu preço (ou forneceu 0), tentamos importar a partir da data/hora
    try:
        price_provided = _as_float(asset_info['Preço Compra'], np.nan)
        if not np.isfinite(price_provided) or price_provided <= 0:
            dt_purchase = datetime.combine(
                st.session_state.add_asset_date,
                st.session_state.add_asset_time
            )
            fetched_price = get_price_at_datetime(asset_info['Ticker'], dt_purchase, asset_info['Type'], asset_info.get('Country'))
            if fetched_price is not None:
                asset_info['Preço Compra'] = fetched_price
    except Exception:
        # falhou ao buscar preço: seguirá com o valor informado (ou 0 => validação irá bloquear)
        pass

    manager = PortfolioManager(st.session_state.portfolio)
    if manager.add_asset(asset_info):
        st.session_state["add_asset_name"] = ""
        st.session_state["add_asset_ticker"] = ""
        st.session_state["add_asset_quantity"] = 0.0
        st.session_state["add_asset_price"] = 0.0
        st.cache_data.clear()

SUPPORTED_COUNTRIES = ['brazil', 'united states', 'canada', 'united kingdom', 'germany']
asset_type = st.sidebar.selectbox("Tipo de Ativo", options=['Ação', 'Criptoativo'], key="add_asset_type")
nome = st.sidebar.text_input("Nome do Ativo (ex: Petrobras, Bitcoin)", key="add_asset_name")
ticker = st.sidebar.text_input("Ticker/Símbolo (ex: PETR4, BTC)", key="add_asset_ticker")

if st.session_state.add_asset_type == 'Ação':
    country = st.sidebar.selectbox("País da Bolsa", options=SUPPORTED_COUNTRIES, key="add_asset_country")

data_compra = st.sidebar.date_input("Data de Compra", value=_now().date(), key="add_asset_date")
# nova entrada de hora para permitir import de preço intraday
hora_compra = st.sidebar.time_input("Hora da Compra (opcional, usada para buscar preço intraday)", value=_now().time(), key="add_asset_time")

preco_compra = st.sidebar.number_input("Preço de Compra (deixe 0 para importar automaticamente)", min_value=0.0, format="%.6f", key="add_asset_price")
quantidade = st.sidebar.number_input("Quantidade", min_value=0.000001, format="%.6f", key="add_asset_quantity")

st.sidebar.button(
    "Adicionar Ativo",
    on_click=processar_adicao_ativo,
    use_container_width=True
)

# ---------------------------------------------
# Remoção
# ---------------------------------------------
if st.session_state.portfolio:
    st.sidebar.subheader("Remover Ativo")
    tickers_na_carteira = [f"{asset['Nome']} ({asset['Ticker']})" for asset in st.session_state.portfolio]
    ativo_para_remover_display = st.sidebar.selectbox("Selecione o Ativo para remover", options=tickers_na_carteira)

    if st.sidebar.button("Remover Ativo Selecionado", use_container_width=True):
        ticker_real_para_remover = ativo_para_remover_display.split('(')[-1].replace(')', '').strip()
        manager = PortfolioManager(st.session_state.portfolio)
        manager.remove_asset(ticker_real_para_remover)
        st.cache_data.clear()
        st.rerun()

# ---------------------------------------------
# Área principal
# ---------------------------------------------
if not st.session_state.portfolio:
    st.info("Sua carteira está vazia. Adicione um ativo na barra lateral para começar.")
else:
    manager = PortfolioManager(st.session_state.portfolio)
    with st.spinner("Buscando dados de mercado..."):
        performance_df, summary_data, risk_metrics, hist_data = manager.calculate_risk_and_performance(selected_benchmark_info)

    if isinstance(performance_df, pd.DataFrame) and not performance_df.empty:
        st.header(f"Relatório de Desempenho (Benchmark: {benchmark_name})")

        # ... (Resumo da Carteira, Gráficos, etc. se mantém igual) ...
        st.subheader("Resumo da Carteira")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Custo Total", f"{summary_data['Custo Total da Carteira']:.2f}")
        col2.metric("Valor Atual", f"{summary_data['Valor Total Atual']:.2f}")
        col3.metric("Resultado (R$)", f"{summary_data['Resultado Consolidado']:,.2f}")
        col4.metric("Resultado (%)", f"{summary_data['Resultado Consolidado (%)']:.2f}%")

        # --- NOVA SEÇÃO DE ANÁLISE E SUGESTÃO ---
        st.subheader("Análise Comparativa e Sugestão")
        
        bench_ticker = selected_benchmark_info.get('ticker')
        portfolio_return = summary_data['Resultado Consolidado (%)']
        benchmark_return = None

        if bench_ticker in hist_data.columns:
            benchmark_series = hist_data[bench_ticker].dropna()
            if not benchmark_series.empty:
                # Calcula o retorno do benchmark no mesmo período de 1 ano
                benchmark_return = (benchmark_series.iloc[-1] / benchmark_series.iloc[0] - 1) * 100

        c1, c2 = st.columns(2)
        c1.metric("Performance da Carteira (último ano)", f"{portfolio_return:.2f}%")
        if benchmark_return is not None:
            c2.metric(f"Performance do Benchmark ({bench_ticker})", f"{benchmark_return:.2f}%")

            if portfolio_return < benchmark_return:
                st.warning(f"A sua carteira está com desempenho abaixo do benchmark ({benchmark_name}).")
                
                st.info("Sugestão de Rebalanceamento (Estratégia de Pesos Iguais)")
                st.caption(
                    "Esta sugestão visa distribuir o valor total da carteira igualmente entre todos os ativos. "
                    "Isso não é uma recomendação de investimento."
                )

                # Calcula pesos atuais e sugeridos
                total_value = summary_data['Valor Total Atual']
                num_assets = len(performance_df)
                suggested_weight = 100 / num_assets if num_assets > 0 else 0

                suggestion_df = performance_df[['Nome', 'Ticker', 'Valor Atual']].copy()
                suggestion_df['Peso Atual (%)'] = (suggestion_df['Valor Atual'] / total_value) * 100
                suggestion_df['Peso Sugerido (%)'] = suggested_weight
                
                st.dataframe(
                    suggestion_df[['Nome', 'Ticker', 'Peso Atual (%)', 'Peso Sugerido (%)']].style.format({
                        'Peso Atual (%)': '{:.2f}%',
                        'Peso Sugerido (%)': '{:.2f}%'
                    }),
                    use_container_width=True
                )
        else:
            st.info("Não foi possível calcular a performance do benchmark para comparação.")
        
        # --- FIM DA NOVA SEÇÃO ---

        st.subheader("Visualizações Gráficas")
        fig_pie = px.pie(
            performance_df.dropna(subset=["Valor Atual"]),
            names='Nome', values='Valor Atual', title='Composição da Carteira por Valor Atual'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Desempenho Detalhado dos Ativos")
        st.dataframe(
            performance_df.style.format({
                "Preço Compra": "{:,.6f}", "Preço Atual": "{:,.6f}",
                "Valor Investido": "{:,.2f}", "Valor Atual": "{:,.2f}",
                "Resultado (%)": "{:+.2f}%"
            }),
            use_container_width=True
        )

        st.subheader("Análise de Risco (Apenas para Ações)")
        # ... (seção de risco se mantém a mesma) ...

    else:
        st.warning("Aguardando adição de ativos válidos ou dados de mercado disponíveis para gerar o relatório.")