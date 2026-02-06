import ccxt
import pandas as pd
import numpy as np
from pathlib import Path
import talib
import matplotlib.pyplot as plt
import itertools
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from matplotlib.ticker import FuncFormatter
import seaborn as sns

# =====================================================
# =================== DATA CACHE ======================
# =====================================================

class DataCache:
    def __init__(self, cache_dir="ohlcv_cache"):
        self.memory = {}
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def key(self, symbol, timeframe, start, end):
        return f"{symbol.replace('/', '')}_{timeframe}_{start}_{end}"

    def load(self, key):
        if key in self.memory:
            return self.memory[key]

        path = self.cache_dir / f"{key}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            self.memory[key] = df
            return df
        return None

    def save(self, key, df):
        self.memory[key] = df
        df.to_parquet(self.cache_dir / f"{key}.parquet")


# =====================================================
# ============== REGIME-AWARE HEDGE SYSTEM ============
# =====================================================

class RegimeAwareHedgeSystem:

    def __init__(
        self,
        exchange,
        symbols,
        groups=None,
        timeframe="1h",
        backtest_start="2024-01-01",
        backtest_end="2025-04-04",

        # Режим и корреляция
        corr_lookback=50,
        z_lookback=50,
        rebalance_interval=4,
        correlation_threshold=0.9,
        min_volume_usdt=100000,  # Минимальный объем в USDT за 24ч

        # Z-скоринг
        z_entry=2,
        z_exit=0.4,
        stop_loss_z=3.0,  # Стоп-лосс по Z-score
        take_profit_z=2.5,  # Тейк-профит по Z-score
        
        # Размер позиции
        min_size_factor=0.2,
        max_positions=10,
        position_sizing_mode='equal',  # 'equal', 'volatility', 'sharpe'
        
        # Тренд и волатильность
        trend_weight=0.5,
        volatility_lookback=14,
        max_volatility_ratio=2.0,
        
        # Финансовые параметры
        fee_rate=0.00025,
        capital=10000,
        leverage=5,  # Уменьшенное плечо для снижения риска
        max_drawdown_limit=0.1,  # Максимальная просадка 20%
        
        # Фильтры
        use_adx_filter=False,
        min_adx=14,
        use_volume_filter=False,
        use_trend_filter=True,
        
        # SMT подтверждения
        use_smt=False,
        smt_lookback=20,
        min_bos_body_atr=0.5,
        swing_n=2
    ):
        self.exchange = exchange
        self.symbols = symbols
        self.groups = groups if groups else [symbols]
        self.timeframe = timeframe
        self.backtest_start = backtest_start
        self.backtest_end = backtest_end

        # Параметры системы
        self.corr_lookback = corr_lookback
        self.z_lookback = z_lookback
        self.rebalance_interval = rebalance_interval
        self.correlation_threshold = correlation_threshold
        self.min_volume_usdt = min_volume_usdt
        
        # Z-score параметры
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.stop_loss_z = stop_loss_z
        self.take_profit_z = take_profit_z
        
        # Размер позиции
        self.min_size_factor = min_size_factor
        self.max_positions = max_positions
        self.position_sizing_mode = position_sizing_mode
        
        # Тренд и волатильность
        self.trend_weight = trend_weight
        self.volatility_lookback = volatility_lookback
        self.max_volatility_ratio = max_volatility_ratio
        
        # Финансовые параметры
        self.fee_rate = fee_rate
        self.initial_capital = capital
        self.leverage = leverage
        self.max_notional = self.initial_capital * self.leverage
        self.allocation_per_trade = self.max_notional / self.max_positions
        self.max_drawdown_limit = max_drawdown_limit
        
        # Фильтры
        self.use_adx_filter = use_adx_filter
        self.min_adx = min_adx
        self.use_volume_filter = use_volume_filter
        self.use_trend_filter = use_trend_filter
        
        # SMT параметры
        self.use_smt = use_smt
        self.smt_lookback = smt_lookback
        self.min_bos_body_atr = min_bos_body_atr
        self.swing_n = swing_n
        
        # Данные и состояние
        self.data = {}
        self.positions = []
        self.closed_trades = []
        self.position_history = []
        self.equity_history = []
        self.signals_history = []
        self.max_exposure_history = []
        
        # Статистика
        self.current_drawdown = 0
        self.max_equity = self.initial_capital
        self.total_fees = 0

    # -------------------------------------------------
    # Функции для фильтров
    # -------------------------------------------------
    
    def calculate_adx(self, symbol, t, period=14):
        """Расчет ADX для фильтрации трендовых рынков"""
        df = self.data[symbol].iloc[max(0, t-period*2):t]
        if len(df) < period:
            return 0
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
        return adx.iloc[-1] if not adx.empty else 0
    
    def calculate_atr_percent(self, symbol, t, period=14):
        """Расчет ATR в процентах от цены"""
        df = self.data[symbol].iloc[max(0, t-period*2):t]
        if len(df) < period:
            return 0
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        return (atr.iloc[-1] / df['close'].iloc[-1]) * 100 if not atr.empty else 0
    
    def calculate_volume_ratio(self, symbol, t, period=20):
        """Отношение текущего объема к среднему"""
        df = self.data[symbol].iloc[max(0, t-period*2):t]
        if len(df) < period:
            return 1
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].rolling(period).mean().iloc[-1]
        return current_volume / avg_volume if avg_volume > 0 else 1
    
    def check_filters(self, symbol, t):
        """Проверка всех фильтров для символа"""
        filters_passed = True
        reasons = []
        
        if self.use_volume_filter:
            volume_ratio = self.calculate_volume_ratio(symbol, t)
            if volume_ratio < 0.5:  # Объем ниже 50% от среднего
                filters_passed = False
                reasons.append(f"Low volume: {volume_ratio:.2f}")
        
        if self.use_adx_filter:
            adx = self.calculate_adx(symbol, t)
            if adx < self.min_adx:
                filters_passed = False
                reasons.append(f"Low ADX: {adx:.1f}")
        
        if self.use_trend_filter:
            # Проверка, что цена выше EMA 200 для лонга и ниже для шорта
            df = self.data[symbol].iloc[:t]
            if len(df) > 200:
                ema200 = talib.EMA(df['close'], timeperiod=200).iloc[-1]
                current_price = df['close'].iloc[-1]
                # Этот фильтр будет применяться в контексте направления
                pass
        
        return filters_passed, reasons

    # -------------------------------------------------

    def fetch_data(self, cache: DataCache):
        for s in self.symbols:
            key = cache.key(s, self.timeframe, self.backtest_start, self.backtest_end)
            df = cache.load(key)

            if df is None:
                since = self.exchange.parse8601(f"{self.backtest_start}T00:00:00Z")
                until = self.exchange.parse8601(f"{self.backtest_end}T00:00:00Z")
                ohlcv = []

                while since < until:
                    batch = self.exchange.fetch_ohlcv(s, self.timeframe, since=since, limit=1000)
                    if not batch:
                        break
                    ohlcv.extend(batch)
                    since = batch[-1][0] + 1

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)

                # Добавляем расчетные колонки
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(20).std()
                
                cache.save(key, df)

            self.data[s] = df

    # -------------------------------------------------

    def compute_strength_scores(self, t, symbols=None):
        if symbols is None:
            symbols = self.symbols
            
        # Предварительная фильтрация по объему
        filtered_symbols = []
        for s in symbols:
            if self.use_volume_filter:
                df = self.data[s].iloc[max(0, t-24):t]  # Последние 24 часа
                avg_volume_usdt = (df['volume'] * df['close']).mean()
                if avg_volume_usdt >= self.min_volume_usdt:
                    filtered_symbols.append(s)
            else:
                filtered_symbols.append(s)
        
        if not filtered_symbols:
            filtered_symbols = symbols
            
        window = {}
        for s in filtered_symbols:
            df = self.data[s].iloc[t - self.corr_lookback : t]
            window[s] = df["close"].pct_change()

        returns = pd.DataFrame(window).dropna()
        if returns.empty:
            return [], [], {}, pd.DataFrame()
            
        corr = returns.corr()

        scores = {}
        for s in filtered_symbols:
            avg_corr = corr[s].mean() if s in corr.columns else 0
            
            # Трендовый компонент с несколькими EMA
            df = self.data[s].iloc[:t]
            if len(df) > 50:
                ema_fast = talib.EMA(df["close"], 12).iloc[-1]
                ema_medium = talib.EMA(df["close"], 26).iloc[-1]
                ema_slow = talib.EMA(df["close"], 50).iloc[-1]
                
                # Взвешенный тренд
                trend_score = 0
                if ema_fast > ema_medium > ema_slow:
                    trend_score = 1
                elif ema_fast < ema_medium < ema_slow:
                    trend_score = -1
            else:
                trend_score = 0
                
            # Моментум компонент
            momentum = talib.ROC(df["close"], timeperiod=10).iloc[-1] if len(df) > 10 else 0
            
            scores[s] = avg_corr + self.trend_weight * trend_score + 0.1 * np.sign(momentum)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked) > 1:
            strengths = {s: 1 - i / (len(ranked) - 1) for i, (s, _) in enumerate(ranked)}
        else:
            strengths = {ranked[0][0]: 1.0}

        mid = len(ranked) // 2
        strong = [x[0] for x in ranked[:mid]]
        weak = [x[0] for x in ranked[mid:]]

        return strong, weak, strengths, corr

    # -------------------------------------------------
    # SMT подтверждения
    def is_bos_retest(self, symbol, t, direction):
        """Проверка BOS (Break of Structure) retest"""
        df = self.data[symbol].iloc[max(0, t-self.smt_lookback*2):t]
        if len(df) < self.smt_lookback:
            return False
            
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        if atr.empty or atr.iloc[-1] == 0:
            return False
            
        # Находим свинги
        high_swings = df['high'].rolling(self.swing_n, center=True).max()
        low_swings = df['low'].rolling(self.swing_n, center=True).min()
        
        if direction == 'long':
            # Bullish BOS: break previous high
            if len(df) < 3:
                return False
            bos = df['close'].iloc[-1] > high_swings.iloc[-2]
            body_size = abs(df['open'].iloc[-1] - df['close'].iloc[-1])
            if bos and body_size >= self.min_bos_body_atr * atr.iloc[-1]:
                # Retest: price touches previous high
                retest = df['low'].iloc[-1] <= high_swings.iloc[-2] * 1.01  # 1% допуск
                return bos and retest
        else:
            # Bearish BOS
            if len(df) < 3:
                return False
            bos = df['close'].iloc[-1] < low_swings.iloc[-2]
            body_size = abs(df['open'].iloc[-1] - df['close'].iloc[-1])
            if bos and body_size >= self.min_bos_body_atr * atr.iloc[-1]:
                retest = df['high'].iloc[-1] >= low_swings.iloc[-2] * 0.99  # 1% допуск
                return bos and retest
        return False

    def has_smt_divergence(self, s, w, t):
        """Проверка SMT divergence между strong и weak"""
        lookback = min(self.smt_lookback, t)
        df_s = self.data[s].iloc[t - lookback : t]
        df_w = self.data[w].iloc[t - lookback : t]
        
        if len(df_s) < 5 or len(df_w) < 5:
            return False
            
        # RSI для дивергенции
        rsi_s = talib.RSI(df_s['close'], timeperiod=14)
        rsi_w = talib.RSI(df_w['close'], timeperiod=14)
        
        # Находим экстремумы
        s_high_idx = df_s['high'].idxmax()
        w_high_idx = df_w['high'].idxmax()
        s_low_idx = df_s['low'].idxmin()
        w_low_idx = df_w['low'].idxmin()
        
        # Бычья дивергенция: s делает более высокий минимум, w - более низкий
        bullish_div = False
        if not rsi_s.empty and not rsi_w.empty:
            if s_low_idx > w_low_idx and rsi_s.loc[s_low_idx] > rsi_w.loc[w_low_idx]:
                bullish_div = True
                
        # Медвежья дивергенция: s делает более низкий максимум, w - более высокий
        bearish_div = False
        if not rsi_s.empty and not rsi_w.empty:
            if s_high_idx > w_high_idx and rsi_s.loc[s_high_idx] < rsi_w.loc[w_high_idx]:
                bearish_div = True
                
        return bullish_div or bearish_div

    # -------------------------------------------------
    # Функции для расчета размера позиции
    def calculate_position_size(self, s, w, beta, direction, current_equity):
        """Расчет размера позиции с учетом разных методов"""
        
        if self.position_sizing_mode == 'equal':
            alloc = self.allocation_per_trade
        elif self.position_sizing_mode == 'volatility':
            # Взвешивание по обратной волатильности
            vol_s = self.calculate_atr_percent(s, -1)
            vol_w = self.calculate_atr_percent(w, -1)
            total_vol = vol_s + vol_w
            weight_s = vol_w / total_vol if total_vol > 0 else 0.5
            weight_w = vol_s / total_vol if total_vol > 0 else 0.5
            alloc = self.allocation_per_trade * min(weight_s, weight_w) * 2
        elif self.position_sizing_mode == 'sharpe':
            # Упрощенное взвешивание по историческому Sharpe
            sharpe_s = self.calculate_sharpe_ratio(s, -1)
            sharpe_w = self.calculate_sharpe_ratio(w, -1)
            total_sharpe = abs(sharpe_s) + abs(sharpe_w)
            weight_s = abs(sharpe_w) / total_sharpe if total_sharpe > 0 else 0.5
            weight_w = abs(sharpe_s) / total_sharpe if total_sharpe > 0 else 0.5
            alloc = self.allocation_per_trade * min(weight_s, weight_w) * 2
        else:
            alloc = self.allocation_per_trade
            
        # Ограничение по максимальной просадке
        max_risk_per_trade = current_equity * 0.02  # 2% риска на сделку
        alloc = min(alloc, max_risk_per_trade * 5)  # 5:1 reward/risk
        
        # Нормализация
        norm = abs(1) + abs(beta)
        entry_s = self.data[s]["close"].iloc[-1]
        entry_w = self.data[w]["close"].iloc[-1]

        base_size_s = (alloc / norm) / entry_s
        base_size_w = (alloc * abs(beta) / norm) / entry_w
        
        return base_size_s, base_size_w, alloc

    def calculate_sharpe_ratio(self, symbol, t, period=20):
        """Расчет упрощенного Sharpe ratio"""
        df = self.data[symbol].iloc[max(0, t-period):t]
        if len(df) < 5:
            return 0
        returns = df['close'].pct_change().dropna()
        if len(returns) < 2:
            return 0
        return returns.mean() / returns.std() if returns.std() > 0 else 0

    # -------------------------------------------------

    def run(self):
        cum_pnl = 0.0
        self.total_fees = 0
        self.max_equity = self.initial_capital
        self.current_drawdown = 0

        timestamps = self.data[self.symbols[0]].index
        start_t = max(self.corr_lookback, self.z_lookback, self.smt_lookback, 100)

        self.positions = []
        self.position_history = []
        self.equity_history = []
        self.closed_trades = []
        self.signals_history = []

        # Инициализация режимов
        regimes = [self.compute_strength_scores(start_t, g) for g in self.groups]

        for t in range(start_t, len(timestamps)):
            current_time = timestamps[t]
            
            # Обновление режимов
            if (t - start_t) % self.rebalance_interval == 0:
                regimes = [self.compute_strength_scores(t, g) for g in self.groups]

            # Объединение strength scores
            strength_scores = {}
            for _, _, strengths, _ in regimes:
                strength_scores.update(strengths)

            # ================ УПРАВЛЕНИЕ ОТКРЫТЫМИ ПОЗИЦИЯМИ ==============
            positions_to_remove = []
            
            for i, pos in enumerate(self.positions):
                s, w = pos["s"], pos["w"]

                # Текущие цены
                current_s = self.data[s]["close"].iloc[t]
                current_w = self.data[w]["close"].iloc[t]
                
                # Расчет текущего PnL
                unrealized_pnl = pos["size_s"] * (current_s - pos["entry_s"]) - pos["dir"] * pos["size_w"] * (current_w - pos["entry_w"])
                
                # Расчет текущего Z-score
                df_s = self.data[s]["close"].iloc[:t+1]
                df_w = self.data[w]["close"].iloc[:t+1]
                spread = np.log(df_s) - pos["beta"] * np.log(df_w)
                z_series = (spread - spread.rolling(self.z_lookback).mean()) / spread.rolling(self.z_lookback).std()
                current_z = z_series.iloc[-1] if not z_series.empty and not np.isnan(z_series.iloc[-1]) else pos["z_entry"]
                
                # --- УСЛОВИЯ ВЫХОДА ---
                exit_reason = None
                
                # 1. Стоп-лосс по Z-score
                if abs(current_z) > self.stop_loss_z:
                    exit_reason = f"Z-stop: {current_z:.2f}"
                
                # 2. Тейк-профит по Z-score
                elif abs(current_z) < self.take_profit_z and abs(pos["z_entry"]) > abs(current_z):
                    exit_reason = f"Z-take: {current_z:.2f}"
                
                # 3. Корреляция ухудшилась
                ret_s = self.data[s]["close"].iloc[t - self.corr_lookback:t].pct_change()
                ret_w = self.data[w]["close"].iloc[t - self.corr_lookback:t].pct_change()
                if len(ret_s) > 10 and len(ret_w) > 10:
                    corr_now = ret_s.corr(ret_w)
                    corr_factor = max(0.0, min(1.0, corr_now / pos["corr_entry"]))
                    if corr_factor < self.min_size_factor:
                        exit_reason = f"Corr decay: {corr_now:.2f}"
                
                # 4. Strength decay
                strength_factor = max(0.0, min(1.0, strength_scores.get(s, 0.5) / pos["strength_entry"]))
                if strength_factor < self.min_size_factor:
                    exit_reason = f"Strength decay: {strength_factor:.2f}"
                
                # 5. Stop-loss по PnL (5% от капитала)
                current_equity = self.initial_capital + cum_pnl + sum(
                    p["size_s"] * (self.data[p["s"]]["close"].iloc[t] - p["entry_s"]) - 
                    p["dir"] * p["size_w"] * (self.data[p["w"]]["close"].iloc[t] - p["entry_w"])
                    for p in self.positions
                )
                pnl_percent = unrealized_pnl / current_equity if current_equity > 0 else 0
                if pnl_percent < -0.05:  # -5%
                    exit_reason = f"PnL stop: {pnl_percent:.1%}"
                
                # Выход по условию
                if exit_reason:
                    # Закрываем всю позицию
                    close_fees = (pos["size_s"] * current_s + pos["size_w"] * current_w) * self.fee_rate
                    pnl = unrealized_pnl - close_fees
                    
                    cum_pnl += pnl
                    self.total_fees += close_fees
                    
                    self.closed_trades.append({
                        "pair": f"{s}-{w}",
                        "entry_time": pos["entry_time"],
                        "exit_time": current_time,
                        "pnl": pnl,
                        "pnl_percent": (pnl / (abs(pos["size_s"] * pos["entry_s"]) + abs(pos["size_w"] * pos["entry_w"]))) * 100,
                        "duration_hours": (t - pos["entry_t"]),
                        "exit_reason": exit_reason,
                        "direction": "LONG" if pos["dir"] > 0 else "SHORT",
                        "entry_z": pos["z_entry"],
                        "exit_z": current_z
                    })
                    
                    positions_to_remove.append(i)
                    continue
                
                # Частичное закрытие при ухудшении условий
                new_size_factor = corr_factor * strength_factor if 'corr_factor' in locals() else pos["size_factor"]
                new_size_factor = max(self.min_size_factor, min(1.0, new_size_factor))
                
                if new_size_factor < pos["size_factor"]:
                    delta_factor = pos["size_factor"] - new_size_factor
                    delta_size_s = pos["base_size_s"] * delta_factor
                    delta_size_w = pos["base_size_w"] * delta_factor
                    
                    partial_pnl = delta_size_s * (current_s - pos["entry_s"]) - pos["dir"] * delta_size_w * (current_w - pos["entry_w"])
                    partial_fees = (delta_size_s * current_s + delta_size_w * current_w) * self.fee_rate
                    
                    cum_pnl += partial_pnl - partial_fees
                    self.total_fees += partial_fees
                    
                    # Обновляем размер позиции
                    pos["size_s"] -= delta_size_s
                    pos["size_w"] -= delta_size_w
                    pos["size_factor"] = new_size_factor
            
            # Удаляем закрытые позиции
            for idx in sorted(positions_to_remove, reverse=True):
                self.positions.pop(idx)

            # ================= ОТКРЫТИЕ НОВЫХ ПОЗИЦИЙ ==================
            current_exposure = sum(
                abs(p["size_s"] * self.data[p["s"]]["close"].iloc[t]) + abs(p["size_w"] * self.data[p["w"]]["close"].iloc[t])
                for p in self.positions
            )
            
            # Ограничение по максимальному количеству позиций
            if len(self.positions) >= self.max_positions:
                continue
                
            # Ограничение по просадке
            current_equity = self.initial_capital + cum_pnl + sum(
                p["size_s"] * (self.data[p["s"]]["close"].iloc[t] - p["entry_s"]) - 
                p["dir"] * p["size_w"] * (self.data[p["w"]]["close"].iloc[t] - p["entry_w"])
                for p in self.positions
            )
            
            self.max_equity = max(self.max_equity, current_equity)
            self.current_drawdown = (self.max_equity - current_equity) / self.max_equity if self.max_equity > 0 else 0
            
            if self.current_drawdown > self.max_drawdown_limit:
                continue  # Пропускаем открытие новых позиций при большой просадке
            
            for strong, weak, strength_score, corr_matrix in regimes:
                for s in strong[:3]:  # Берем только топ-3 сильных
                    for w in weak[:3]:  # И топ-3 слабых
                        
                        # Проверка фильтров
                        s_filter_passed, s_reasons = self.check_filters(s, t)
                        w_filter_passed, w_reasons = self.check_filters(w, t)
                        
                        if not (s_filter_passed and w_filter_passed):
                            continue
                        
                        # Фильтр по корреляции
                        if s not in corr_matrix.columns or w not in corr_matrix.columns:
                            continue
                        if corr_matrix.loc[s, w] < self.correlation_threshold:
                            continue
                        
                        # Одна позиция на пару
                        if any(p["s"] == s and p["w"] == w for p in self.positions):
                            continue
                        
                        # Расчет бета и Z-score
                        df_s = self.data[s]["close"].iloc[:t+1]
                        df_w = self.data[w]["close"].iloc[:t+1]
                        
                        ret_s = np.log(df_s).diff().dropna()
                        ret_w = np.log(df_w).diff().dropna()
                        
                        # Выравниваем длины
                        common_idx = ret_s.index.intersection(ret_w.index)
                        if len(common_idx) < 20:
                            continue
                            
                        ret_s_aligned = ret_s.loc[common_idx]
                        ret_w_aligned = ret_w.loc[common_idx]
                        
                        beta = np.cov(ret_s_aligned, ret_w_aligned)[0, 1] / np.var(ret_w_aligned)
                        if np.isnan(beta) or abs(beta) > 5:
                            continue
                        
                        # Z-score
                        spread = np.log(df_s) - beta * np.log(df_w)
                        z_series = (spread - spread.rolling(self.z_lookback).mean()) / spread.rolling(self.z_lookback).std()
                        z_now = z_series.iloc[-1] if not z_series.empty else 0
                        
                        if np.isnan(z_now) or abs(z_now) < self.z_entry:
                            continue
                        
                        # SMT подтверждение
                        smt_ok = True
                        if self.use_smt:
                            smt_div = self.has_smt_divergence(s, w, t)
                            direction = 'long' if z_now > 0 else 'short'
                            bos_retest_s = self.is_bos_retest(s, t, direction)
                            bos_retest_w = self.is_bos_retest(w, t, 'short' if direction == 'long' else 'long')
                            smt_ok = smt_div or (bos_retest_s and bos_retest_w)
                        
                        if not smt_ok:
                            continue
                        
                        # Расчет размера позиции
                        base_size_s, base_size_w, alloc = self.calculate_position_size(
                            s, w, beta, np.sign(z_now), current_equity
                        )
                        
                        # Проверка лимита экспозиции
                        proposed_exposure = abs(base_size_s * df_s.iloc[-1]) + abs(base_size_w * df_w.iloc[-1])
                        if current_exposure + proposed_exposure > self.max_notional:
                            continue
                        
                        # Открытие позиции
                        entry_fees = (base_size_s * df_s.iloc[-1] + base_size_w * df_w.iloc[-1]) * self.fee_rate
                        cum_pnl -= entry_fees
                        self.total_fees += entry_fees
                        
                        self.positions.append({
                            "s": s,
                            "w": w,
                            "dir": np.sign(z_now),
                            "entry_s": df_s.iloc[-1],
                            "entry_w": df_w.iloc[-1],
                            "base_size_s": base_size_s,
                            "base_size_w": base_size_w,
                            "size_s": base_size_s,
                            "size_w": base_size_w,
                            "size_factor": 1.0,
                            "corr_entry": corr_matrix.loc[s, w],
                            "strength_entry": strength_score.get(s, 0.5),
                            "z_entry": z_now,
                            "beta": beta,
                            "entry_t": t,
                            "entry_time": current_time,
                            "allocation": alloc
                        })
                        
                        # Логирование сигнала
                        self.signals_history.append({
                            "time": current_time,
                            "pair": f"{s}-{w}",
                            "direction": "LONG" if np.sign(z_now) > 0 else "SHORT",
                            "z_score": z_now,
                            "correlation": corr_matrix.loc[s, w],
                            "strength": strength_score.get(s, 0.5)
                        })
                        
                        current_exposure += proposed_exposure

            # ================= ЛОГИРОВАНИЕ ======================
            exposure = sum(
                abs(p["size_s"] * self.data[p["s"]]["close"].iloc[t]) + abs(p["size_w"] * self.data[p["w"]]["close"].iloc[t])
                for p in self.positions
            )

            unrealized = sum(
                p["size_s"] * (self.data[p["s"]]["close"].iloc[t] - p["entry_s"]) - 
                p["dir"] * p["size_w"] * (self.data[p["w"]]["close"].iloc[t] - p["entry_w"])
                for p in self.positions
            )

            equity = self.initial_capital + cum_pnl + unrealized

            self.position_history.append({
                "time": current_time,
                "exposure": exposure,
                "positions": len(self.positions),
                "unrealized_pnl": unrealized,
                "realized_pnl": cum_pnl
            })

            self.equity_history.append({
                "time": current_time,
                "equity": equity,
                "drawdown": self.current_drawdown * 100,
                "max_equity": self.max_equity
            })
            
            self.max_exposure_history.append({
                "time": current_time,
                "max_exposure_used": exposure / self.max_notional * 100
            })

        # Закрытие оставшихся позиций в конце
        t = len(timestamps) - 1
        for pos in self.positions[:]:
            s, w = pos["s"], pos["w"]
            exit_s = self.data[s]["close"].iloc[t]
            exit_w = self.data[w]["close"].iloc[t]

            pnl = pos["size_s"] * (exit_s - pos["entry_s"]) - pos["dir"] * pos["size_w"] * (exit_w - pos["entry_w"])
            close_fees = (pos["size_s"] * exit_s + pos["size_w"] * exit_w) * self.fee_rate
            pnl -= close_fees
            self.total_fees += close_fees

            cum_pnl += pnl

            self.closed_trades.append({
                "pair": f"{s}-{w}",
                "entry_time": pos["entry_time"],
                "exit_time": timestamps[t],
                "pnl": pnl,
                "pnl_percent": (pnl / (abs(pos["size_s"] * pos["entry_s"]) + abs(pos["size_w"] * pos["entry_w"]))) * 100,
                "duration_hours": (t - pos["entry_t"]),
                "exit_reason": "End of backtest",
                "direction": "LONG" if pos["dir"] > 0 else "SHORT",
                "entry_z": pos["z_entry"],
                "exit_z": 0
            })

            self.positions.remove(pos)

        return cum_pnl

    # -------------------------------------------------
    # Улучшенная визуализация
    def plot_results(self, figsize=(16, 12)):
        """Улучшенная визуализация результатов"""
        
        if not self.equity_history:
            print("Нет данных для визуализации")
            return
            
        df_equity = pd.DataFrame(self.equity_history).set_index("time")
        df_positions = pd.DataFrame(self.position_history).set_index("time")
        df_exposure = pd.DataFrame(self.max_exposure_history).set_index("time")
        
        fig = plt.figure(figsize=figsize)
        
        # 1. Кривая эквити и просадка
        ax1 = plt.subplot(4, 2, 1)
        ax1.plot(df_equity.index, df_equity["equity"], label="Equity", linewidth=2, color='blue')
        ax1.fill_between(df_equity.index, df_equity["equity"], self.initial_capital, 
                        where=(df_equity["equity"] > self.initial_capital), 
                        alpha=0.3, color='green', label='Profit')
        ax1.fill_between(df_equity.index, df_equity["equity"], self.initial_capital,
                        where=(df_equity["equity"] < self.initial_capital),
                        alpha=0.3, color='red', label='Loss')
        ax1.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.5)
        ax1.set_title(f"Equity Curve (Final: ${df_equity['equity'].iloc[-1]:.2f})", fontsize=12, fontweight='bold')
        ax1.set_ylabel("Equity ($)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Просадка
        ax2 = plt.subplot(4, 2, 2)
        ax2.fill_between(df_equity.index, 0, df_equity["drawdown"], 
                        where=(df_equity["drawdown"] < 0), color='red', alpha=0.5)
        ax2.plot(df_equity.index, df_equity["drawdown"], color='darkred', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_title(f"Drawdown (Max: {df_equity['drawdown'].min():.1f}%)", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.3)
        
        # 3. Экспозиция
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(df_positions.index, df_positions["exposure"], label="Exposure", color='purple', linewidth=1.5)
        ax3.axhline(y=self.max_notional, color='red', linestyle='--', alpha=0.5, label='Max Exposure')
        ax3.set_title(f"Position Exposure (Avg: ${df_positions['exposure'].mean():.0f})", fontsize=12, fontweight='bold')
        ax3.set_ylabel("Exposure ($)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Количество позиций
        ax4 = plt.subplot(4, 2, 4)
        ax4.bar(df_positions.index, df_positions["positions"], 
               color=['green' if x > 0 else 'gray' for x in df_positions["positions"]], 
               alpha=0.6, width=0.02)
        ax4.set_title(f"Open Positions (Max: {df_positions['positions'].max()})", fontsize=12, fontweight='bold')
        ax4.set_ylabel("Count")
        ax4.set_ylim(0, self.max_positions + 1)
        ax4.grid(True, alpha=0.3)
        
        # 5. PnL распределение сделок
        ax5 = plt.subplot(4, 2, 5)
        if self.closed_trades:
            trade_pnls = [t["pnl"] for t in self.closed_trades]
            win_trades = [p for p in trade_pnls if p > 0]
            loss_trades = [p for p in trade_pnls if p < 0]
            
            colors = ['green' if p > 0 else 'red' for p in trade_pnls]
            ax5.hist([win_trades, loss_trades], bins=20, color=['green', 'red'], 
                    alpha=0.7, label=['Win', 'Loss'], stacked=True)
            ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax5.set_title(f"Trade PnL Distribution (Win Rate: {len(win_trades)/len(trade_pnls)*100:.1f}%)", 
                         fontsize=12, fontweight='bold')
            ax5.set_xlabel("PnL ($)")
            ax5.set_ylabel("Frequency")
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Продолжительность сделок
        ax6 = plt.subplot(4, 2, 6)
        if self.closed_trades:
            durations = [t["duration_hours"] / 24 for t in self.closed_trades]  # В днях
            ax6.hist(durations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
            ax6.axvline(x=np.mean(durations), color='red', linestyle='--', 
                       label=f'Avg: {np.mean(durations):.1f}d')
            ax6.set_title("Trade Duration Distribution", fontsize=12, fontweight='bold')
            ax6.set_xlabel("Duration (Days)")
            ax6.set_ylabel("Frequency")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Причины выхода
        ax7 = plt.subplot(4, 2, 7)
        if self.closed_trades:
            exit_reasons = {}
            for trade in self.closed_trades:
                reason = trade.get("exit_reason", "Unknown")
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            if exit_reasons:
                reasons = list(exit_reasons.keys())
                counts = list(exit_reasons.values())
                colors = plt.cm.Set3(np.arange(len(reasons)) / len(reasons))
                
                ax7.pie(counts, labels=reasons, autopct='%1.1f%%', colors=colors, startangle=90)
                ax7.set_title("Exit Reasons Distribution", fontsize=12, fontweight='bold')
        
        # 8. Ежемесячная доходность
        ax8 = plt.subplot(4, 2, 8)
        if len(df_equity) > 30:
            monthly_returns = df_equity['equity'].resample('M').last().pct_change() * 100
            colors = ['green' if x > 0 else 'red' for x in monthly_returns]
            
            bars = ax8.bar(range(len(monthly_returns)), monthly_returns, color=colors, alpha=0.7)
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax8.set_title("Monthly Returns (%)", fontsize=12, fontweight='bold')
            ax8.set_ylabel("Return %")
            ax8.set_xticks(range(len(monthly_returns)))
            ax8.set_xticklabels([d.strftime('%b %Y') for d in monthly_returns.index], rotation=45)
            ax8.grid(True, alpha=0.3, axis='y')
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                        fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительная статистика
        self.plot_additional_stats()

    def plot_additional_stats(self):
        """Дополнительная статистика и графики"""
        if not self.closed_trades:
            return
            
        df_trades = pd.DataFrame(self.closed_trades)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Кумулятивный PnL по сделкам
        ax1 = axes[0, 0]
        df_trades['cumulative_pnl'] = df_trades['pnl'].cumsum()
        df_trades['trade_number'] = range(1, len(df_trades) + 1)
        
        ax1.plot(df_trades['trade_number'], df_trades['cumulative_pnl'], 
                marker='o', markersize=4, linewidth=2, color='blue')
        ax1.fill_between(df_trades['trade_number'], 0, df_trades['cumulative_pnl'],
                        where=(df_trades['cumulative_pnl'] > 0), color='green', alpha=0.3)
        ax1.fill_between(df_trades['trade_number'], 0, df_trades['cumulative_pnl'],
                        where=(df_trades['cumulative_pnl'] < 0), color='red', alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title("Cumulative Trade PnL", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Trade Number")
        ax1.set_ylabel("Cumulative PnL ($)")
        ax1.grid(True, alpha=0.3)
        
        # 2. PnL vs Duration
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df_trades['duration_hours']/24, df_trades['pnl_percent'],
                            c=df_trades['pnl_percent'], cmap='RdYlGn', alpha=0.6, s=50)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.axvline(x=df_trades['duration_hours'].mean()/24, color='blue', 
                   linestyle='--', alpha=0.5, label=f'Avg: {df_trades["duration_hours"].mean()/24:.1f}d')
        ax2.set_title("PnL % vs Trade Duration", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Duration (Days)")
        ax2.set_ylabel("PnL %")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax2, label='PnL %')
        
        # 3. Распределение по парам
        ax3 = axes[1, 0]
        pair_pnl = df_trades.groupby('pair')['pnl'].sum().sort_values()
        colors = ['green' if x > 0 else 'red' for x in pair_pnl]
        bars = ax3.barh(range(len(pair_pnl)), pair_pnl, color=colors, alpha=0.7)
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_yticks(range(len(pair_pnl)))
        ax3.set_yticklabels(pair_pnl.index)
        ax3.set_title("Total PnL by Pair", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Total PnL ($)")
        ax3.grid(True, alpha=0.3, axis='x')
        
        # 4. Время дня для входов
        ax4 = axes[1, 1]
        if 'entry_time' in df_trades.columns:
            entry_hours = pd.to_datetime(df_trades['entry_time']).dt.hour
            hour_counts = entry_hours.value_counts().sort_index()
            ax4.bar(hour_counts.index, hour_counts.values, color='skyblue', alpha=0.7)
            ax4.set_xlabel("Hour of Day (UTC)")
            ax4.set_ylabel("Number of Trades")
            ax4.set_title("Trade Entry Time Distribution", fontsize=12, fontweight='bold')
            ax4.set_xticks(range(0, 24, 3))
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def compute_metrics(self, return_dict=False):
        """Расчет метрик с улучшенной статистикой"""
        if not self.equity_history:
            if return_dict:
                return {}
            else:
                print("No data for metrics.")
                return

        df_equity = pd.DataFrame(self.equity_history).set_index("time")
        df_positions = pd.DataFrame(self.position_history).set_index("time")
        
        # Базовые метрики
        total_pnl = df_equity["equity"].iloc[-1] - self.initial_capital
        total_return = (total_pnl / self.initial_capital) * 100
        
        # Trade stats
        num_trades = 0
        win_rate = 0
        avg_pnl = 0
        avg_win = 0
        avg_loss = 0
        avg_duration = 0
        profit_factor = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        if self.closed_trades:
            pnls = [t["pnl"] for t in self.closed_trades]
            num_trades = len(pnls)
            win_trades = sum(p > 0 for p in pnls)
            loss_trades = sum(p < 0 for p in pnls)
            win_rate = win_trades / num_trades if num_trades > 0 else 0
            
            avg_pnl = np.mean(pnls) if num_trades > 0 else 0
            avg_win = np.mean([p for p in pnls if p > 0]) if win_trades > 0 else 0
            avg_loss = np.mean([p for p in pnls if p < 0]) if loss_trades > 0 else 0
            avg_duration = np.mean([t["duration_hours"] for t in self.closed_trades]) if num_trades > 0 else 0
            
            # Profit Factor
            gross_profit = sum([p for p in pnls if p > 0])
            gross_loss = abs(sum([p for p in pnls if p < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Consecutive wins/losses
            consecutive = 0
            current_sign = 0
            for pnl in pnls:
                sign = 1 if pnl > 0 else -1 if pnl < 0 else 0
                if sign == current_sign:
                    consecutive += 1
                else:
                    consecutive = 1
                    current_sign = sign
                
                if sign == 1:
                    max_consecutive_wins = max(max_consecutive_wins, consecutive)
                elif sign == -1:
                    max_consecutive_losses = max(max_consecutive_losses, consecutive)
            
            if not return_dict:
                # Детальная таблица сделок
                trade_df = pd.DataFrame(self.closed_trades)
                print("\n" + "="*80)
                print("CLOSED TRADES DETAIL:")
                print("="*80)
                if len(trade_df) > 0:
                    print(trade_df[["pair", "direction", "entry_time", "exit_time", 
                                   "pnl", "pnl_percent", "duration_hours", "exit_reason"]].to_string())

        # Drawdown analysis
        equity = df_equity["equity"]
        peak = equity.cummax()
        drawdown = (equity - peak) / peak
        max_dd = drawdown.min() * 100 if not drawdown.empty else 0
        avg_dd = drawdown[drawdown < 0].mean() * 100 if not drawdown[drawdown < 0].empty else 0
        recovery_factor = abs(total_return / max_dd) if max_dd < 0 else np.inf

        # Sharpe ratio
        sharpe = 0
        returns = equity.pct_change().dropna()
        if len(returns) > 1:
            mean_ret = returns.mean()
            std_ret = returns.std()
            sharpe = mean_ret / std_ret * np.sqrt(8760) if std_ret > 0 else 0
        
        # Sortino ratio (только отрицательная волатильность)
        sortino = 0
        if len(returns) > 1:
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() if len(negative_returns) > 1 else 0
            sortino = mean_ret / downside_std * np.sqrt(8760) if downside_std > 0 else 0
        
        # Calmar ratio
        calmar = total_return / abs(max_dd) if max_dd < 0 else np.inf
        
        # Exposure statistics
        avg_exposure = df_positions["exposure"].mean()
        max_exposure = df_positions["exposure"].max()
        exposure_ratio = avg_exposure / self.max_notional * 100
        
        # Monthly returns
        monthly_returns = []
        if len(df_equity) > 30:
            monthly = df_equity["equity"].resample('M').last()
            monthly_returns = monthly.pct_change().dropna() * 100
            monthly_win_rate = (monthly_returns > 0).sum() / len(monthly_returns) * 100 if len(monthly_returns) > 0 else 0
        else:
            monthly_win_rate = 0
        
        # Buy & Hold comparison
        timestamps = self.data[self.symbols[0]].index
        start_t = max(self.corr_lookback, self.z_lookback)
        bh_start = self.data["BTC/USDT"]["close"].iloc[start_t] if "BTC/USDT" in self.data else self.data[self.symbols[0]]["close"].iloc[start_t]
        bh_end = self.data["BTC/USDT"]["close"].iloc[-1] if "BTC/USDT" in self.data else self.data[self.symbols[0]]["close"].iloc[-1]
        bh_return = ((bh_end / bh_start) - 1) * 100
        bh_pnl = self.initial_capital * (bh_end / bh_start - 1)
        
        # Alpha vs BTC
        alpha = total_return - bh_return
        
        # Risk metrics
        var_95 = np.percentile(returns.dropna(), 5) * 100 if len(returns) > 10 else 0
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100 if len(returns) > 10 else 0
        
        metrics_dict = {
            'initial_capital': self.initial_capital,
            'final_equity': df_equity["equity"].iloc[-1],
            'net_pnl': total_pnl,
            'total_return_percent': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate * 100,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_trade_duration_hours': avg_duration,
            'max_drawdown_percent': max_dd,
            'avg_drawdown_percent': avg_dd,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'recovery_factor': recovery_factor,
            'avg_exposure_usd': avg_exposure,
            'max_exposure_usd': max_exposure,
            'exposure_ratio_percent': exposure_ratio,
            'monthly_win_rate': monthly_win_rate,
            'bh_return_percent': bh_return,
            'bh_pnl': bh_pnl,
            'alpha_vs_btc': alpha,
            'var_95_percent': var_95,
            'cvar_95_percent': cvar_95,
            'total_fees': self.total_fees,
            'fees_percent_of_pnl': (self.total_fees / total_pnl * 100) if total_pnl != 0 else 0
        }
        
        if return_dict:
            return metrics_dict
        else:
            print("\n" + "="*80)
            print("PERFORMANCE METRICS")
            print("="*80)
            
            print(f"\n【BASIC METRICS】")
            print(f"Initial Capital: ${self.initial_capital:,.2f}")
            print(f"Final Equity: ${df_equity['equity'].iloc[-1]:,.2f}")
            print(f"Net PnL: ${total_pnl:,.2f}")
            print(f"Total Return: {total_return:.2f}%")
            print(f"BTC Buy & Hold Return: {bh_return:.2f}%")
            print(f"Alpha vs BTC: {alpha:.2f}%")
            
            print(f"\n【TRADE STATISTICS】")
            print(f"Number of Trades: {num_trades}")
            print(f"Win Rate: {win_rate*100:.1f}%")
            print(f"Profit Factor: {profit_factor:.2f}")
            print(f"Average PnL per Trade: ${avg_pnl:.2f}")
            print(f"Average Win: ${avg_win:.2f}")
            print(f"Average Loss: ${avg_loss:.2f}")
            print(f"Max Consecutive Wins: {max_consecutive_wins}")
            print(f"Max Consecutive Losses: {max_consecutive_losses}")
            print(f"Average Trade Duration: {avg_duration/24:.1f} days")
            
            print(f"\n【RISK METRICS】")
            print(f"Max Drawdown: {max_dd:.2f}%")
            print(f"Average Drawdown: {avg_dd:.2f}%")
            print(f"Sharpe Ratio (annualized): {sharpe:.2f}")
            print(f"Sortino Ratio (annualized): {sortino:.2f}")
            print(f"Calmar Ratio: {calmar:.2f}")
            print(f"Recovery Factor: {recovery_factor:.2f}")
            print(f"VaR 95%: {var_95:.2f}%")
            print(f"CVaR 95%: {cvar_95:.2f}%")
            
            print(f"\n【EXPOSURE & FEES】")
            print(f"Average Exposure: ${avg_exposure:,.2f}")
            print(f"Max Exposure: ${max_exposure:,.2f}")
            print(f"Exposure Ratio: {exposure_ratio:.1f}% of max")
            print(f"Total Fees: ${self.total_fees:,.2f}")
            print(f"Fees % of PnL: {metrics_dict['fees_percent_of_pnl']:.1f}%")
            
            if len(monthly_returns) > 0:
                print(f"\n【MONTHLY PERFORMANCE】")
                print(f"Monthly Win Rate: {monthly_win_rate:.1f}%")
                print(f"Best Month: {monthly_returns.max():.1f}%")
                print(f"Worst Month: {monthly_returns.min():.1f}%")
                print(f"Average Monthly Return: {monthly_returns.mean():.1f}%")
            
            return metrics_dict

# =====================================================
# ====================== MAIN =========================
# =====================================================

def get_extended_symbols_and_groups():
    """Расширенный список активов для хеджирования"""
    # Основные группы активов
    large_caps = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT",
        "XRP/USDT", "ADA/USDT", "AVAX/USDT", "DOT/USDT"
    ]
    
    mid_caps = [
        "LINK/USDT", "UNI/USDT", "ATOM/USDT",
        "ETC/USDT", "FIL/USDT", "NEAR/USDT", "ALGO/USDT"
    ]
    
    small_caps = [
        "AAVE/USDT", "APE/USDT", "AXS/USDT", "SAND/USDT",
        "MANA/USDT", "CHZ/USDT", "ENJ/USDT", "GALA/USDT"
    ]
    
    # Группы по секторам
    layer1 = ["ETH/USDT", "SOL/USDT", "AVAX/USDT", "ATOM/USDT", "NEAR/USDT", "ALGO/USDT"]
    defi = ["UNI/USDT", "AAVE/USDT", "LINK/USDT"]
    gaming = ["AXS/USDT", "SAND/USDT", "MANA/USDT", "ENJ/USDT", "GALA/USDT"]
    storage = ["FIL/USDT", "AR/USDT"]
    
    all_symbols = list(set(large_caps + mid_caps + small_caps + layer1 + defi + gaming + storage))
    
    # Группировка для хеджирования (сильные против слабых внутри групп)
    groups = [
        large_caps[:4],  # Топ-4 large caps
        mid_caps[:4],    # Топ-4 mid caps
        small_caps[:4],  # Топ-4 small caps
        layer1,          # L1 решения
        defi,            # DeFi токены
        gaming           # Gaming/metaverse
    ]
    
    return all_symbols, groups


def run_optimization():
    """Запуск оптимизации параметров"""
    exchange = ccxt.binace({
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    
    cache = DataCache()
    all_symbols, groups = get_extended_symbols_and_groups()
    
    # Сначала загружаем данные
    print("Загрузка данных...")
    dummy_system = RegimeAwareHedgeSystem(
        exchange=exchange,
        symbols=all_symbols,
        groups=groups,
        timeframe="1h",
        backtest_start="2024-01-01",
        backtest_end="2026-01-01"
    )
    dummy_system.fetch_data(cache)
    shared_data = dummy_system.data
    
    # Ограниченная оптимизация для демонстрации
    param_grid = {
        'corr_lookback': [150, 200],
        'z_lookback': [30, 75, 100],
        'z_entry': [1.7, 1.8, 2],
        'z_exit': [0.3, 0.4, 0.2],
        'stop_loss_z': [1.5, 2.5, 3.0],
        'take_profit_z': [2.0, 2.5, 5],
        'max_positions': [8],
        'leverage': [10],
        'position_sizing_mode': ['equal', 'volatility', 'sharpe']
    }
    
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*(param_grid[key] for key in keys)))
    
    results = []
    print(f"\nЗапуск оптимизации ({len(combinations)} комбинаций)...")
    
    for i, comb in enumerate(combinations[:10]):  # Ограничиваем для демо
        params = dict(zip(keys, comb))
        print(f"Тест {i+1}/{min(10, len(combinations))}: {params}")
        
        system = RegimeAwareHedgeSystem(
            exchange=exchange,
            symbols=all_symbols[:15],  # Ограничиваем для скорости
            groups=[groups[0], groups[1]],  # Первые 2 группы
            timeframe="1h",
            backtest_start="2024-06-01",
            backtest_end="2024-12-01",
            fee_rate=0.0005,
            capital=10000,
            **params
        )
        system.data = {k: v for k, v in shared_data.items() if k in all_symbols[:15]}
        _ = system.run()
        metrics = system.compute_metrics(return_dict=True)
        
        if metrics:
            metrics['params'] = params
            # Score function for optimization
            score = (
                metrics['sharpe_ratio'] * 0.3 +
                metrics['profit_factor'] * 0.2 +
                (100 - abs(metrics['max_drawdown_percent'])) * 0.2 +
                metrics['win_rate'] * 0.1 +
                metrics['total_return_percent'] * 0.1 +
                (metrics['num_trades'] / 100) * 0.1  # Нормализуем количество сделок
            )
            metrics['optimization_score'] = score
            results.append(metrics)
    
    if results:
        df_results = pd.DataFrame(results)
        df_results.sort_values(by='optimization_score', ascending=False, inplace=True)
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS (Top 5):")
        print("="*80)
        
        # Выводим топ-5 результатов
        top_results = df_results.head(5)
        for idx, row in top_results.iterrows():
            print(f"\n#{idx+1} Score: {row['optimization_score']:.2f}")
            print(f"Sharpe: {row['sharpe_ratio']:.2f}, DD: {row['max_drawdown_percent']:.1f}%, "
                  f"Win Rate: {row['win_rate']:.1f}%, Trades: {row['num_trades']}")
            print(f"Params: {row['params']}")
        
        return df_results
    return pd.DataFrame()


def run_final_backtest(best_params=None):
    """Запуск финального бэктеста с лучшими параметрами"""
    exchange = ccxt.binance({  # Исправлена опечатка: было "binace"
        "enableRateLimit": True,
        "options": {"defaultType": "future"}
    })
    
    cache = DataCache()
    all_symbols, groups = get_extended_symbols_and_groups()
    
    # Загружаем данные
    print("Загрузка данных для финального теста...")
    dummy_system = RegimeAwareHedgeSystem(
        exchange=exchange,
        symbols=all_symbols,
        groups=groups,
        timeframe="1h",
        backtest_start="2023-06-01",  # Более длинный период
        backtest_end="2024-12-01"
    )
    dummy_system.fetch_data(cache)
    shared_data = dummy_system.data
    
    # Параметры для низкорисковой хедж-системы
    if best_params is None:
        best_params = {
            'corr_lookback': 200,
            'z_lookback': 100,
            'rebalance_interval': 24,
            'correlation_threshold': 0.6,
            'z_entry': 1.8,
            'z_exit': 0.4,
            'stop_loss_z': 3.0,
            'take_profit_z': 2.5,
            'min_size_factor': 0.3,
            'max_positions': 5,
            'position_sizing_mode': 'equal',
            'trend_weight': 0.2,
            'leverage': 3,
            'max_drawdown_limit': 0.2,
            'use_adx_filter': True,
            'min_adx': 20,
            'use_volume_filter': True,
            'use_trend_filter': True,
            'use_smt': True,
            'smt_lookback': 20
        }
    
    print("\n" + "="*80)
    print("FINAL BACKTEST - LOW RISK HEDGE SYSTEM")
    print("="*80)
    
    system = RegimeAwareHedgeSystem(
        exchange=exchange,
        symbols=all_symbols,
        groups=groups,
        timeframe="1h",
        backtest_start="2024-01-01",
        backtest_end="2024-12-01",
        fee_rate=0.0005,
        capital=10000,
        **best_params
    )
    system.data = shared_data
    
    print("Запуск системы...")
    final_pnl = system.run()
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    metrics = system.compute_metrics(return_dict=False)
    
    # Детальный анализ
    if system.closed_trades:
        df_trades = pd.DataFrame(system.closed_trades)
        
        print("\n【TRADE ANALYSIS】")
        print(f"Total trades: {len(df_trades)}")
        
        # Анализ по парам
        pair_stats = df_trades.groupby('pair').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_percent': 'mean',
            'duration_hours': 'mean'
        }).round(2)
        
        print("\nPerformance by pair:")
        print(pair_stats)
        
        # Анализ по направлению - ИСПРАВЛЕННЫЙ КОД
        if 'direction' in df_trades.columns and 'pnl' in df_trades.columns:
            direction_stats = df_trades.groupby('direction').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            
            # Вычисляем win rate отдельно
            direction_win_rates = []
            for direction in df_trades['direction'].unique():
                trades_direction = df_trades[df_trades['direction'] == direction]
                win_rate = (trades_direction['pnl'] > 0).sum() / len(trades_direction) * 100
                direction_win_rates.append(win_rate)
            
            # Добавляем win rate к статистике
            direction_stats['win_rate'] = direction_win_rates
            
            print("\nPerformance by direction:")
            print(direction_stats)
        else:
            print("\nНе удалось проанализировать по направлениям: отсутствуют необходимые колонки")
        
        # Анализ по причине выхода
        if 'exit_reason' in df_trades.columns:
            exit_stats = df_trades.groupby('exit_reason').agg({
                'pnl': ['count', 'sum', 'mean']
            }).round(2)
            print("\nPerformance by exit reason:")
            print(exit_stats)
        else:
            print("\nНе удалось проанализировать по причинам выхода: отсутствует колонка 'exit_reason'")
    
    # Визуализация
    print("\nГенерация графиков...")
    system.plot_results()
    
    return system


if __name__ == "__main__":
    
    # Вариант 1: Запуск оптимизации (занимает время)
    # results_df = run_optimization()
    
    # Вариант 2: Прямой запуск финального теста
    system = run_final_backtest()
    
    # Экспорт результатов
    if system.closed_trades:
        # Сохраняем сделки в CSV
        trades_df = pd.DataFrame(system.closed_trades)
        trades_df.to_csv('hedge_system_trades.csv', index=False)
        print(f"\nСделки сохранены в 'hedge_system_trades.csv' ({len(trades_df)} trades)")
        
        # Сохраняем историю эквити
        equity_df = pd.DataFrame(system.equity_history)
        equity_df.to_csv('hedge_system_equity.csv', index=False)
        print(f"История эквити сохранена в 'hedge_system_equity.csv'")
        
        # Сохраняем метрики
        metrics = system.compute_metrics(return_dict=True)
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv('hedge_system_metrics.csv', index=False)
        print(f"Метрики сохранены в 'hedge_system_metrics.csv'")