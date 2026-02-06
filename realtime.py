import time
import pandas as pd
import numpy as np
from datetime import datetime, UTC
import logging
from pathlib import Path

# Твой клиент BingX (из прикреплённого файла)
from bingx_client import BingxClient  # ← помести bingx_client.py в ту же папку

# ================== НАСТРОЙКИ ==================
API_KEY = ""
API_SECRET = ""

SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT",
    "ADA-USDT", "XRP-USDT", "LTC-USDT",
    "BNB-USDT", "DOT-USDT", "LINK-USDT",
    "AVAX-USDT", "DOGE-USDT"
]

GROUPS = [
    ["BTC-USDT", "ETH-USDT", "BNB-USDT"],         # Majors
    ["SOL-USDT", "ADA-USDT", "XRP-USDT", "LTC-USDT"],  # Alts 1
    ["DOT-USDT", "LINK-USDT", "AVAX-USDT", "DOGE-USDT"]  # Alts 2
]

TIMEFRAME = "1h"  # "1h", "15m", "4h" и т.д.
CHECK_INTERVAL_SEC = 300  # 5 минут

# Лучшие параметры из твоего лога
PARAMS = {
    'corr_lookback': 150,
    'z_lookback': 75,
    'rebalance_interval': 24,           # каждые 24 часа обновляем режим
    'correlation_threshold': 0.55,
    'z_entry': 1.8,
    'z_exit': 0.3,
    'min_size_factor': 0.3,
    'trend_weight': 0.2
}

CAPITAL = 300.0
LEVERAGE = 10
MAX_NOTIONAL = CAPITAL * LEVERAGE
ALLOCATION_PER_TRADE = MAX_NOTIONAL / 5  # до 5 позиций одновременно

FEE_RATE = 0.0005  # 0.05% (BingX perpetual futures taker fee)

# ================== ЛОГИРОВАНИЕ ==================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("bingx_hedge_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== ДАННЫЕ ==================
class RealTimeData:
    def __init__(self, client: BingxClient):
        self.client = client
        self.data = {sym: pd.DataFrame() for sym in SYMBOLS}
        self.last_fetch_time = {sym: None for sym in SYMBOLS}

    def fetch_ohlcv(self, symbol: str, limit: int = 1000):
        """Получаем последние бары через BingX API"""
        try:
            # BingX endpoint для klines (OHLCV)
            path = "/openApi/swap/v2/quote/klines"
            params = {
                "symbol": symbol,
                "interval": TIMEFRAME,
                "limit": limit,
                'timestamp': int(time.time()*1000)
            }
            resp = self.client._public_request(path, params)
            if resp.get("code") != 0:
                logger.error(f"Ошибка получения klines для {symbol}: {resp}")
                return None

            data = resp["data"]
            df = pd.DataFrame(data, columns=["openTime", "open", "high", "low", "close", "volume", "closeTime"])
            df["timestamp"] = pd.to_datetime(df["openTime"], unit="ms")
            df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
            df = df.astype(float)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Fix potential inf
            df.dropna(inplace=True)  # Drop NaN
            return df
        except Exception as e:
            logger.error(f"Исключение при fetch_ohlcv {symbol}: {e}")
            return None

    def update(self):
        now = datetime.now(UTC)
        for sym in SYMBOLS:
            if self.last_fetch_time[sym] is None or (now - self.last_fetch_time[sym]).total_seconds() > 3600:
                df_new = self.fetch_ohlcv(sym, limit=500)
                if df_new is not None and not df_new.empty:
                    if self.data[sym].empty:
                        self.data[sym] = df_new
                    else:
                        # Добавляем новые бары, удаляем дубли
                         self.data[sym] = pd.concat([self.data[sym], df_new]).drop_duplicates()
                    self.last_fetch_time[sym] = now
                    logger.info(f"Обновлены данные {sym} ({len(self.data[sym])} баров)")

# ================== ОСНОВНАЯ СИСТЕМА ==================
class RealTimeHedgeBot:
    def __init__(self, client: BingxClient):
        self.client = client
        self.data = RealTimeData(client)
        self.positions = []  # текущие открытые пары
        self.regimes = None
        self.last_regime_update = None
        self.cum_pnl = 0.0

    def get_current_positions(self):
        """Получаем ВСЕ текущие позиции с BingX (без фильтра по symbol)"""
        try:
            path = "/openApi/swap/v2/user/positions"
            # Без параметра symbol → вернёт все открытые позиции
            params = {}  # или {"recvWindow": 5000} если хочешь
            urlpa = self.client.parseParam(params)
            resp = self.client.send_request("GET", path, urlpa, {})
            
            if resp.get("code") != 0:
                logger.error(f"Ошибка получения позиций: {resp}")
                return {}

            # data обычно список словарей
            pos_list = resp.get("data", [])
            
            active_pos = {}
            for p in pos_list:
                sym = p.get("symbol")
                amt = float(p.get("positionAmt", 0))
                if amt != 0 and sym:
                    active_pos[sym] = {
                        "side": "long" if amt > 0 else "short",  # или смотри positionSide
                        "size": abs(amt),
                        "entry_price": float(p.get("entryPrice", 0)),
                        "unrealized_pnl": float(p.get("unrealizedPnl", 0)),
                        # можно добавить больше полей при необходимости
                    }
            return active_pos
        
        except Exception as e:
            logger.error(f"Исключение при get_current_positions: {e}", exc_info=True)
            return {}

    def compute_strength_scores(self):
        t = len(self.data.data[SYMBOLS[0]]) - 1  # последний бар
        regimes = []
        for group in GROUPS:
            strong, weak, strengths, corr = self._compute_strength_scores_group(group, t)
            regimes.append((strong, weak, strengths, corr))
        return regimes

    def _compute_strength_scores_group(self, symbols, t):
        window = {}
        for s in symbols:
            df = self.data.data[s].iloc[max(0, t - PARAMS['corr_lookback']):t+1]
            pct = df["close"].pct_change()
            pct.replace([np.inf, -np.inf], np.nan, inplace=True)
            window[s] = pct.dropna()

        returns = pd.DataFrame(window)
        if returns.empty:
            return [], [], {}, pd.DataFrame()

        corr = returns.corr()

        scores = {}
        for s in symbols:
            avg_corr = corr[s].mean()
            close = self.data.data[s]["close"].iloc[:t+1]
            close = close.replace([np.inf, -np.inf], np.nan).dropna()
            ema_fast = close.ewm(span=12, adjust=False).mean()
            ema_slow = close.ewm(span=26, adjust=False).mean()
            trend = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
            scores[s] = avg_corr + PARAMS['trend_weight'] * trend

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        strengths = {s: 1 - i / (len(ranked) - 1) for i, (s, _) in enumerate(ranked)} if len(ranked) > 1 else {ranked[0][0]: 1.0}

        mid = len(ranked) // 2
        strong = [x[0] for x in ranked[:mid]]
        weak = [x[0] for x in ranked[mid:]]

        return strong, weak, strengths, corr

    def get_mark_price(self, symbol):
        price = self.client.get_mark_price(symbol)
        return price if price is not None else 0.0

    def place_or_adjust_position(self, s, w, z_now, beta, dir_sign, alloc, strengths, corr):
        """Открыть новую или скорректировать существующую позицию"""
        entry_s = self.data.data[s]["close"].iloc[-1]
        entry_w = self.data.data[w]["close"].iloc[-1]

        norm = abs(1) + abs(beta)
        base_size_s = (alloc / norm) / entry_s
        base_size_w = (alloc * abs(beta) / norm) / entry_w

        # Округление до precision BingX
        qty_s = round(base_size_s, 3) if "BTC" in s or "ETH" in s else round(base_size_s, 1)
        qty_w = round(base_size_w, 3) if "BTC" in w or "ETH" in w else round(base_size_w, 1)

        # Направление
        side_s = "long" if dir_sign > 0 else "short"
        side_w = "short" if dir_sign > 0 else "long"

        try:
            # Открываем/увеличиваем позицию по сильному
            if qty_s > 0:
                resp = self.client.place_market_order(
                    side=side_s,
                    qty=qty_s,
                    symbol=s
                )
                print(resp)
                logger.info(f"Открыта/увеличена позиция {s} {side_s.upper()} {qty_s}")

            # Открываем/увеличиваем позицию по слабому
            if qty_w > 0:
                resp = self.client.place_market_order(
                    side=side_w,
                    qty=qty_w,
                    symbol=w
                )
                print(resp)
                logger.info(f"Открыта/увеличена позиция {w} {side_w.upper()} {qty_w}")

            return {
                "s": s,
                "w": w,
                "dir": dir_sign,
                "entry_s": entry_s,
                "entry_w": entry_w,
                "base_size_s": qty_s,
                "base_size_w": qty_w,
                "size_factor": 1.0,
                "corr_entry": corr.loc[s, w],
                "strength_entry": strengths[s],
                "z_entry": z_now,
                "beta": beta
            }
        except Exception as e:
            logger.error(f"Ошибка открытия позиции {s}-{w}: {e}")
            return None

    def run(self):
        logger.info("=== BingX Hedge Bot запущен ===")
        self.data.update()  # Первичная загрузка

        while True:
            try:
                self.data.update()

                now = datetime.now(UTC)
                # Обновляем режим каждые rebalance_interval часов
                if self.last_regime_update is None or (now - self.last_regime_update).total_seconds() > PARAMS['rebalance_interval'] * 3600:
                    self.regimes = self.compute_strength_scores()
                    self.last_regime_update = now
                    logger.info("Обновлён режим (strength scores)")

                # Текущие позиции на бирже
                current_pos = self.get_current_positions()

                # Compute strength_scores
                strength_scores = {}
                for _, _, strengths, _ in self.regimes:
                    strength_scores.update(strengths)

                # Управление существующими позициями
                for pos in self.positions[:]:
                    s, w = pos["s"], pos["w"]
                    if s not in current_pos or w not in current_pos:
                        logger.warning(f"Позиция {s}-{w} исчезла с биржи!")
                        self.positions.remove(pos)
                        continue

                    # Пересчитываем текущий z-score
                    df_s = self.data.data[s]["close"]
                    df_w = self.data.data[w]["close"]
                    df_s = df_s.replace([np.inf, -np.inf], np.nan).dropna()
                    df_w = df_w.replace([np.inf, -np.inf], np.nan).dropna()
                    spread = np.log(df_s) - pos["beta"] * np.log(df_w)
                    z = (spread - spread.rolling(PARAMS['z_lookback']).mean()) / spread.rolling(PARAMS['z_lookback']).std()
                    current_z = z.iloc[-1] if not np.isnan(z.iloc[-1]) else 0

                    # Decay факторов
                    ret_s = df_s.pct_change().iloc[-PARAMS['corr_lookback']:]
                    ret_w = df_w.pct_change().iloc[-PARAMS['corr_lookback']:]
                    ret_s.replace([np.inf, -np.inf], np.nan, inplace=True)
                    ret_w.replace([np.inf, -np.inf], np.nan, inplace=True)
                    corr_now = ret_s.corr(ret_w)
                    if np.isnan(corr_now):
                        corr_now = pos["corr_entry"]

                    corr_factor = max(0.0, min(1.0, corr_now / pos["corr_entry"]))
                    strength_factor = max(0.0, min(1.0, strength_scores.get(s, pos["strength_entry"]) / pos["strength_entry"]))
                    new_size_factor = corr_factor * strength_factor

                    if new_size_factor < PARAMS['min_size_factor'] or abs(current_z) < PARAMS['z_exit']:
                        # Закрываем позицию
                        close_side_s = "short" if pos["dir"] > 0 else "long"
                        close_side_w = "long" if pos["dir"] > 0 else "short"
                        self.client.place_market_order(close_side_s, pos["base_size_s"], s)
                        self.client.place_market_order(close_side_w, pos["base_size_w"], w)
                        logger.info(f"Закрыта позиция {s}-{w} (z={current_z:.2f}, factor={new_size_factor:.2f})")
                        self.positions.remove(pos)
                    elif new_size_factor < pos["size_factor"]:
                        # Частичное закрытие
                        delta_factor = pos["size_factor"] - new_size_factor
                        delta_s = pos["base_size_s"] * delta_factor
                        delta_w = pos["base_size_w"] * delta_factor
                        close_side_s = "short" if pos["dir"] > 0 else "long"
                        close_side_w = "long" if pos["dir"] > 0 else "short"
                        self.client.place_market_order(close_side_s, delta_s, s)
                        self.client.place_market_order(close_side_w, delta_w, w)
                        pos["size_factor"] = new_size_factor
                        pos["base_size_s"] -= delta_s  # Update base sizes for future closes
                        pos["base_size_w"] -= delta_w
                        logger.info(f"Частичное уменьшение {s}-{w} на {delta_factor:.2f}")

                # Новые входы
                current_exposure = sum(
                    abs(current_pos.get(sym, {"size": 0})["size"] * (self.get_mark_price(sym) or 0)) for sym in SYMBOLS
                )

                for strong, weak, strengths, corr in self.regimes:
                    for s in strong:
                        for w in weak:
                            if corr.loc[s, w] < PARAMS['correlation_threshold']:
                                continue
                            if any(p["s"] == s and p["w"] == w for p in self.positions):
                                continue

                            # Вычисляем beta и z-score
                            df_s = self.data.data[s]["close"]
                            df_w = self.data.data[w]["close"]
                            df_s = df_s.replace([np.inf, -np.inf], np.nan).dropna()
                            df_w = df_w.replace([np.inf, -np.inf], np.nan).dropna()
                            ret_s = np.log(df_s).diff().dropna()
                            ret_w = np.log(df_w).diff().dropna()
                            if len(ret_s) < 2 or len(ret_w) < 2:
                                continue
                            beta = np.cov(ret_s, ret_w)[0,1] / np.var(ret_w)

                            if np.isnan(beta) or abs(beta) > 5:
                                continue

                            spread = np.log(df_s) - beta * np.log(df_w)
                            z = (spread - spread.rolling(PARAMS['z_lookback']).mean()) / spread.rolling(PARAMS['z_lookback']).std()
                            z_now = z.iloc[-1]
                            if np.isnan(z_now):
                                continue

                            if abs(z_now) < PARAMS['z_entry']:
                                continue

                            alloc = min(ALLOCATION_PER_TRADE, MAX_NOTIONAL - current_exposure)
                            if alloc <= 0:
                                continue

                            new_pos = self.place_or_adjust_position(s, w, z_now, beta, np.sign(z_now), alloc, strengths, corr)
                            if new_pos:
                                self.positions.append(new_pos)
                                current_exposure += alloc
                                logger.info(f"Новая позиция {s}-{w} | z={z_now:.2f} | alloc={alloc:.2f}")

                # Пауза до следующей итерации
                time.sleep(CHECK_INTERVAL_SEC)

            except Exception as e:
                logger.error(f"Критическая ошибка в цикле: {e}", exc_info=True)
                time.sleep(60)

if __name__ == "__main__":
    client = BingxClient(API_KEY, API_SECRET)
    bot = RealTimeHedgeBot(client)
    bot.run()