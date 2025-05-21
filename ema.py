import pandas as pd
import pandas_ta as ta # For EMA calculation, install with: pip install pandas_ta
import numpy as np

class EMACrossoverStrategy:
    def __init__(self, df, ema_fast_period=10, ema_slow_period=50, trailing_stop_pips=50, pip_value=0.0001):
        """
        Initializes the strategy.
        Args:
            df (pd.DataFrame): DataFrame with at least 'close' prices.
                               It's recommended to also have 'high' and 'low' for more realistic SL checking.
            ema_fast_period (int): Period for the fast EMA.
            ema_slow_period (int): Period for the slow EMA.
            trailing_stop_pips (float): Trailing stop in pips.
            pip_value (float): The value of one pip for the asset (e.g., 0.0001 for EURUSD).
        """
        if not isinstance(df, pd.DataFrame) or 'close' not in df.columns:
            raise ValueError("Input 'df' must be a pandas DataFrame with a 'close' column.")
        
        self.df = df.copy() # Work on a copy to avoid modifying original DataFrame
        self.ema_fast_period = ema_fast_period
        self.ema_slow_period = ema_slow_period
        self.trailing_stop_pips = trailing_stop_pips
        self.pip_value = pip_value 
        self.trailing_stop_price_diff = self.trailing_stop_pips * self.pip_value

        self._calculate_indicators()
        self._generate_signals()

    def _calculate_indicators(self):
        """Calculates EMAs and adds them to the DataFrame."""
        self.df[f'ema_fast'] = ta.ema(self.df['close'], length=self.ema_fast_period)
        self.df[f'ema_slow'] = ta.ema(self.df['close'], length=self.ema_slow_period)
        print(f"Calculated EMAs: Fast ({self.ema_fast_period}), Slow ({self.ema_slow_period})")

    def _generate_signals(self):
        """Generates buy and sell signals based on EMA crossovers."""
        # Shift EMAs to get previous values for crossover detection
        self.df['ema_fast_prev'] = self.df[f'ema_fast'].shift(1)
        self.df['ema_slow_prev'] = self.df[f'ema_slow'].shift(1)

        # Buy signal: Fast EMA crosses above Slow EMA
        self.df['buy_signal'] = (self.df[f'ema_fast'] > self.df[f'ema_slow']) & \
                                (self.df['ema_fast_prev'] <= self.df['ema_slow_prev'])

        # Sell signal: Fast EMA crosses below Slow EMA
        self.df['sell_signal'] = (self.df[f'ema_fast'] < self.df[f'ema_slow']) & \
                                 (self.df['ema_fast_prev'] >= self.df['ema_slow_prev'])
        
        # Remove signals where indicator data might be NaN (at the beginning)
        # Ensure boolean type for signal columns after NaN handling
        self.df['buy_signal'] = self.df['buy_signal'].fillna(False).astype(bool)
        self.df['sell_signal'] = self.df['sell_signal'].fillna(False).astype(bool)
        
        # Further ensure that signals are False if any EMA value is NaN
        condition_nan = self.df[f'ema_fast'].isna() | self.df[f'ema_slow'].isna() | \
                        self.df['ema_fast_prev'].isna() | self.df['ema_slow_prev'].isna()
        self.df.loc[condition_nan, 'buy_signal'] = False
        self.df.loc[condition_nan, 'sell_signal'] = False

        print("Generated Buy/Sell signals.")

    def run_backtest_simulation(self):
        """
        Simulates the strategy execution and trailing stop loss.
        This is a simplified simulation. A proper backtester would be more complex.
        Requires 'high' and 'low' columns in the DataFrame for stop-loss checking.
        """
        if 'high' not in self.df.columns or 'low' not in self.df.columns:
            print("Warning: 'high' and 'low' columns not found in DataFrame. Stop loss checks will use 'close' price, which is less realistic.")
            # Add high/low columns based on close if they don't exist, for the simulation to run
            if 'high' not in self.df.columns: self.df['high'] = self.df['close']
            if 'low' not in self.df.columns: self.df['low'] = self.df['close']


        positions = []  # To store active trades: {'type': 'buy'/'sell', 'entry_price': float, 'stop_loss': float, 'entry_bar': int}
        trade_log = []  # To log completed trades

        print(f"\nRunning Backtest Simulation with Trailing Stop: {self.trailing_stop_pips} pips ({self.trailing_stop_price_diff:.5f} price diff)")

        for i in range(len(self.df)):
            current_price_close = self.df['close'].iloc[i]
            current_price_high = self.df['high'].iloc[i]
            current_price_low = self.df['low'].iloc[i]

            # --- Manage existing positions and trailing stops ---
            active_position_indices_to_remove = []
            for pos_idx, pos in enumerate(positions):
                if pos['type'] == 'buy':
                    # Check if stop loss hit
                    if current_price_low <= pos['stop_loss']:
                        exit_price = pos['stop_loss'] # Exit at SL price
                        profit = exit_price - pos['entry_price']
                        trade_log.append({'entry_bar': pos['entry_bar'], 'exit_bar': i, 'type': 'buy', 
                                          'entry_price': pos['entry_price'], 'exit_price': exit_price, 'profit': profit, 'reason': 'SL Hit'})
                        active_position_indices_to_remove.append(pos_idx)
                        print(f"Bar {i}: BUY position (entered at {pos['entry_price']:.5f}) closed at {exit_price:.5f} (SL Hit). Profit: {profit:.5f}")
                        continue 

                    # Update trailing stop loss
                    potential_new_sl = current_price_close - self.trailing_stop_price_diff 
                    if potential_new_sl > pos['stop_loss']:
                        pos['stop_loss'] = potential_new_sl
                        print(f"Bar {i}: BUY position SL trailed to {pos['stop_loss']:.5f}")

                elif pos['type'] == 'sell':
                    # Check if stop loss hit
                    if current_price_high >= pos['stop_loss']:
                        exit_price = pos['stop_loss'] # Exit at SL price
                        profit = pos['entry_price'] - exit_price 
                        trade_log.append({'entry_bar': pos['entry_bar'], 'exit_bar': i, 'type': 'sell', 
                                          'entry_price': pos['entry_price'], 'exit_price': exit_price, 'profit': profit, 'reason': 'SL Hit'})
                        active_position_indices_to_remove.append(pos_idx)
                        print(f"Bar {i}: SELL position (entered at {pos['entry_price']:.5f}) closed at {exit_price:.5f} (SL Hit). Profit: {profit:.5f}")
                        continue

                    # Update trailing stop loss
                    potential_new_sl = current_price_close + self.trailing_stop_price_diff
                    if potential_new_sl < pos['stop_loss']:
                        pos['stop_loss'] = potential_new_sl
                        print(f"Bar {i}: SELL position SL trailed to {pos['stop_loss']:.5f}")
            
            for pos_idx in sorted(active_position_indices_to_remove, reverse=True):
                del positions[pos_idx]

            # --- Check for new signals ---
            # Only one position at a time. Close existing before opening new if it's an opposite signal.
            if self.df['buy_signal'].iloc[i]:
                if positions and positions[0]['type'] == 'sell': # Close existing sell if buy signal
                    closed_pos = positions.pop(0)
                    exit_price = current_price_close 
                    profit = closed_pos['entry_price'] - exit_price
                    trade_log.append({'entry_bar': closed_pos['entry_bar'], 'exit_bar': i, 'type': 'sell', 
                                      'entry_price': closed_pos['entry_price'], 'exit_price': exit_price, 'profit': profit, 'reason': 'New Opposite (BUY) Signal'})
                    print(f"Bar {i}: SELL position closed at {exit_price:.5f} due to new BUY signal. Profit: {profit:.5f}")

                if not positions: # Ensure no position is open before entering new
                    entry_price = current_price_close 
                    initial_sl = entry_price - self.trailing_stop_price_diff
                    positions.append({'type': 'buy', 'entry_price': entry_price, 'stop_loss': initial_sl, 'entry_bar': i})
                    print(f"Bar {i}: BUY signal. Entered at {entry_price:.5f}. Initial SL: {initial_sl:.5f}")

            elif self.df['sell_signal'].iloc[i]:
                if positions and positions[0]['type'] == 'buy': # Close existing buy if sell signal
                    closed_pos = positions.pop(0)
                    exit_price = current_price_close
                    profit = exit_price - closed_pos['entry_price']
                    trade_log.append({'entry_bar': closed_pos['entry_bar'], 'exit_bar': i, 'type': 'buy', 
                                      'entry_price': closed_pos['entry_price'], 'exit_price': exit_price, 'profit': profit, 'reason': 'New Opposite (SELL) Signal'})
                    print(f"Bar {i}: BUY position closed at {exit_price:.5f} due to new SELL signal. Profit: {profit:.5f}")
                
                if not positions:
                    entry_price = current_price_close
                    initial_sl = entry_price + self.trailing_stop_price_diff
                    positions.append({'type': 'sell', 'entry_price': entry_price, 'stop_loss': initial_sl, 'entry_bar': i})
                    print(f"Bar {i}: SELL signal. Entered at {entry_price:.5f}. Initial SL: {initial_sl:.5f}")
        
        # Close any open positions at the end of the data
        for pos_idx, pos in enumerate(positions):
            exit_price = self.df['close'].iloc[-1]
            if pos['type'] == 'buy':
                profit = exit_price - pos['entry_price']
            else: # sell
                profit = pos['entry_price'] - exit_price
            trade_log.append({'entry_bar': pos['entry_bar'], 'exit_bar': len(self.df)-1, 'type': pos['type'], 
                              'entry_price': pos['entry_price'], 'exit_price': exit_price, 'profit': profit, 'reason': 'End of Data'})
            print(f"Bar {len(self.df)-1}: {pos['type'].upper()} position (entered at {pos['entry_price']:.5f}) closed at {exit_price:.5f} (End of Data). Profit: {profit:.5f}")

        self.trade_log_df = pd.DataFrame(trade_log)
        return self.trade_log_df

# --- Example Usage ---
if __name__ == '__main__':
    # Create sample data (replace with your actual market data)
    # Using a slightly longer and more varied dataset for better signal generation
    data_dict = {
        'timestamp': pd.to_datetime([
            '2023-01-01 00:00', '2023-01-01 01:00', '2023-01-01 02:00', '2023-01-01 03:00', 
            '2023-01-01 04:00', '2023-01-01 05:00', '2023-01-01 06:00', '2023-01-01 07:00',
            '2023-01-01 08:00', '2023-01-01 09:00', '2023-01-01 10:00', '2023-01-01 11:00',
            '2023-01-01 12:00', '2023-01-01 13:00', '2023-01-01 14:00', '2023-01-01 15:00',
            '2023-01-01 16:00', '2023-01-01 17:00', '2023-01-01 18:00', '2023-01-01 19:00',
            '2023-01-01 20:00', '2023-01-01 21:00', '2023-01-01 22:00', '2023-01-01 23:00'
        ]),
        'open':  [1.1000, 1.1010, 1.1020, 1.0980, 1.0950, 1.0970, 1.0990, 1.1030, 1.1050, 1.1040, 1.1020, 1.1000, 1.0970, 1.0950, 1.0930, 1.0960, 1.0980, 1.0970, 1.0990, 1.1010, 1.0980, 1.0960, 1.0940, 1.0920],
        'high':  [1.1015, 1.1025, 1.1030, 1.0995, 1.0975, 1.0995, 1.1035, 1.1055, 1.1060, 1.1050, 1.1030, 1.1010, 1.0980, 1.0960, 1.0940, 1.0970, 1.0990, 1.0985, 1.1000, 1.1020, 1.0995, 1.0975, 1.0950, 1.0930],
        'low':   [1.0990, 1.1005, 1.0975, 1.0945, 1.0940, 1.0960, 1.0980, 1.1020, 1.1035, 1.1015, 1.0995, 1.0965, 1.0945, 1.0925, 1.0920, 1.0950, 1.0965, 1.0960, 1.0975, 1.0985, 1.0955, 1.0935, 1.0915, 1.0900],
        'close': [1.1010, 1.1020, 1.0980, 1.0950, 1.0970, 1.0990, 1.1030, 1.1050, 1.1040, 1.1020, 1.1000, 1.0970, 1.0950, 1.0930, 1.0960, 1.0955, 1.0975, 1.0980, 1.0995, 1.1005, 1.0970, 1.0945, 1.0925, 1.0910]
    }
    sample_df = pd.DataFrame(data_dict)
    sample_df.set_index('timestamp', inplace=True)

    # --- Parameters for the strategy ---
    fast_ema_period = 5 
    slow_ema_period = 10
    trailing_stop_pips_val = 15 # e.g., 15 pips
    asset_pip_value_val = 0.0001 # For a pair like EURUSD

    # Initialize and run strategy
    strategy_runner = EMACrossoverStrategy(df=sample_df, 
                                           ema_fast_period=fast_ema_period, 
                                           ema_slow_period=slow_ema_period, 
                                           trailing_stop_pips=trailing_stop_pips_val,
                                           pip_value=asset_pip_value_val)
    
    print("\nDataFrame with Indicators and Signals (last 10 rows):")
    print(strategy_runner.df[['close', 'ema_fast', 'ema_slow', 'buy_signal', 'sell_signal']].tail(10))

    # Run the simplified backtest
    trade_log_output_df = strategy_runner.run_backtest_simulation()

    print("\nTrade Log:")
    if not trade_log_output_df.empty:
        print(trade_log_output_df)
        total_profit = trade_log_output_df['profit'].sum()
        print(f"\nTotal P/L (price terms): {total_profit:.5f}")
        print(f"Total P/L (pips): {total_profit / asset_pip_value_val:.2f}")
        print(f"Total Trades: {len(trade_log_output_df)}")
        
        # Basic performance metrics
        winning_trades = trade_log_output_df[trade_log_output_df['profit'] > 0]
        losing_trades = trade_log_output_df[trade_log_output_df['profit'] <= 0] # Including break-even as non-winning
        win_rate = (len(winning_trades) / len(trade_log_output_df) * 100) if not trade_log_output_df.empty else 0
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Losing Trades: {len(losing_trades)}")
        print(f"Win Rate: {win_rate:.2f}%")
        if not winning_trades.empty:
            print(f"Average Win: {winning_trades['profit'].mean():.5f}")
        if not losing_trades.empty:
            print(f"Average Loss: {losing_trades['profit'].mean():.5f}")

    else:
        print("No trades were executed.")
