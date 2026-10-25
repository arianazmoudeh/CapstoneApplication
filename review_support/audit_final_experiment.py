from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results/china_factor_timing_final'
BASELINE = ROOT / 'results/china_factor_timing_m6'
FACTORS = ['Value', 'Profitability', 'Momentum']
MODELS = ['Ridge', 'Elastic Net', 'KNN', 'SVR', 'Random Forest', 'Extra Trees', 'XGBoost', 'Decision Tree', 'AdaBoost', 'LightGBM', 'MLP', 'LSTM', 'GRU', 'CNN', 'Prophet']
SEEDS = [7, 21, 42, 84, 123]
SELECTIONS = pd.to_datetime(['2024-12-31', '2025-06-30', '2025-12-31', '2026-06-30'])
CONTROLS = ['Six-month selector', 'Equal factors', 'Initial model family', 'Initial factor mix']
checks = []


def read(name, dates=(), folder=RESULTS):
    return pd.read_csv(folder / name, dtype={'code': str}, parse_dates=list(dates))


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def close(left, right, label, atol=1e-7, rtol=1e-8):
    try:
        np.testing.assert_allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), atol=atol, rtol=rtol)
    except AssertionError as error:
        raise AssertionError(label + ': ' + str(error)) from error


def record(name, operation):
    try:
        detail = operation()
        checks.append({'check': name, 'status': 'PASS', 'detail': str(detail or '')})
    except Exception as error:
        checks.append({'check': name, 'status': 'FAIL', 'detail': f'{type(error).__name__}: {error}'})


required = [
    'predictions_all_seeds.csv', 'factor_targets.csv', 'selection_scores.csv', 'selection_windows.csv',
    'selected_models_all_seeds.csv', 'factor_weights_all_seeds.csv', 'stock_weights_all_seeds.csv',
    'seed_runs/performance.csv', 'seed_runs/monthly_returns.csv', 'seed_runs/daily_exposure.csv',
    'shadow/performance.csv', 'shadow/monthly_returns.csv', 'shadow/daily_exposure.csv',
    'uncertainty.csv', 'window_sensitivity.csv', 'window_sensitivity_predictions.csv',
]
missing = [name for name in required if not (RESULTS / name).exists()]
if missing:
    print('NOT READY: ' + ', '.join(missing))
    sys.exit(2)

predictions = read('predictions_all_seeds.csv', ['signal', 'entry', 'exit'])
targets = read('factor_targets.csv', ['signal', 'entry', 'exit']).set_index('signal')
reference = predictions[['signal', 'entry', 'exit', 'split']].drop_duplicates().sort_values('signal')
eval_reference = reference.loc[reference.split.eq('evaluation')]
eval_signals = pd.DatetimeIndex(eval_reference.signal)
shadow_reference = reference.set_index('signal')
selected = read('selected_models_all_seeds.csv', ['selection_date', 'execution_date', 'first_entry', 'last_completed_exit'])
scores = read('selection_scores.csv', ['selection_date', 'first_signal', 'last_signal', 'last_exit'])
windows = read('selection_windows.csv', ['selection_date', 'signal', 'entry', 'exit'])
factor_weights = read('factor_weights_all_seeds.csv', ['signal'])
stock_weights = read('stock_weights_all_seeds.csv', ['signal'])
baskets = read('factor_baskets.csv', ['signal'])
companies = read('companies.csv')
stock_codes = companies.code.tolist()
source_prices = pd.read_csv(ROOT / 'inputs/china_factor_100/prices.csv', dtype={'code': str}, parse_dates=['date'])
source_calendar = pd.DatetimeIndex(pd.read_csv(ROOT / 'inputs/china_benchmark/market_calendar.csv', parse_dates=['date']).date)
source_quotes = source_prices.pivot(index='date', columns='code', values='adjusted_close').reindex(index=source_calendar, columns=stock_codes)
source_quotes = source_quotes.where(source_quotes.gt(0)).ffill()


def check_dimensions_and_reproduction():
    require(len(companies) == 100 and companies.code.nunique() == 100, 'Expected100 distinct companies.')
    require(set(predictions.seed) == set(SEEDS), 'Missing seed forecasts.')
    require(set(predictions.model) == set(MODELS + ['Historical mean']), 'Model names differ.')
    require(len(predictions) == 5 * 45 * 16 * 3, 'Incomplete forecast grid.')
    require(not predictions.duplicated(['seed', 'signal', 'model', 'factor']).any(), 'Duplicate forecast key.')
    require(np.isfinite(predictions[['predicted_return', 'actual_return']]).all().all(), 'Nonfinite forecast.')
    require(len(reference) == 45 and reference.split.value_counts().to_dict() == {'validation': 23, 'evaluation': 21, 'bridge': 1}, 'Shared45-month calendar differs.')
    require(reference.entry.iloc[0] == pd.Timestamp('2023-01-03'), 'Shadow start differs.')
    require(eval_reference.entry.min() == pd.Timestamp('2025-01-02') and eval_reference.exit.max() == pd.Timestamp('2026-10-08'), 'Evaluation dates differ.')
    close((reference.exit.iloc[:-1].to_numpy() == reference.entry.iloc[1:].to_numpy()).astype(float), 1, 'Shadow calendar has a gap')
    expected = targets[FACTORS].stack().rename('expected').rename_axis(['signal', 'factor']).reset_index()
    paired = predictions.merge(expected, on=['signal', 'factor'], validate='many_to_one')
    require(len(paired) == len(predictions), 'Missing factor outcome.')
    close(paired.actual_return, paired.expected, 'Factor targets differ', atol=1e-12)
    original = read('predictions_all.csv', ['signal'], BASELINE)
    main = predictions.loc[predictions.seed.eq(42)]
    match = original.merge(main, on=['signal', 'model', 'factor'], suffixes=('_m6', '_final'), validate='one_to_one')
    require(len(match) == len(original) == 44 * 16 * 3, 'M6 overlap differs.')
    close(match.predicted_return_m6, match.predicted_return_final, 'Seed42 forecasts did not reproduce M6', atol=1e-8, rtol=0)
    deterministic = ['Historical mean', 'Ridge', 'Elastic Net', 'KNN', 'SVR', 'Prophet']
    invariant = predictions.loc[predictions.model.isin(deterministic)].groupby(['signal', 'model', 'factor']).predicted_return.agg(['min', 'max'])
    close(invariant['min'], invariant['max'], 'A fixed-model forecast changes across seeds', atol=0, rtol=0)
    return '10,800 forecasts; all 45 common dates; bridge present; 2,112 seed 42 forecasts reproduce M6.'


def check_training_and_bridge():
    audit = read('training_audit.csv', ['signal', 'last_training_exit', 'last_input_week'])
    require(len(audit) == 2520, 'Expected2520 fitted-model audit records.')
    require(not audit.duplicated(['seed', 'signal', 'model']).any(), 'Duplicate training audit.')
    require(audit.last_training_exit.le(audit.signal).all(), 'Future target entered fitting.')
    require(audit.last_input_week.le(audit.signal).all(), 'Future input entered fitting.')
    bridge = audit.loc[audit.signal.eq(pd.Timestamp('2024-11-29'))]
    require(bridge.training_months.eq(70).all(), 'Bridge fitting count should be70.')
    require(bridge.last_training_exit.eq(pd.Timestamp('2024-11-01')).all(), 'Bridge target cutoff differs.')
    deep = audit.loc[audit.model.isin(['LSTM', 'GRU', 'CNN'])].copy()
    require(len(deep) == 3 * 5 * 45, 'Deep-model audit is incomplete.')
    for column in ['fit_target_end', 'internal_validation_start', 'internal_validation_end']:
        deep[column] = pd.to_datetime(deep[column])
    require(deep.epochs_run.between(1, 100).all(), 'Deep-model epoch bound differs.')
    require(deep.best_epoch.between(1, deep.epochs_run).all(), 'Deep-model best epoch invalid.')
    require(deep.fit_target_end.le(deep.internal_validation_start).all(), 'Fit labels overlap internal decision.')
    require(deep.internal_validation_end.le(deep.signal).all(), 'Early stopping uses a future outcome.')
    return 'All 2,520 fits use completed outcomes; 675 deep fits have purged early stopping within 100 epochs; bridge cutoff verified.'


def check_selection():
    monthly = read('monthly_returns.csv', ['signal', 'entry', 'exit'], RESULTS / 'shadow')
    trades = read('trades.csv', ['signal', 'date'], RESULTS / 'shadow')
    monthly[['seed', 'model']] = monthly.strategy.str.split('|', n=1, expand=True)
    trades[['seed', 'model']] = trades.strategy.str.split('|', n=1, expand=True)
    monthly['seed'] = monthly.seed.astype(int)
    trades['seed'] = trades.seed.astype(int)
    require(len(selected) == 20 and len(scores) == 300 and len(windows) == 360, 'Selection table dimensions differ.')
    expected_counts = [23, 29, 35, 41]
    for seed in SEEDS:
        for selection_index, date in enumerate(SELECTIONS):
            complete = shadow_reference.loc[shadow_reference.exit.le(date)]
            require(len(complete) == expected_counts[selection_index], 'Available completed count differs.')
            expected = complete.tail(18)
            reported_window = windows.loc[windows.seed.eq(seed) & windows.selection_date.eq(date)].sort_values('signal').set_index('signal')
            require(reported_window.index.equals(expected.index), 'Window is not latest18 expected completed periods.')
            require(reported_window.exit.le(date).all(), 'Unfinished holding period entered selection.')
            require(reported_window.entry.equals(expected.entry) and reported_window.exit.equals(expected.exit), 'Window entry/exit dates differ.')
            past = monthly.loc[monthly.seed.eq(seed) & monthly.signal.isin(expected.index)].pivot(index='signal', columns='model', values='return').reindex(expected.index)
            turnover = trades.loc[trades.seed.eq(seed) & trades.signal.isin(expected.index)].pivot(index='signal', columns='model', values='turnover').reindex(expected.index)
            require(set(past.columns) == set(MODELS + ['Equal factors']) and past.notna().all().all(), 'Candidate history is incomplete.')
            advantages = past[MODELS].sub(past['Equal factors'], axis=0).mean()
            average_turnover = turnover[MODELS].mean()
            own_scores = scores.loc[scores.seed.eq(seed) & scores.selection_date.eq(date)].set_index('model').reindex(MODELS)
            close(own_scores.score, advantages.reindex(MODELS), 'Selection score differs', atol=1e-12)
            close(own_scores.mean_turnover, average_turnover.reindex(MODELS), 'Selection turnover differs', atol=1e-12)
            candidates = advantages.loc[(advantages - advantages.max()).abs().le(1e-12)].index
            chosen = sorted(candidates, key=lambda model: (average_turnover[model], model))[0]
            row = selected.loc[selected.seed.eq(seed) & selected.selection_date.eq(date)].iloc[0]
            require(row.model == chosen, 'Winner differs from declared tie/score rule.')
            require(own_scores.loc[own_scores.selected].index.tolist() == [chosen], 'Selected flag differs.')
            require(row.execution_date == targets.loc[date, 'entry'], 'Selection executes on wrong date.')
            require(row.first_entry == expected.entry.iloc[0] and row.last_completed_exit == expected.exit.iloc[-1], 'Selection boundary metadata differs.')
    return 'All 20 selections recomputed from 18 completed paired shadow returns; ties, turnover and execution dates match.'


def check_weights_and_controls():
    allocation_rows = factor_weights.sort_values(['seed', 'strategy', 'signal'])
    require(len(allocation_rows) == (20 + 4 * 4) * 21, 'Evaluation factor rows incomplete.')
    require(not allocation_rows.duplicated(['seed', 'strategy', 'signal']).any(), 'Duplicate factor allocation.')
    close(allocation_rows[FACTORS].sum(axis=1), 1, 'Factor weights do not sum to1', atol=1e-12)
    stock_table = stock_weights.set_index(['seed', 'strategy', 'signal', 'code']).sort_index()
    basket_table = baskets.pivot(index=['signal', 'code'], columns='factor', values='weight').fillna(0)
    for seed in SEEDS:
        seed_selection = selected.loc[selected.seed.eq(seed)].set_index('selection_date').sort_index()
        first_family = seed_selection.model.iloc[0]
        first_mix = None
        for row in allocation_rows.loc[allocation_rows.seed.eq(seed)].itertuples(index=False):
            actual = pd.Series([getattr(row, f) for f in FACTORS], index=FACTORS)
            if row.strategy == 'Equal factors':
                expected = pd.Series(1 / 3, index=FACTORS)
            elif row.strategy == 'Initial factor mix':
                fixed_forecast = predictions.loc[predictions.seed.eq(seed) & predictions.signal.eq(eval_signals[0]) & predictions.model.eq(first_family)].set_index('factor').predicted_return.reindex(FACTORS)
                expected = pd.Series(0.0, index=FACTORS)
                expected.loc[fixed_forecast.sort_values(ascending=False, kind='stable').index] = [0.5, 0.3, 0.2]
            else:
                family = seed_selection.loc[:row.signal].model.iloc[-1] if row.strategy == 'Six-month selector' else first_family if row.strategy == 'Initial model family' else row.strategy
                require(row.selected_model == family, 'Forecast source family differs.')
                forecast = predictions.loc[predictions.seed.eq(seed) & predictions.signal.eq(row.signal) & predictions.model.eq(family)].set_index('factor').predicted_return.reindex(FACTORS)
                expected = pd.Series(0.0, index=FACTORS)
                expected.loc[forecast.sort_values(ascending=False, kind='stable').index] = [0.5, 0.3, 0.2]
            close(actual, expected, 'Factor mix differs from forecast/control rule', atol=1e-12)
            raw = basket_table.loc[row.signal].reindex(stock_codes, fill_value=0).mul(expected).sum(axis=1)
            target = raw / raw.abs().sum()
            reported = stock_table.loc[(seed, row.strategy, row.signal)].reindex(stock_codes).weight
            close(reported, target, 'Netted stock target differs', atol=1e-12)
            close(reported.sum(), 0, 'Target net exposure is notzero', atol=1e-12)
            close(reported.abs().sum(), 1, 'Target gross exposure is not1', atol=1e-12)
    return 'All 756 factor mixes and 75,600 stock targets match monthly forecasts, frozen-family/frozen-mix controls and netting.'


def check_ledger(folder, shadow=False):
    performance = read('performance.csv', ['start', 'end'], folder)
    exposure = read('daily_exposure.csv', ['date'], folder)
    monthly = read('monthly_returns.csv', ['signal', 'entry', 'exit'], folder)
    trades = read('trades.csv', ['signal', 'date'], folder)
    logs = read('trade_log.csv', ['signal', 'date'], folder)
    holdings = read('actual_stock_weights.csv', ['signal', 'date'], folder)
    values = read('portfolio_values.csv', ['date'], folder).set_index('date')
    if not shadow:
        for frame in [performance, exposure, monthly, trades, logs, holdings]:
            frame['strategy'] = frame.seed.astype(str) + '|' + frame.strategy
    performance = performance.set_index('strategy')
    expected_portfolios = 80 if shadow else 36
    expected_periods = 45 if shadow else 21
    require(len(performance) == expected_portfolios and len(values.columns) == expected_portfolios, 'Portfolio count differs.')
    require(len(monthly) == expected_portfolios * expected_periods, 'Monthly ledger count differs.')
    close(exposure.value, exposure.cash + exposure.long_value - exposure.short_value, 'NAV does not reconcile')
    close(exposure.gross, (exposure.long_value + exposure.short_value) / exposure.value, 'Gross exposure differs')
    close(exposure.net, (exposure.long_value - exposure.short_value) / exposure.value, 'Net exposure differs')
    close(logs.cost, logs.signed_trade_value.abs() * 0.001, 'Stock fee differs')
    net_trades = logs.groupby(['strategy', 'date']).signed_trade_value.sum()
    stock_fees = logs.groupby(['strategy', 'date']).cost.sum()
    for name, frame in exposure.groupby('strategy', sort=False):
        frame = frame.sort_values('date').set_index('date')
        signed = net_trades.loc[name].reindex(frame.index, fill_value=0)
        fees = stock_fees.loc[name].reindex(frame.index, fill_value=0)
        previous_cash = frame.cash.shift(1, fill_value=100000)
        close(frame.cash, previous_cash - signed - fees - frame.borrow_fee, 'Cash account differs')
        close(frame.trading_fee, fees, 'Daily fees differ from net stock trades')
        previous_short = frame.short_value.shift(1, fill_value=0)
        preceding_dates = frame.index.to_series().shift(1, fill_value=values.index[0])
        days = (frame.index.to_series() - preceding_dates).dt.days
        close(frame.borrow_fee, previous_short * 0.03 * days / 365, 'Borrow fee differs')
        close(frame.trading_fee.sum(), performance.loc[name, 'trading_costs'], 'Trading fee total differs')
        close(frame.borrow_fee.sum(), performance.loc[name, 'borrow_costs'], 'Borrow fee total differs')
        close(frame.value, values.loc[frame.index, name], 'Chart curve differs')
        units = holdings.loc[holdings.strategy.eq(name)].pivot(index='date', columns='code', values='units').reindex(index=frame.index, columns=stock_codes).ffill().fillna(0)
        market = units * source_quotes.reindex(index=frame.index, columns=stock_codes).fillna(0)
        close(market.clip(lower=0).sum(axis=1), frame.long_value, 'Long values differ from held units and input quotes')
        close(-market.clip(upper=0).sum(axis=1), frame.short_value, 'Short values differ from held units and input quotes')
        months = monthly.loc[monthly.strategy.eq(name)].sort_values('signal')
        starts = frame.value_before_trade.reindex(months.entry).to_numpy()
        ends = frame.value_before_trade.reindex(months.exit).to_numpy()
        close(months.start_value_before_trade, starts, 'Month starting NAV differs')
        close(months.end_value_before_next_trade, ends, 'Month ending NAV differs')
        close(months['return'], ends / starts - 1, 'Month return uses wrong boundary', atol=1e-12)
        close(np.prod(1 + months['return']), performance.loc[name, 'final_value'] / 100000, 'Monthly returns fail to compound')
        curve = values[name]
        changes = curve.pct_change(fill_method=None).dropna()
        years = (performance.loc[name, 'end'] - performance.loc[name, 'start']).days / 365.25
        close(curve.iloc[0], 100000, 'Initial capital differs')
        close(curve.iloc[-1], performance.loc[name, 'final_value'], 'Final NAV differs')
        close((curve.iloc[-1] / 100000 - 1) * 100, performance.loc[name, 'total_return'], 'Total return differs')
        close(((curve.iloc[-1] / 100000) ** (1 / years) - 1) * 100, performance.loc[name, 'annual_return'], 'Annual return differs')
        close(changes.std(ddof=1) * np.sqrt(252) * 100, performance.loc[name, 'volatility'], 'Volatility differs')
        close(changes.mean() / changes.std(ddof=1) * np.sqrt(252), performance.loc[name, 'sharpe'], 'Sharpe differs')
        close((curve / curve.cummax() - 1).min() * 100, performance.loc[name, 'drawdown'], 'Drawdown differs')
    ordered = holdings.sort_values(['strategy', 'code', 'date']).copy()
    ordered['previous_units'] = ordered.groupby(['strategy', 'code']).units.shift(1, fill_value=0)
    frozen = ordered.loc[~ordered.tradable]
    close(frozen.units, frozen.previous_units, 'Frozen holding changed units', atol=1e-10)
    join = logs.merge(holdings[['strategy', 'date', 'code', 'tradable']], on=['strategy', 'date', 'code'], validate='one_to_one')
    require(join.tradable.all(), 'A trade used no fresh quote.')
    rebalance = trades.merge(exposure[['strategy', 'date', 'value']], on=['strategy', 'date'], validate='one_to_one')
    close(rebalance.value_before_trade - rebalance.value_after_trade, rebalance.cost, 'Trade fee does not explain NAV movement')
    close(rebalance.value, rebalance.value_after_trade, 'Rebalance NAV differs from ledger')
    close(trades.turnover, trades.traded_notional / trades.value_before_trade, 'Normalized turnover differs')
    require(performance.start.nunique() == 1 and performance.end.nunique() == 1, 'Portfolios use different dates.')
    return f'{expected_portfolios} continuous ledgers; {expected_periods} periods each; cash, fees, borrowing, frozen holdings, NAV and all risk/return metrics reconcile.'


def check_m6_portfolio_reproduction():
    old = read('performance.csv', folder=BASELINE).set_index('strategy')
    new = read('performance.csv').set_index('strategy')
    names = ['Equal factors', 'Historical mean'] + MODELS
    for column in ['final_value', 'annual_return', 'sharpe', 'drawdown', 'trading_costs', 'borrow_costs']:
        close(new.loc[names, column], old.loc[names, column], 'Old portfolio result changed: ' + column, atol=1e-7)
    return 'All 17 original comparable portfolio results reproduce the M6 engine.'


def check_uncertainty_and_costs():
    monthly = read('monthly_returns.csv')
    pivot = monthly.pivot(index='signal', columns='strategy', values='return').sort_index()
    differences = (pivot['Six-month selector'] - pivot['Equal factors']).to_numpy()
    intervals = read('uncertainty.csv').set_index('block_length')
    require(set(intervals.index) == {2, 3, 6}, 'Bootstrap block checks differ.')
    require(intervals.months.eq(21).all() and intervals.repetitions.eq(10000).all() and intervals.seed.eq(20261025).all(), 'Bootstrap settings differ.')
    close(intervals.mean_difference, differences.mean() * 100, 'Mean paired difference differs', atol=1e-12)
    for length in [3, 2, 6]:
        rng = np.random.default_rng(20261025)
        current = rng.integers(0, 21, size=10000)
        sums = differences[current].copy()
        for step in range(1, 21):
            restart = rng.random(10000) < 1 / length
            starts = rng.integers(0, 21, size=10000)
            current = np.where(restart, starts, (current + 1) % 21)
            sums += differences[current]
        low, high = np.quantile(sums / 21 * 100, [0.025, 0.975])
        close(intervals.loc[length, ['lower', 'upper']], [low, high], 'Stationary bootstrap interval differs', atol=1e-10)
    costs = read('cost_sensitivity.csv')
    require(len(costs) == 36, 'Six comparisons each need three values for each cost.')
    for (kind, strategy), frame in costs.groupby(['kind', 'strategy']):
        if kind == 'Trading cost':
            close(sorted(frame.trading_cost_rate), [0, 0.001, 0.003], 'Trading cost range differs')
            close(frame.annual_borrow_cost_rate, 0.03, 'Borrow cost changed within trading check')
        else:
            close(sorted(frame.annual_borrow_cost_rate), [0, 0.03, 0.06], 'Borrow cost range differs')
            close(frame.trading_cost_rate, 0.001, 'Trading cost changed within borrow check')
        require(frame.start.nunique() == 1 and frame.end.nunique() == 1, 'Cost check dates differ.')
        close(frame.loc[frame.trading_cost_rate.eq(0), 'trading_costs'], 0, 'Zero trading rate charged costs')
        close(frame.loc[frame.annual_borrow_cost_rate.eq(0), 'borrow_costs'], 0, 'Zero borrowing rate charged costs')
    seeds = read('seed_results.csv')
    require(set(seeds.seed) == set(SEEDS) and len(seeds) == 5, 'Seed results incomplete.')
    return 'All three stationary bootstrap intervals exactly recomputed; 36 cost rows and all five seeds present.'


def check_window_sensitivity():
    predictions = read('window_sensitivity_predictions.csv', ['signal', 'entry', 'exit'])
    audit = read('window_sensitivity_training.csv', ['signal', 'first_training_signal', 'last_training_exit', 'last_input_week'])
    metrics = read('window_sensitivity.csv')
    coverage = read('window_sensitivity_coverage.csv')
    require(len(predictions) == 3 * 3 * 44 * 3 and len(metrics) == 18, 'Window sensitivity dimensions differ.')
    require(set(predictions.window) == {'1/4/13; risk13', '1/3/12; risk12', '1/6/26; risk26'}, 'Window definitions differ.')
    require(coverage.common_samples.eq(89).all(), 'Expected89 matched training/sample months.')
    require(audit.first_training_signal.eq(pd.Timestamp('2019-04-30')).all(), 'Common history starts on another date.')
    require(audit.training_months.min() == 43, 'First common fit should contain43 months.')
    require(audit.last_training_exit.le(audit.signal).all() and audit.last_input_week.le(audit.signal).all(), 'Future information in window sensitivity.')
    require(audit.groupby('signal').training_months.nunique().eq(1).all(), 'Window variants use unequal training histories.')
    for row in metrics.itertuples(index=False):
        frame = predictions.loc[predictions.window.eq(row.window) & predictions.model.eq(row.model) & predictions.split.eq(row.split)]
        actual = frame.pivot(index='signal', columns='factor', values='actual_return').reindex(columns=FACTORS)
        forecast = frame.pivot(index='signal', columns='factor', values='predicted_return').reindex(index=actual.index, columns=FACTORS)
        expected = reference.loc[reference.split.eq(row.split), 'signal']
        require(actual.index.equals(pd.DatetimeIndex(expected)), 'Window evaluation dates differ.')
        error = forecast.to_numpy() - actual.to_numpy()
        close(row.mae_pp, np.abs(error).mean() * 100, 'Window MAE differs')
        close(row.direction_accuracy_pct, ((forecast.to_numpy() > 0) == (actual.to_numpy() > 0)).mean() * 100, 'Window direction accuracy differs')
        close(row.best_factor_hit_rate_pct, (forecast.to_numpy().argmax(axis=1) == actual.to_numpy().argmax(axis=1)).mean() * 100, 'Window factor hit rate differs')
    return 'All three return/risk definitions, three models, common training history and identical 23/21 forecast dates verified.'


record('Forecast grid, target dates and baseline reproduction', check_dimensions_and_reproduction)
record('Training timing, bridge and early stopping', check_training_and_bridge)
record('Past-only six-month model selection', check_selection)
record('Factor mixes, fixed controls and stock-level netting', check_weights_and_controls)
record('Continuous shadow accounting', lambda: check_ledger(RESULTS / 'shadow', shadow=True))
record('Evaluation accounting across all seeds', lambda: check_ledger(RESULTS / 'seed_runs'))
record('Unchanged M6 portfolio results', check_m6_portfolio_reproduction)
record('Bootstrap, costs and seed coverage', check_uncertainty_and_costs)
record('Matched return-window sensitivity', check_window_sensitivity)

status = 'PASS' if all(row['status'] == 'PASS' for row in checks) else 'FAIL'
report = {'status': status, 'checks': checks, 'limitations': 'This audit checks saved calculations and timing. It does not establish point-in-time source completeness, historical borrow availability or independent out-of-sample evidence.'}
destination = RESULTS / 'verification'
destination.mkdir(exist_ok=True)
(destination / 'calculation_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
lines = [row['status'] + ': ' + row['check'] + ' — ' + row['detail'] for row in checks]
(destination / 'calculation_audit.txt').write_text('\n'.join(lines) + '\n\n' + report['limitations'] + '\n', encoding='utf-8')
print('\n'.join(lines))
sys.exit(0 if status == 'PASS' else 1)
