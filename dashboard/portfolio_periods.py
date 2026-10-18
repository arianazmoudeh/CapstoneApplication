import hashlib
import math
from decimal import Decimal, InvalidOperation, ROUND_FLOOR

import numpy as np
import pandas as pd

from .china_factor_study import RESULTS, FactorStudyError
from .research import chart_data


def calculate_period(prices, tradable, target, entry, end, initial, cost_rate):
    calendar = prices.index
    entry_position = calendar.get_loc(entry)
    if entry_position == 0 or end not in calendar or not entry < end:
        raise ValueError('Invalid holding period')
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError('Invalid trading calendar')
    target = target.reindex(prices.columns, fill_value=0)
    if not np.isfinite(target).all() or target.lt(0).any() or not np.isclose(target.sum(), 1):
        raise ValueError('Invalid stock allocation')
    if not math.isfinite(initial) or initial <= 0 or not 0 <= cost_rate < 1:
        raise ValueError('Invalid investment settings')
    base = calendar[entry_position - 1]
    dates = calendar[(calendar >= base) & (calendar <= end)]
    available = tradable.loc[entry].reindex(target.index, fill_value=False)
    unavailable = target.loc[target.gt(0) & ~available].index.tolist()
    target = target.where(available, 0)
    allocation = target / target.sum() if target.sum() > 0 else target * 0
    invested = initial / (1 + cost_rate) if allocation.sum() > 0 else 0
    costs = invested * cost_rate
    cash = max(initial - invested - costs, 0)
    held = allocation.loc[allocation.gt(0)].index
    relative = prices.loc[entry:end, held].div(prices.loc[entry, held])
    if not np.isfinite(relative.to_numpy()).all() or relative.le(0).any().any():
        raise ValueError('A held stock has no valid valuation price')
    values = relative.dot(allocation.loc[held]) * invested + cash
    values.loc[base] = initial
    values = values.reindex(dates)
    if not np.isfinite(values).all() or values.le(0).any():
        raise ValueError('Invalid portfolio values')
    return values, {
        'base': base, 'initial_value': initial, 'final_value': values.iloc[-1],
        'profit_loss': values.iloc[-1] - initial, 'total_return': (values.iloc[-1] / initial - 1) * 100,
        'drawdown': (values / values.cummax() - 1).min() * 100, 'costs': costs,
        'entry_cash': cash, 'entry_value': initial - costs, 'cost_rate': cost_rate,
        'company_count': len(held), 'unavailable_target_count': len(unavailable),
        'unavailable_codes': '|'.join(unavailable),
    }


def period_context(snapshot, budget):
    try:
        budget = Decimal(budget)
        if not budget.is_finite() or budget <= 0 or budget > Decimal('1000000000') or budget != budget.quantize(Decimal('0.01')):
            raise ValueError('Invalid portfolio budget')
        signal = pd.Timestamp(snapshot['signal'])
        model = str(snapshot['model'])
        names = ['Eligible equal stocks', 'Equal factors', model]
        if len(set(names)) != 3:
            raise ValueError('Invalid selected model')
        summary = pd.read_csv(RESULTS / 'data_summary.csv', keep_default_na=False).set_index('item').value
        initial = float(summary['initial_value'])
        fee = float(summary['cost_rate'])
        if not math.isfinite(initial) or initial <= 0 or not 0 <= fee < 1:
            raise ValueError('Invalid trading settings')
        predictions = pd.read_csv(RESULTS / 'predictions.csv', parse_dates=['signal', 'entry', 'exit'])
        if 'split' in predictions:
            predictions = predictions.loc[predictions.split.eq('evaluation')]
        forecasts = predictions.loc[predictions.signal.eq(signal) & predictions.model.eq(model)]
        if len(forecasts) != 3 or forecasts.factor.nunique() != 3 or forecasts.entry.nunique() != 1 or forecasts.exit.nunique() != 1:
            raise ValueError('The saved month is not available')
        entry = forecasts.entry.iloc[0]
        end = forecasts.exit.iloc[0]
        if entry != pd.Timestamp(snapshot['entry']) or end != pd.Timestamp(snapshot['exit']) or not signal < entry < end:
            raise ValueError('The saved portfolio dates have changed')
        results = pd.read_csv(RESULTS / 'period_results.csv', parse_dates=['signal', 'base', 'entry', 'exit'], keep_default_na=False, float_precision='round_trip')
        values = pd.read_csv(RESULTS / 'period_values.csv', parse_dates=['signal', 'date'], float_precision='round_trip')
        weights = pd.read_csv(RESULTS / 'stock_weights.csv', dtype={'code': str}, parse_dates=['signal'], float_precision='round_trip')
        if results.duplicated(['signal', 'strategy']).any() or values.duplicated(['signal', 'strategy', 'date']).any():
            raise ValueError('Duplicate portfolio results')
        rows = results.loc[results.signal.eq(signal) & results.strategy.isin(names)].set_index('strategy')
        if set(rows.index) != set(names):
            raise ValueError('A comparison portfolio is missing')
        numeric = ['initial_value', 'final_value', 'profit_loss', 'total_return', 'drawdown', 'costs', 'entry_cash', 'entry_value', 'cost_rate', 'company_count', 'unavailable_target_count']
        rows[numeric] = rows[numeric].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(rows[numeric].to_numpy()).all():
            raise ValueError('Invalid portfolio results')
        if not rows.base.eq(signal).all() or not rows.entry.eq(entry).all() or not rows.exit.eq(end).all():
            raise ValueError('The result dates do not match the saved portfolio')
        calendar = pd.DatetimeIndex(pd.read_csv(RESULTS / 'portfolio_values.csv', usecols=['date'], parse_dates=['date']).date)
        dates = calendar[(calendar >= signal) & (calendar <= end)]
        if len(dates) < 3 or not dates.is_unique or not dates.is_monotonic_increasing or dates[0] != signal or dates[1] != entry or dates[-1] != end:
            raise ValueError('The holding period does not match the study calendar')
        curves = {}
        comparison_rows = []
        selected_weights = None
        selected_unavailable = []
        for name in names:
            result = rows.loc[name]
            source = weights.loc[weights.signal.eq(signal) & weights.model.eq(name)].sort_values('code')
            if source.empty or source.code.duplicated().any() or not np.isfinite(source.weight).all() or source.weight.le(0).any() or not np.isclose(source.weight.sum(), 1):
                raise ValueError('Invalid source stock weights')
            text = '\n'.join(f'{row.code}:{float(row.weight).hex()}' for row in source.itertuples())
            if hashlib.sha256(text.encode()).hexdigest() != result.weight_signature:
                raise ValueError('The source stock weights have changed')
            unavailable = result.unavailable_codes.split('|') if result.unavailable_codes else []
            if len(set(unavailable)) != len(unavailable) or not set(unavailable).issubset(source.code):
                raise ValueError('Invalid unavailable companies')
            if result.unavailable_target_count != len(unavailable) or result.company_count != len(source) - len(unavailable):
                raise ValueError('The purchased company count does not match')
            expected_cost = initial / (1 + fee) * fee if result.company_count > 0 else 0
            expected_cash = 0 if result.company_count > 0 else initial
            if not np.isclose(result.initial_value, initial, rtol=0, atol=1e-7) or not np.isclose(result.cost_rate, fee, rtol=0, atol=1e-12):
                raise ValueError('The initial value or cost rate has changed')
            if not np.isclose(result.costs, expected_cost, rtol=0, atol=1e-7) or not np.isclose(result.entry_cash, expected_cash, rtol=0, atol=1e-7):
                raise ValueError('The purchase costs do not match')
            curve = values.loc[values.signal.eq(signal) & values.strategy.eq(name)].set_index('date').value
            if not curve.index.equals(dates) or not np.isfinite(curve).all() or curve.le(0).any():
                raise ValueError('The daily portfolio values are incomplete')
            checks = [
                (curve.iloc[0], initial), (curve.loc[entry], initial - expected_cost),
                (result.entry_value, initial - expected_cost), (curve.iloc[-1], result.final_value),
                (curve.iloc[-1] - initial, result.profit_loss),
                ((curve.iloc[-1] / initial - 1) * 100, result.total_return),
                ((curve / curve.cummax() - 1).min() * 100, result.drawdown),
            ]
            if any(not np.isclose(actual, expected, rtol=1e-10, atol=1e-7) for actual, expected in checks):
                raise ValueError('The period totals do not match the daily values')
            scale = float(budget) / initial
            curves[name] = curve * scale
            comparison_rows.append({
                'name': name, 'initial_value': float(budget), 'final_value': result.final_value * scale,
                'profit_loss': result.profit_loss * scale, 'total_return': float(result.total_return),
                'drawdown': float(result.drawdown), 'costs': result.costs * scale,
                'entry_cash': result.entry_cash * scale, 'entry_value': result.entry_value * scale,
                'company_count': int(result.company_count), 'unavailable_target_count': len(unavailable),
                'unavailable_codes': unavailable,
            })
            if name == model:
                selected_weights = source.set_index('code').weight
                selected_unavailable = unavailable
        saved = pd.Series({str(row['code']): float(row['weight']) / 100 for row in snapshot['holdings']})
        if len(saved) != len(snapshot['holdings']) or set(saved.index) != set(selected_weights.index):
            raise ValueError('The saved companies do not match the research month')
        if not np.allclose(saved.reindex(selected_weights.index), selected_weights, rtol=1e-10, atol=1e-10):
            raise ValueError('The saved allocation does not match the research month')
        frame = pd.DataFrame(curves)
        chart = chart_data(frame)
        indices = np.unique(np.linspace(0, len(frame) - 1, 4).astype(int))
        for label, index in zip(chart['labels'], indices):
            label['label'] = frame.index[index].strftime('%d %b')
        if float(budget) < 100:
            low = float(frame.min().min())
            high = float(frame.max().max())
            padding = max((high - low) * 0.1, 1)
            low, high = max(0, low - padding), high + padding
            for index, tick in enumerate(chart['ticks']):
                tick['label'] = f'{low + index / 4 * (high - low):,.2f}'
        stocks, factors, selected = comparison_rows
        held = selected_weights.drop(selected_unavailable)
        actual_weights = held / held.sum() if len(held) else held
        invested = budget - Decimal(str(selected['costs'])).quantize(Decimal('0.01')) - Decimal(str(selected['entry_cash'])).quantize(Decimal('0.01'))
        held_rows = list(actual_weights.items())
        exact_cents = [Decimal(str(weight)) / sum(Decimal(str(value)) for _, value in held_rows) * invested * 100 for _, weight in held_rows]
        cents = [int(value.to_integral_value(rounding=ROUND_FLOOR)) for value in exact_cents]
        order = sorted(range(len(held_rows)), key=lambda index: exact_cents[index] - cents[index], reverse=True)
        for index in order[:int(invested * 100) - sum(cents)]:
            cents[index] += 1
        amounts = {code: Decimal(cents[index]).scaleb(-2) for index, (code, _) in enumerate(held_rows)}
        executed_holdings = []
        for holding in snapshot['holdings']:
            code = str(holding['code'])
            weight = float(actual_weights.get(code, 0))
            executed_holdings.append({**holding, 'target_weight': float(holding['weight']), 'weight': weight * 100,
                'amount': amounts.get(code, Decimal('0.00')), 'purchased': code in actual_weights.index})
        return {
            'start': entry.date(), 'entry': entry.date(), 'end': end.date(), 'exit': end.date(), 'base': signal.date(),
            'rows': comparison_rows, 'selected': selected, 'stocks': stocks, 'factors': factors, 'chart': chart,
            'cost_rate': fee * 100, 'budget': budget, 'holdings': executed_holdings,
            'gains_vs_stocks': {'amount': selected['final_value'] - stocks['final_value'], 'percentage_points': selected['total_return'] - stocks['total_return']},
            'gains_vs_factors': {'amount': selected['final_value'] - factors['final_value'], 'percentage_points': selected['total_return'] - factors['total_return']},
        }
    except (OSError, KeyError, TypeError, ValueError, InvalidOperation, OverflowError, AttributeError, pd.errors.ParserError) as exc:
        raise FactorStudyError('The results for this saved portfolio could not be loaded. Please create it again from Research.') from exc
