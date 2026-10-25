from pathlib import Path
import ast
import hashlib
import json
import shutil
import nbformat
import numpy as np
import pandas as pd

root=Path(__file__).resolve().parents[1]
result=root/'results/china_factor_timing_final'
path=root/'experiments/China_A_Share_Final_Factor_Timing.ipynb'
nb=nbformat.read(path,4)
if any(o.output_type=='error' for c in nb.cells if c.cell_type=='code' for o in c.outputs):
    raise RuntimeError('Resolve notebook errors before writing findings.')
if any(c.execution_count is None for c in nb.cells if c.cell_type=='code'):
    raise RuntimeError('Run all notebook code before writing findings.')
p=pd.read_csv(result/'performance.csv').set_index('strategy')
s=pd.read_csv(result/'selection.csv',parse_dates=['selection_date','execution_date'])
u=pd.read_csv(result/'uncertainty.csv').set_index('block_length')
seeds=pd.read_csv(result/'seed_results.csv')
windows=pd.read_csv(result/'window_sensitivity.csv')
cost=pd.read_csv(result/'cost_sensitivity.csv')
allpred=pd.read_csv(result/'predictions_all.csv',parse_dates=['signal'])
weights=pd.read_csv(result/'factor_weights.csv',parse_dates=['signal'])
selected_weights=weights.loc[weights.strategy.eq('Six-month selector')].set_index('signal')
selected_predictions=[]
for signal,row in selected_weights.iterrows():
    frame=allpred.loc[allpred.signal.eq(signal)&allpred.model.eq(row.selected_model)].copy()
    frame['model']='Six-month selector'
    selected_predictions.append(frame)
selected_predictions=pd.concat(selected_predictions,ignore_index=True)
selected_predictions.to_csv(result/'selector_predictions.csv',index=False)
actual=selected_predictions.pivot(index='signal',columns='factor',values='actual_return').reindex(columns=['Value','Profitability','Momentum'])
forecast=selected_predictions.pivot(index='signal',columns='factor',values='predicted_return').reindex(index=actual.index,columns=actual.columns)
selector_forecast={'model':'Six-month selector','split':'evaluation','months':len(actual),
    'mae_pp':float(np.abs(forecast.to_numpy()-actual.to_numpy()).mean()*100),
    'direction_accuracy_pct':float(((forecast.to_numpy()>0)==(actual.to_numpy()>0)).mean()*100),
    'best_factor_hit_rate_pct':float((forecast.to_numpy().argmax(axis=1)==actual.to_numpy().argmax(axis=1)).mean()*100)}
metrics=pd.read_csv(result/'forecast_metrics.csv')
metrics=metrics.loc[metrics.model.ne('Six-month selector')]
metrics=pd.concat([metrics,pd.DataFrame([selector_forecast])],ignore_index=True)
metrics.to_csv(result/'forecast_metrics.csv',index=False)
main=p.loc['Six-month selector']
equal=p.loc['Equal factors']
locked=p.loc['Initial model family']
mix=p.loc['Initial factor mix']
gap=main.annual_return-equal.annual_return
rankchanges=int(selected_weights[['Value','Profitability','Momentum']].diff().abs().max(axis=1).gt(1e-12).sum())
choices='; '.join(row.selection_date.strftime('%B %Y')+': '+row.model for row in s.itertuples())
selected_summary=(f"We selected {choices}. Each decision used eighteen completed past periods. "
    f"The factor mix changed in {rankchanges} of the twenty monthly transitions. We charged the stock trades caused by these changes.")
main_summary=(f"The six-month selector returned {main.annual_return:.2f}% a year after costs. Equal factor weights returned {equal.annual_return:.2f}%. "
    f"The difference was {gap:+.2f} percentage points a year. Starting with CNY 100,000, the selector ended at CNY {main.final_value:,.0f} and equal factors at CNY {equal.final_value:,.0f}.")
control_summary=(f"Keeping the first selected model family returned {locked.annual_return:.2f}% a year. Keeping its first factor mix returned {mix.annual_return:.2f}%. "
    f"The selector's differences were {main.annual_return-locked.annual_return:+.2f} and {main.annual_return-mix.annual_return:+.2f} percentage points. "
    "These controls keep the company baskets changing each month. They separate a changing family from a changing factor mix.")
risk_summary=(f"The selector's maximum drawdown was {abs(main.drawdown):.2f}%, compared with {abs(equal.drawdown):.2f}% for equal factors. "
    f"Its Sharpe ratio was {main.sharpe:.2f}, compared with {equal.sharpe:.2f}. Its trading fees were CNY {main.trading_costs:,.0f} and borrowing fees were CNY {main.borrow_costs:,.0f}.")
interval=u.loc[3]
uncertainty_summary=(f"The mean monthly return difference was {interval.mean_difference:+.3f} percentage points. The 95% stationary-bootstrap interval was "
    f"{interval.lower:+.3f} to {interval.upper:+.3f} percentage points, using an expected block length of three months. "
    + ("This interval includes zero, so these months do not give a precise positive or negative average advantage. " if interval.lower<=0<=interval.upper else "This conditional interval stays on one side of zero. ")
    +"The calculation uses 21 paired months and does not undo earlier research choices.")
seed_summary=(f"Across the five training seeds, the selector's annual return ranged from {seeds.annual_return.min():.2f}% to {seeds.annual_return.max():.2f}%, "
    f"with a median of {seeds.annual_return.median():.2f}%. We keep seed 42 as the main result. These are training checks on the same market history, not five independent market tests.")
forecast_summary=(f"The selector's mean absolute forecast error was {selector_forecast['mae_pp']:.2f} percentage points. It predicted the factor direction correctly in "
    f"{selector_forecast['direction_accuracy_pct']:.2f}% of the 63 forecasts and chose the best factor in {selector_forecast['best_factor_hit_rate_pct']:.2f}% of the 21 months. "
    "Forecast accuracy and net portfolio return answer different questions.")
best=p.loc[[x for x in p.index if x not in ['Equal factors','Initial model family','Initial factor mix','Six-month selector','Historical mean']],'annual_return'].idxmax()
earlier_summary=(f"The original M6 rule selected Prophet before the evaluation, using the validation best-factor hit rate. Its later annual return remains {p.loc['Prophet','annual_return']:.2f}%. "
    f"{best} had the highest individual model return at {p.loc[best,'annual_return']:.2f}%, but this later ranking was not used to choose the original model. We report both facts.")
conclusion=("Our main question is whether using forecasts to change factor weights helps after costs. "
    + (f"The recorded six-month selection rule earned {gap:.2f} percentage points more annual return than equal factors in this sample. " if gap>0 else f"The recorded six-month selection rule earned {abs(gap):.2f} percentage points less annual return than equal factors in this sample. ")
    +f"The original Prophet-selected strategy returned {p.loc['Prophet','annual_return']:.2f}% a year. The two selection rules therefore need to be read separately. "
    +"The six-month selector earned less than keeping the first model family and less than keeping its first factor mix. Changing model families did not improve on those controls in this sample. "
    +"We found that the selection rule, initial factor mix and trading costs matter when we turn forecasts into a portfolio. We do not conclude that machine learning always improves investing. These findings describe the selected A-share group, the stated data and 21 completed monthly periods.")
limitations=[
    'These dates were already examined in M6. This is a retrospective supplementary test, not a new unseen evaluation.',
    'The selected company list can favour surviving firms. Of 839 usable financial records, 835 have a later provider update. A delay rule cannot establish fully point-in-time accounts.',
    'The research portfolios assume shares can be borrowed. We include borrowing fees but do not model historical loan availability, recalls, price limits, collateral rules or market impact.',
    'Only 21 monthly periods are evaluated. The bootstrap intervals are conditional on these results and do not correct the complete history of earlier model trials.',
    'EBIT divided by cash-net invested capital can be large when capital is small. Our five-report sample check confirms amounts for selected records, not every field in the full data set.',
]
models=['Ridge','Elastic Net','KNN','SVR','Random Forest','Extra Trees','XGBoost','Decision Tree','AdaBoost','LightGBM','MLP','LSTM','GRU','CNN','Prophet']
summary={'status':'complete','protocol_version':'2','evaluation_start':'2025-01-02','evaluation_end':'2026-10-08','period_count':21,
    'company_count':100,'eligible_min':90,'eligible_max':94,'model_count':15,'models':models,'initial_value':100000,
    'primary_strategy':'Six-month selector','selected_before_test':'Prophet',
    'comparison_strategies':['Six-month selector','Equal factors','Initial model family','Initial factor mix','Prophet','Historical mean'],
    'notebook_filename':path.name,'findings':[main_summary,control_summary,risk_summary,uncertainty_summary],
    'limitations':limitations,'conclusion':conclusion,'selection_description':selected_summary,
    'seed_description':seed_summary,'forecast_description':forecast_summary,'earlier_result':earlier_summary}
(result/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(result/'findings.txt').write_text('\n\n'.join(['China A-share factor timing results','Completed 25 October 2026','Research question',conclusion,
    'Main portfolio result',main_summary,'Selected models',selected_summary,'Fixed controls',control_summary,'Risk and costs',risk_summary,
    'Uncertainty',uncertainty_summary,'Training seeds',seed_summary,'Forecasts',forecast_summary,'Original M6 result',earlier_summary,
    'How we read these findings',*limitations])+'\n',encoding='utf-8')

old=nbformat.read(root/'experiments/China_A_Share_Long_Short_Factor_Timing_M6.ipynb',4)
settings_rows=[]
for index,name in zip([15,17,19,21,23,25,27,29,31,33,35,37,39,41,43],models):
    tree=ast.parse(old.cells[index].source)
    values=[]
    for node in tree.body:
        if isinstance(node,ast.For):
            break
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            key=node.targets[0].id
            try:
                value=ast.literal_eval(node.value)
            except (ValueError,TypeError):
                continue
            if not key.endswith(('_rows','_warnings')) and value not in [[],{}]:
                values.append(key+'='+str(value))
    estimators=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ['Ridge','ElasticNet','KNeighborsRegressor','SVR','RandomForestRegressor','ExtraTreesRegressor','XGBRegressor','DecisionTreeRegressor','AdaBoostRegressor','LGBMRegressor','MLPRegressor','Prophet']:
            estimators.append(ast.unparse(node))
    settings_rows.append({'model':name,'settings':'; '.join(values),'estimator':'; '.join(estimators),
        'inputs':'Completed monthly factor returns' if name=='Prophet' else '8 weeks x 13 measures',
        'refit':'Monthly on completed earlier outcomes'})
pd.DataFrame(settings_rows).to_csv(result/'model_settings.csv',index=False)
shutil.copy2(root/'results/china_factor_timing_m6/input_comparison_metrics.csv',result/'prior_input_length_comparison.csv')
inventory=[]
for item in sorted((root/'experiments').glob('*.ipynb')):
    trial=nbformat.read(item,4)
    inventory.append({'notebook':item.name,'code_cells':sum(c.cell_type=='code' for c in trial.cells),
        'saved_executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in trial.cells),
        'sha256':hashlib.sha256(item.read_bytes()).hexdigest()})
pd.DataFrame(inventory).to_csv(result/'trial_inventory.csv',index=False)

replacements={
    'We select a model after the June': selected_summary,
    'Our main comparison is the difference': uncertainty_summary+'\n\n'+seed_summary,
    'We compare portfolio value': main_summary+'\n\n'+risk_summary,
}
nb.cells=[c for c in nb.cells if not c.metadata.get('final_findings')]
for prefix,text in replacements.items():
    target=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith(prefix))
    item=nbformat.v4.new_markdown_cell(text)
    item.metadata['final_findings']=True
    nb.cells.insert(target,item)
end=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('The financial data may'))
item=nbformat.v4.new_markdown_cell('What did we find?\n\n'+conclusion+'\n\n'+control_summary+'\n\n'+forecast_summary+'\n\n'+earlier_summary)
item.metadata['final_findings']=True
nb.cells.insert(end,item)
nbformat.validate(nb)
nbformat.write(nb,path)
trial_inventory = pd.read_csv(result/'trial_inventory.csv')
trial_inventory.loc[trial_inventory.notebook.eq(path.name),'sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
trial_inventory.to_csv(result/'trial_inventory.csv',index=False)

bundle=root/'dashboard/china_final_results'
bundle.mkdir(parents=True,exist_ok=True)
for name in ['summary.json','performance.csv','portfolio_values.csv','selection.csv','factor_weights.csv','uncertainty.csv','seed_results.csv','cost_sensitivity.csv','forecast_metrics.csv']:
    shutil.copy2(result/name,bundle/name)
print(main_summary)
print(control_summary)
print(uncertainty_summary)
print(seed_summary)
print(selected_summary)
print('Bundle:',bundle)
