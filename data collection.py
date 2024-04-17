import pandas as pd

df = pd.read_csv(r"C:\Users\HP\Downloads\Compressed\Kickstarter_2024-04-15T06_47_07_694Z\Kickstarter.csv")
df = df.drop(columns=["fx_rate","staff_pick","country","country_displayable_name","created_at","creator","currency","currency_symbol","currency_trailing_code","current_currency","deadline","disable_communication","goal","id","is_disliked","is_launched","is_liked","is_starrable","launched_at","location","name","percent_funded","photo","pledged","prelaunch_activated","profile","slug","source_url","spotlight","staff_pick","state","state_changed_at","static_usd_rate","usd_type","video"])

print(df)