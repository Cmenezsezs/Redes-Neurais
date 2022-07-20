import plotly.express as px
import pandas as pd

centrais = ("GUNNING1", "MUSSELR1", "WATERLWF")


for i, central in enumerate(centrais):
    if i == 0:
        df = pd.read_csv(f"Dados/Potência_Horaria_{central}.csv", index_col=0,
                         parse_dates=True)
    else:
        df = pd.concat([df, pd.read_csv(f"Dados/Potência_Horaria_{central}.csv",
                                        index_col=0, parse_dates=True)], axis=1)
    

df = df.last("M")
fig = px.line(df, x=df.index, y=df.columns, labels={
    'value': "Potência [MW]",
    'variable': "Centrais",
    'index': 'Timestamps [x30 minutos]'
}, title="Série temporal de potência no ano de 2018")
fig.show()