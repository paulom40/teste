from dash import Dash, html, dcc, Input, Output
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.read_excel("VComerciais.xlsx")
df["Ano"] = pd.to_numeric(df["Ano"], errors='coerce')
print(df["Ano"])
# criando o gráfico
fig = px.bar(df, x="Vendedor", y="Valor", color="Vendedor", barmode="group")
opcoes = list(df['Vendedor'].unique())
opcoes.append("Todos os Vendedores")


app.layout = html.Div(children=[
    html.H1(children='Faturamento das Vendedores'),
    html.H2(children='Gráfico com o Faturamento de Todos os Produtos separados por Vendedor'),
    html.Div(children='''
        Obs: Esse gráfico mostra o faturamento.
    '''),

    dcc.Dropdown(
    id='Vendedor',
    options=[{'label': vendedor, 'value': vendedor} for vendedor in df['Vendedor'].unique()],
    value=[],  # Start with no selection or preselect a list like ['Ana', 'Carlos']
    multi=True
),
dcc.Dropdown(
    id='Mes',
    options=[{'label': mes, 'value': mes} for mes in df['Mês'].unique()],
    value=[],
    multi=True
),
dcc.Dropdown(
    id='Ano',
    options=[{'label': int(ano), 'value': int(ano)} for ano in df['Ano'].dropna().unique()],
    value=[],
    multi=True
),




    dcc.Graph(
        id='grafico_quantidade_vendas',
        figure=fig
    )
])

@app.callback(
    Output('grafico_quantidade_vendas', 'figure'),
    [Input('Vendedor', 'value'),
     Input('Mes', 'value'),
     Input('Ano', 'value')]
)
def update_output(vendedores, meses, anos):
    tabela_filtrada = df.copy()

    if vendedores:
        tabela_filtrada = tabela_filtrada[tabela_filtrada['Vendedor'].isin(vendedores)]
    if meses:
        tabela_filtrada = tabela_filtrada[tabela_filtrada['Mês'].isin(meses)]
    if anos:
        tabela_filtrada = tabela_filtrada[tabela_filtrada['Ano'].isin(anos)]

    if tabela_filtrada.empty:
        fig = px.bar(title="Sem dados para os filtros selecionados.")
    else:
        fig = px.bar(tabela_filtrada, x="Mês", y="Valor", color="Vendedor", barmode="group", text="Valor")

    return fig



if __name__ == '__main__':
    app.run(debug=True)

