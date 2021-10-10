import os
from pathlib import Path

import pandas as pd

WORK_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[0]
DATA_DIR = WORK_DIR / 'data'

data = pd.read_csv(DATA_DIR / 'telecom_users.csv')
data.drop(columns=['Unnamed: 0', 'IDCliente', 'Codigo'], inplace=True)
data['TotalGasto'] = pd.to_numeric(data.TotalGasto, errors='coerce')
data['MesesComoCliente'] = data['MesesComoCliente'].astype(float)
data.dropna(inplace=True)

pd.get_dummies(data, columns=['Genero', 'Casado', 'Aposentado', 'Dependentes', 'ServicoTelefone', 'MultiplasLinhas', 'ServicoInternet', 'ServicoSegurancaOnline', 'ServicoBackupOnline', 'ProtecaoEquipamento', 'ServicoSuporteTecnico', 'ServicoStreamingTV', 'ServicoFilmes', 'TipoContrato', 'FaturaDigital', 'FormaPagamento', 'Churn'], drop_first=True, dtype=bool).to_csv(DATA_DIR / 'clean_telecom_users.csv', index=False)
