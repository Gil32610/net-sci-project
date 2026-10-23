from __future__ import annotations

import pathlib
from collections.abc import Iterator

import numpy as np
import pandas as pd
from mlex import DataReader

ACCOUNT_GROUP = ["CONTA_TITULAR", "TIPO"]
TIMESTAMP_COLUMN = "DATA_LANCAMENTO"

DATA_DIR = pathlib.Path("data")
DATASETS: dict[str, pathlib.Path] = {
    "pcpe_03": DATA_DIR / pathlib.Path("pcpe_03.csv"),
    "pcpe_04": DATA_DIR/ pathlib.Path("pcpe_04.csv"),
}

TARGET_COLUMN = "I-d"


def get_pcpe_dtype_dict():
    return {
        "NUMERO_CASO": "str",
        "NUMERO_BANCO": "str",
        "NOME_BANCO": "str",
        "NUMERO_AGENCIA": "str",
        "NUMERO_CONTA": "str",
        "TIPO": "str",
        "CPF_CNPJ_TITULAR": "str",
        "NOME_TITULAR": "str",
        "DATA_LANCAMENTO": "str",
        "CPF_CNPJ_OD": "str",
        "NOME_PESSOA_OD": "str",
        "CNAB": "str",
        "DESCRICAO_LANCAMENTO": "str",
        "VALOR_TRANSACAO": "float64",
        "NATUREZA_LANCAMENTO": "str",
        "I-d": "uint8",
        "I-e": "uint8",
        "IV-n": "uint8",
        "RAMO_ATIVIDADE_1": "str",
        "RAMO_ATIVIDADE_2": "str",
        "RAMO_ATIVIDADE_3": "str",
        "LOCAL_TRANSACAO": "str",
        "NUMERO_DOCUMENTO": "str",
        "NUMERO_DOCUMENTO_TRANSACAO": "str",
        "VALOR_SALDO": "float64",
        "NATUREZA_SALDO": "str",
        "NUMERO_BANCO_OD": "str",
        "NUMERO_AGENCIA_OD": "str",
        "NUMERO_CONTA_OD": "str",
        "NOME_ENDOSSANTE_CHEQUE": "str",
        "DOC_ENDOSSANTE_CHEQUE": "str",
        "DIA_LANCAMENTO": "uint8",
        "MES_LANCAMENTO": "uint8",
        "ANO_LANCAMENTO": "uint16",
    }


def pcpe_preprocessing_read_func(df):
    df["CONTA_TITULAR"] = (
        df["NUMERO_BANCO"] + "_" + df["NUMERO_AGENCIA"] + "_" + df["NUMERO_CONTA"]
    )
    df["CONTA_OD"] = (
        df["NUMERO_BANCO_OD"]
        + "_"
        + df["NUMERO_AGENCIA_OD"]
        + "_"
        + df["NUMERO_CONTA_OD"].astype(str)
    )
    df["CONTA_OD"] = df["CONTA_OD"].fillna("EMPTY")
    df.loc[df["CONTA_OD"].str.contains("0_0"), "CONTA_OD"] = "EMPTY"

    df["DATA_LANCAMENTO"] = pd.to_datetime(df["DATA_LANCAMENTO"])
    balance_sign = np.where(df["NATUREZA_SALDO"] == "D", -1, 1)
    tx_sign = np.where(df["NATUREZA_LANCAMENTO"] == "D", -1, 1)

    df["VALOR_SALDO"] = (df["VALOR_SALDO"] * 100).round() / 100.0
    df["VALOR_TRANSACAO"] = (df["VALOR_TRANSACAO"] * 100).round() / 100.0

    df["VALOR_SALDO"] *= balance_sign
    df["VALOR_TRANSACAO"] *= tx_sign
    df = df.sort_values(["DATA_LANCAMENTO"]).reset_index(drop=True)

    return df


def get_reader(path):
    return DataReader(
        path,
        target_columns=[TARGET_COLUMN],
        dtype_dict=get_pcpe_dtype_dict(),
        preprocessing_func=pcpe_preprocessing_read_func,
    )


def load_dataframe(dataset: str) -> pd.DataFrame:
    return get_reader(str(DATASETS[dataset])).read_df()


def account_key(account, category) -> str:
    return f"{account}_{category}"


def iter_timestamp_sequences(df: pd.DataFrame) -> Iterator[tuple[str, str, list]]:
    from mlex import TransactionFlow

    for (account, category), account_df in df.groupby(ACCOUNT_GROUP):
        for timestamp, day_df in account_df.groupby(TIMESTAMP_COLUMN):
            sequence = [
                TransactionFlow(
                    tx_index=row.Index,
                    balance_amount=row.VALOR_SALDO,
                    tx_value=row.VALOR_TRANSACAO,
                    timestamp=row.DATA_LANCAMENTO,
                )
                for row in day_df.itertuples()
            ]
            yield account_key(account, category), str(timestamp.date()), sequence
