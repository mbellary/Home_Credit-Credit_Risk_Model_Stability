import polars as pl

class Processor:

    def set_table_dtypes(df):
        df_columns = df.collect_schema().names()
        for col in df_columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  # !!?
                df = df.with_columns(pl.col(col).dt.total_days()) # t - t-1
        df = df.drop("date_decision", "MONTH")
        return df

    def filter_cols(df):
        df_schema = df.collect_schema()
        for col in df_schema.names():
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df.select(pl.col(col).is_null().mean()).collect().item()
                if isnull > 0.7:
                    df = df.drop(col)

        for col in df_schema.names():
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df_schema[col] == "String"):
                freq = dt.select(pl.col(col)).collect().n_unique(subset=[col])
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df

    def set_categorical_dtypes(df):
        df = df.with_columns(pl.String).cast(pl.Categorical)
        return df
