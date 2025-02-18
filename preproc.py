# Credit:
# This work is based on the top Kernel from the competition : https://www.kaggle.com/code/majiaqi111/home-credit-lgb-cat-ensemble


import polars as pl
import argparse
from pathlib import Path

class Pipeline:

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
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))  #!!?
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

class Aggregator:    
    def num_expr():
        expr_max = [pl.all().exclude('^.*[^P|A]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^P|A]$').last().name.prefix("last_")]
        expr_mean = [pl.all().exclude('^.*[^P|A]$').mean().name.prefix("mean_")]
        return expr_max + expr_last + expr_mean
    
    def date_expr():
        expr_max = [pl.all().exclude('^.*[^D]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^D]$').last().name.prefix("last_")]
        expr_mean = [pl.all().exclude('^.*[^D]$').mean().name.prefix("mean_")]
        return  expr_max +expr_last+expr_mean
    
    def str_expr():
        expr_max = [pl.all().exclude('^.*[^M]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^M]$').last().name.prefix("last_")]
        return  expr_max +expr_last
    
    def other_expr():
        expr_max = [pl.all().exclude('^.*[^T|L]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^T|L]$').last().name.prefix("last_")]
        return  expr_max +expr_last
    
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols] 
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return  expr_max +expr_last
    
    def get_exprs():
        exprs = Aggregator.num_expr() + \
                Aggregator.date_expr() + \
                Aggregator.str_expr() + \
                Aggregator.other_expr()
                # Aggregator.count_expr(df)
        return exprs

def merge_datastores(df, data_stores):
    for data_store in data_stores:
        df = df.join(data_store, on='case_id', how='left', coalesce=True)
    return df

def select_num_features(df):
    # feature selection only for categorical columns
    columns = df.columns
    corr_col_groups = []
    for i, this in enumerate(columns):
        for other in columns[i:]:
            df = df.with_columns(pl.corr(this, other).alias(f'corr_{this}_{other}')).with_columns((pl.col(f'corr_{this}_{other}') > 0.8).mean().alias(f'corr_mean_{this}_{other}'))
            rank = df.select(pl.col(f'corr_mean_{this}_{other}').first()).collect().item()
            if rank == 0:
                df = df.drop([f'corr_{this}_{other}', f'corr_mean_{this}_{other}'])
            elif this != other:
                corr_col_groups.append((this, other))
        df = df.drop([f'corr_{this}_{this}', f'corr_mean_{this}_{this}'])
    
    col1, col2 = zip(*corr_col_groups)
    sel_cols = list(col1 + col2)
    for col1, col2 in corr_col_groups:
        df_uncorr = (df.with_columns(pl.col(col1, col2).n_unique().name.suffix('_nunique'))
                .select(pl.when(pl.col(f'{col1}_nunique') > pl.col(f'{col2}_nunique')).then(col2))
             )
        sel_cols.remove(df_uncorr.columns[0])
        df = df.drop(df_uncorr.columns)
    cols_remove = df.select(pl.exclude(sel_cols))
    df = df.drop(cols_remove.columns)
    return df


def preprocessor(dirpath):
	start_msg = "Reading data files.."
	end_msg = "completed processing data files"
	print(start_msg)
	TRAIN_DIR = Path(dirpath)
    data_store = {
        "df_base" : pl.scan_parquet(TRAIN_DIR / "train_base.parquet").pipe(Pipeline.set_table_dtypes).pipe(Pipeline.set_categorical_dtypes),
        "depth_0" : [
            pl.scan_parquet(TRAIN_DIR / "train_static_cb_0.parquet").pipe(Pipeline.set_table_dtypes).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_static_0_*.parquet").pipe(Pipeline.set_table_dtypes).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes)        
        ],
        "depth_1" : [
            pl.scan_parquet(TRAIN_DIR / "train_applprev_1_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_tax_registry_a_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_tax_registry_b_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_tax_registry_c_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_b_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_other_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_person_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_deposit_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_debitcard_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
        ],
        "depth_2" : [
            pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_b_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_applprev_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes),
            pl.scan_parquet(TRAIN_DIR / "train_person_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Pipeline.filter_cols).pipe(Pipeline.set_categorical_dtypes)
        ]
    }
	print(end_msg)
    data_stores = data_store['depth_0'] + data_store['depth_1'] + data_store['depth_2']
    train_df = data_store['df_base'].pipe(merge_datastores, data_stores=data_stores)
	print(train_df)
    return train_df
	



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess Home Creadit CRMS training data')
	parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
	args = parser.parse_args()
	preprocessor(args.dir_path)