# Credit:
# This work is based on the top Kernel from the competition : https://www.kaggle.com/code/majiaqi111/home-credit-lgb-cat-ensemble


import polars as pl
import argparse
from pathlib import Path

class Pipeline:

    def set_table_dtypes(df):
        for col in df.columns:
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
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.7:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
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



def main(dirpath):
	start_msg = "Reading data files.."
	end_msg = "completed processing data files"
	print(start_msg)
	TRAIN_DIR = Path(dirpath)
	data_store = {
	    "df_base" : pl.scan_parquet(TRAIN_DIR / "train_base.parquet").pipe(Pipeline.set_table_dtypes),
	    "depth_0" : [
	        pl.scan_parquet(TRAIN_DIR / "train_static_cb_0.parquet").pipe(Pipeline.set_table_dtypes),
	        pl.scan_parquet(TRAIN_DIR / "train_static_0_*.parquet").pipe(Pipeline.set_table_dtypes)        
	    ],
	    "depth_1" : [
	        pl.scan_parquet(TRAIN_DIR / "train_applprev_1_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_tax_registry_a_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_tax_registry_b_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_tax_registry_c_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_a_1_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_b_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_other_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_person_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_deposit_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_debitcard_1.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	    ],
	    "depth_2" : [
	        pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_b_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_credit_bureau_a_2_*.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_applprev_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()),
	        pl.scan_parquet(TRAIN_DIR / "train_person_2.parquet").pipe(Pipeline.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs())
	    ]
	}
	print(end_msg)
	result = [data_store['df_base']] + data_store['depth_0'] + data_store['depth_1'] + data_store['depth_2']
	print(result)
	



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Preprocess Home Creadit CRMS training data')
	parser.add_argument('--dir_path', metavar='path', required=True, help='the path to data files')
	args = parser.parse_args()
	main(args.dir_path)