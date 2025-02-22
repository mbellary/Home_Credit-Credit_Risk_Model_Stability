import polars as pl

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
        return  expr_max +expr_last + expr_mean

    def str_expr():
        expr_max = [pl.all().exclude('^.*[^M]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^M]$').last().name.prefix("last_")]
        return expr_max + expr_last

    def other_expr():
        expr_max = [pl.all().exclude('^.*[^T|L]$').max().name.prefix("max_")]
        expr_last = [pl.all().exclude('^.*[^T|L]$').last().name.prefix("last_")]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

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