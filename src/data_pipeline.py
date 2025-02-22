import polars as pl
import yaml

from process import Processor
from transform import Aggregator, merge_datastores

def pipeline():

    with open('../config/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    root_dir = config['data_stores']['root']
    df_base = root_dir + config['data_stores']['train_base']
    depth_0_0 = root_dir + config['data_stores']['train_static_cb_0']
    depth_0_1 = root_dir + config['data_stores']['train_static_0']

    depth_1_0 = root_dir + config['data_stores']['train_applprev_1']
    depth_1_1 = root_dir + config['data_stores']['train_tax_registry_a_1']
    depth_1_2 = root_dir + config['data_stores']['train_tax_registry_b_1']
    depth_1_3 = root_dir + config['data_stores']['train_tax_registry_c_1']
    depth_1_4 = root_dir + config['data_stores']['train_credit_bureau_a_1']
    depth_1_5 = root_dir + config['data_stores']['train_credit_bureau_b_1']
    depth_1_6 = root_dir + config['data_stores']['train_other_1']
    depth_1_7 = root_dir + config['data_stores']['train_person_1']
    depth_1_8 = root_dir + config['data_stores']['train_deposit_1']
    depth_1_9 = root_dir + config['data_stores']['train_debitcard_1']

    depth_2_0 = root_dir + config['data_stores']['train_credit_bureau_b_2']
    depth_2_1 = root_dir + config['data_stores']['train_credit_bureau_a_2']
    depth_2_2 = root_dir + config['data_stores']['train_applprev_2']
    depth_2_3 = root_dir + config['data_stores']['train_person_2']


    data_store = {
        "df_base": pl.scan_parquet(df_base).pipe(Processor.set_table_dtypes).pipe(Processor.set_categorical_dtypes),
        "depth_0": [
            pl.scan_parquet(depth_0_0).pipe(Processor.set_table_dtypes).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_0_1).pipe(Processor.set_table_dtypes).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes)
        ],
        "depth_1": [
            pl.scan_parquet(depth_1_0).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_1).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_2).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_3).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_4).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_5).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_6).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_7).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_8).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_1_9).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
        ],
        "depth_2": [
            pl.scan_parquet(depth_2_0).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_2_1).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_2_2).pipe(Processor.set_table_dtypes).group_by('case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes),
            pl.scan_parquet(depth_2_3).pipe(Processor.set_table_dtypes).group_by( 'case_id').agg(Aggregator.get_exprs()).pipe(Processor.filter_cols).pipe(Processor.set_categorical_dtypes)
        ]
    }
    data_stores = data_store['depth_0'] + data_store['depth_1'] + data_store['depth_2']
    train_df = data_store['df_base'].pipe(merge_datastores, data_stores=data_stores)
    return train_df
