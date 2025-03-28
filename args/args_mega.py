# Author: Bishal Shrestha
# Date: 03-24-2025  
# Description: The main configuration file where the parameters for the HiC4D-SPOT are set. The "select" is changed according to the user's need.
# For example: 
# To generate the data, select each combinations number on select (1-7), then run `python generate_data.py -id mega`
# To train the model, only the 'default' data_type is used. Select the combinations (1 or 4) and run `python train.py -id mega` to train on either Du dataset or Reed dataset.
# To predict the mode, select any combination number (1-7) and run `python predict.py -id mega` to predict on the selected dataset's data_type on the selected model.
# To evaluate the model, select any combination number (1-7) and run `python eval.py -id mega` to evaluate on the selected dataset's data_type on the selected model.
# Here, if select = 1, then the model is trained on Du dataset, uses Du dataset (default - no perturbation or simulation) for prediction and evaluation.
# Here, if select = 2, then the model is trained on Du dataset, uses Du dataset (with time-swap condition) for prediction and evaluation.


import os 

user = 'bshrestha'

combinations = { # model, data, data_type
    1: ['Du', 'Du', 'default'],
    2: ['Du', 'Du', 'time_swap'],
    3: ['Du', 'Du', 'tad_simulated'],
    4: ['Reed', 'Reed', 'default'],
    5: ['Reed', 'Reed', 'loop_simulated'],
    6: ['Reed', 'Zhang', 'default'],
    7: ['Reed', 'Cohesinloss_Rao', 'default'],
}

select = 1
model = combinations[select][0]
data = combinations[select][1]
data_type = combinations[select][2]

base_dir = f'/home/{user}/HiC4D-SPOT'

args = {
    
    'data': data,
    'data_type': data_type,
    'model': model,
    
    'resolution': 10_000,
    'num_timepoints': 8,
    'num_chr': 24,  # (eg. hg19: 24, mm9: 21)
    'chr_prefix': '',
    'cooler_balance': False,
    
    'cooler_dir': f'{base_dir}/data/data_{data}/cool_10KB',
    'data_output_dir': f'{base_dir}/data/data_{data}/input_data/{data_type}',
    'misc_output_dir': f'{base_dir}/data/data_{data}/misc_data/{data_type}',
    'rna_seq_dir': f'{base_dir}/data/data_{data}/RNA_seq',
    'rna_seq_parsed_dir': f'{base_dir}/data/data_{data}/RNA_seq_parsed',
    
    'default_input_data': f'{base_dir}/data/data_{data}/input_data/default',
    'input_data': f'{base_dir}/data/data_{data}/input_data/{data_type}',
    'output_model_dir': f'{base_dir}/models/best_models_{model}',
    
    'predict_chr': '8',
    'best_model': f'{base_dir}/models/best_models_{model}/model.pt',
    'output_predict_dir': f'{base_dir}/predictions/model_{model}/predictions_{data}_{data_type}',
    
    # Data generation
    'augmentation': False if data_type == 'default' else True,
    'aug_region_range': (93_000_000, 96_000_000),
    'aug_chr': '8' if data_type == 'loop_simulated' else '6',
    'aug_type': 'loop' if data_type == 'loop_simulated'
                else 'tad' if data_type == 'tad_simulated'
                else 'time_swap' if data_type == 'time_swap'
                else 'default',
    
    'batch_size': 32,
    'patch_size': 1,
    'epochs': 100,
    'weight_decay': 0.0001,
    'no_cuda': False,
    'GPU_index': 1,  
    'dummy': False,
    'seed': 1,
    'maxV': 100, #0.02,
    'sub_matrix_size': 50,
    'step': 25,
    'lr': 0.0001,
    'lr_patience': 5,
    'lr_min': 0.00001,
    'lr_factor': 0.5,
    'termination_patience': 10,
    
    'chrs_valid': ['chr19'],
    'chrs_test': ['chr2', 'chr6'],
    
    # Eval
    'memory_efficient': False,
    'eval_tads': False, # True if data_type == 'tad_simulated' else False,
    'eval_loops': False, #True if data_type == 'loop_simulated' else False,
    'plot_hictracks': True,
    'plot_triangular': False,
    
    'verbose': True,
}


if data == 'Du':
    args['resolution'] = 40_000
    args['cooler_balance'] = False
    args['num_timepoints'] = 6
    args['ids'] = ["PN5","early_2cell","late_2cell","8cell","ICM","mESC_500"]
    args['step'] = 3
    args['maxV'] = 100
    args['aug_region_range'] = (93_000_000, 96_000_000)
    args['cooler_dir'] = f'{base_dir}/data/data_{data}/cool_40KB'
    args['allValidPairs_dir'] = f'{base_dir}/data/data_{data}/allValidPairs'
    args['allValidPairs_downsample_dir'] = f'{base_dir}/data/data_{data}/allValidPairs_downsample'
    args['num_chr'] = 21
    args['chr_prefix'] = 'chr'
    args['chrs_valid'] = ['19']
    args['chrs_test'] = ['2', '6']    # 6 is priority
    args['predict_chr'] = '6'
    args['chrom_size_file'] = f'{base_dir}/data/data_{data}/mm9.chrom.sizes'
    args['assembly'] = 'mm9'
    
    if args['data_type'] == 'time_swap':
        args['input_data'] = f'{base_dir}/data/data_{data}/input_data/default'
    
elif data == 'Reed':
    args['resolution'] = 10_000
    args['num_timepoints'] = 8
    args['cooler_balance'] = True
    args['ids'] = ["t1","t2","t3","t4","t5","t6","t7","t8"]
    args['step'] = 25
    args['maxV'] = 0.02
    args['num_chr'] = 24
    args['chr_prefix'] = ''
    args['chrs_valid'] = ['19']
    args['chrs_test'] = ['2', '6', '8']    # 8 is priority
    args['assembly'] = 'hg19'
    
    args['cooler_dir'] = f'{base_dir}/data/data_{data}/cool_10KB'
    args['hic_dir'] = f'{base_dir}/data/data_{data}/Hi-C'
    
    args['aug_chr'] = '8'
    
    args['predict_chr'] = '8'
    
    args['loop_info'] = f'{base_dir}/data/data_{data}/Table S1. Chromatin loops and differential information.xlsx'
    
elif data == 'Zhang':
    args['resolution'] = 10_000
    args['num_timepoints'] = 6
    args['cooler_balance'] = True
    args['ids'] = ["t1","t2","t3","t4","t5","t6"]
    args['step'] = 25
    args['maxV'] = 0.02
    args['num_chr'] = 24
    args['chr_prefix'] = 'chr'
    args['assembly'] = 'hg19'
    
    args['cooler_dir'] = f'{base_dir}/data/data_{data}/cool_10KB'
    
    args['predict_chr'] = '3'
    
    args['hic_dir'] = f'{base_dir}/data/data_{data}/Hi-C'
    
    args['gtf_file'] = f'{base_dir}/data/data_{data}/gencode.v19.annotation.gtf'
    
    args['plot_rna'] = True
    args['plot_triangular'] = True
    
elif data == 'Cohesinloss_Rao':
    args['resolution'] = 10_000
    args['num_timepoints'] = 6
    args['cooler_balance'] = True
    args['ids'] = ["t1","t2","t3","t4","t5","t6"]
    args['step'] = 25
    args['maxV'] = 0.02
    args['num_chr'] = 24
    args['chr_prefix'] = ''
    args['assembly'] = 'hg19'
    
    args['cooler_dir'] = f'{base_dir}/data/data_{data}/cool_10KB'
    
    args['predict_chr'] = '5'
    
    args['hic_dir'] = f'{base_dir}/data/data_{data}/Hi-C'


args['output_predict_file'] = os.path.join(args['output_predict_dir'], f"{args['chr_prefix']}{args['predict_chr']}.npy")
args['output_eval_file'] = os.path.join(args['output_predict_dir'], "eval")

    
if data_type == 'tad_simulated': args['eval_tads'] = True
# if data_type == 'loop_simulated': args['eval_loops'] = True

def get_args():
    return args


if __name__ == "__main__":
    import json
    print(json.dumps(args))
