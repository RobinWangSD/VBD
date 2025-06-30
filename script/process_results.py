from glob import glob 
import pickle
import numpy as np
from tqdm import tqdm

def table_2_stats(results_dir):
    scenario_results_path = glob(results_dir)
    print(scenario_results_path[0])

    speed_acc_ds = []
    steer_acc_ds = []
    combined_acc_ds = []
    minOR = []
    meanOR = []
    minADE = []
    meanADE = []
    minFDE = []
    meanFDE = []
    steer_key = []
    speed_key = []

    for scenario_result_path in tqdm(scenario_results_path):
        with open(scenario_result_path, 'rb') as scenario_result_f:
            scenario_result = pickle.load(scenario_result_f) 
        # print(scenario_result.keys())
        speed_acc = np.mean(scenario_result['speed_acc'])
        steer_acc = np.mean(scenario_result['steer_acc'])
        speed_acc_ds.append(speed_acc)
        steer_acc_ds.append(steer_acc)
        speed_key.append(scenario_result['speed_label'].detach().cpu().numpy())
        steer_key.append(scenario_result['steer_label'].detach().cpu().numpy())
        combined_acc = np.mean(scenario_result['speed_acc'] * scenario_result['steer_acc'])
        combined_acc_ds.append(combined_acc)
        # minADE
        # meanADE
        # minFDE
        # meanFDE.append()
        minOR.append(scenario_result['offroad'].min())
        meanOR.append(scenario_result['offroad'].mean())
    return speed_acc_ds, steer_acc_ds, speed_key, steer_key, minOR, meanOR, combined_acc_ds



results_dir_dict = {
    # '1': '/robin-west/VBD/config/_table_2/results/scale_0.0_cond_None_means_type_steer_and_speed_gradients_scale_1/*.pkl',
    # '01': '/robin-west/VBD/config/_table_2/results/scale_0.0_cond_None_means_type_steer_and_speed_gradients_scale_0.1/*.pkl',
    # '10': '/robin-west/VBD/config/_table_2/results/scale_0.0_cond_None_means_type_steer_and_speed_gradients_scale_10/*.pkl',
    # 'c': '/robin-west/VBD/config/_table_2/results/scale_0.0_cond_20_means_type_steer_and_speed_gradients_scale_0.0/*.pkl',
    'ms8': '/robin-west/VBD/config/_table_2/results/scale_8.0_cond_None_means_type_steer_and_speed_gradients_scale_0.0/*.pkl',
    'ms15': '/robin-west/VBD/config/_table_2/results/scale_15.0_cond_None_means_type_steer_and_speed_gradients_scale_0.0/*.pkl',
}

combined_results = {}
for key in results_dir_dict:
    results_dir = results_dir_dict[key]
    speed_acc_ds, steer_acc_ds, speed_key, steer_key, minOR, meanOR, combined_acc_ds = table_2_stats(results_dir)
    combined_results[key] = {
        'steer_key': steer_key,
        'speed_key': speed_key,
        'steer_acc': steer_acc_ds,
        'speed_acc': speed_acc_ds,
        'min_offroad': minOR,
        'mean_offroad': meanOR,
        'acc': combined_acc_ds,
    }
    print(key,np.mean(speed_acc_ds), np.mean(steer_acc_ds), np.mean(minOR))

with open('/robin-west/VBD/config/_table_2/results/table2_with_min_.pkl', 'wb') as table_2_f:
    pickle.dump(combined_results, table_2_f)