import sys
import os
import torch
import pickle
import glob
import numpy as np
from torch.utils.data import Dataset
from .data_utils import *
import functools
import pickle


class WaymaxDataset(Dataset):
    """
    Dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
    """

    def __init__(
        self,
        data_dir,
        future_len,
        predict_ego_only,
        action_labels_path,
        anchor_path = "data/cluster_64_center_dict.pkl",
    ):
        self.data_list = glob.glob(data_dir+'/*') if data_dir is not None else []
        self.anchors = pickle.load(open(anchor_path, "rb"))

        self.future_len = future_len
        self.predict_ego_only = predict_ego_only

        self.action_labels = pickle.load(open(action_labels_path, "rb"))
        
        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.data_list)
    
    def _process(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 1:
                anchors.append(self.anchors['TYPE_VEHICLE'])
            elif types[i] == 2:
                anchors.append(self.anchors['TYPE_PEDESTRIAN'])
            elif types[i] == 3:
                anchors.append(self.anchors['TYPE_CYCLIST'])
            else:
                anchors.append(np.zeros_like(self.anchors['TYPE_VEHICLE']))

        return np.array(anchors, dtype=np.float32)
    
    def gen_tensor(self, data, scenario_id):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """
        scenario = data['scenario_raw']
        sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
        sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
        agents_history = data['agents_history'] # A, T, D 
        if not self.predict_ego_only:
            agents_interested = data['agents_interested']
        else: 
            agents_interested = np.zeros_like(data['agents_interested'])
            agents_interested[sdc_id_in_processed] = 10
        agents_future = data['agents_future'][:, :self.future_len + 1]
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        anchors = self._process(agents_type)
        roadgraph_points = scenario.roadgraph_points

        # if self.future_len == 40:
        #     steering_action_labels = self.action_labels[scenario_id]["4s_action"][0,1] 
        #     speed_action_labels = self.action_labels[scenario_id]["4s_action"][0,0] 
        # elif self.future_len == 80:
        #     steering_action_labels = self.action_labels[scenario_id]["8s_action"][1] 
        #     speed_action_labels = self.action_labels[scenario_id]["8s_action"][0]
        action_label = self.action_labels[scenario_id]['action_label'] 

        tensors = {
            "scenario_id": scenario_id,
            "sdc_idx": sdc_id_in_processed,
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors),
            "sdc_steer_label": action_label,
            "sdc_speed_label": torch.inf, #speed_action_labels,
        }
        
        return tensors

    def __getitem__(self, idx):
        with open(self.data_list[idx], 'rb') as f:
            data = pickle.load(f)
        scenario_id = self.data_list[idx].split("/")[-1].rstrip(".pkl").split("_")[-1]
        return self.gen_tensor(data, scenario_id)


class WaymaxTestDataset(WaymaxDataset):
    """
    Test dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
        max_object (int, optional): Maximum number of objects. Defaults to 16.
        max_polylines (int, optional): Maximum number of polylines. Defaults to 256.
        history_length (int, optional): Length of history. Defaults to 11.
        num_points_polyline (int, optional): Number of points in each polyline. Defaults to 30.
    """

    def __init__(
        self,
        data_dir: str,
        future_len,
        predict_ego_only,
        action_labels_path,
        anchor_path = "data/cluster_64_center_dict.pkl",
        max_object: int = 16,
        # max_map_points: int = 3000,
        # max_polylines: int = 256,
        # history_length: int = 11,
        # num_points_polyline: int = 30,
    ) -> None:
        super().__init__(
            data_dir = data_dir, 
            future_len = future_len,
            predict_ego_only=predict_ego_only,
            action_labels_path=action_labels_path,
            anchor_path=anchor_path
            )

        self.max_object = max_object
        # self.max_polylines = max_polylines
        # self.max_map_points = max_map_points
        # self.history_length = history_length
        # self.num_points_polyline = num_points_polyline
        
        
        self.base_path = os.path.dirname(os.path.abspath(self.data_list[0])) if len(self.data_list) > 0 else None
                
    def process_scenario(self, scenario_raw, current_index: int = 10,
                        use_log: bool = True, selected_agents=None,
                        remove_history=False):
        """
        Process a scenario and generate tensors.

        Args:
            scenario_raw (dict): Raw scenario data.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.
            selected_agents (list, optional): List of selected agents. Defaults to None.

        Returns:
            dict: Dictionary of tensors.
        """
        data_dict = data_process_scenario(
            scenario_raw,
            current_index=current_index,
            max_num_objects=self.max_object,
            max_polylines=self.max_polylines,
            num_points_polyline=self.num_points_polyline,
            use_log=use_log,
            selected_agents=selected_agents,
            remove_history=remove_history,
        )
        
        data_dict['anchors'] = self._process(data_dict['agents_type'])

        return data_dict
        
    def reset_agent_length(self,max_object):
        """
        Reset the maximum number of objects.

        Args:
            max_object (int): Maximum number of objects.
        """
        self.max_object = max_object
    

    def gen_tensor(self, data, scenario_id):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """
        scenario = data['scenario_raw']
        sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
        sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
        if sdc_id_in_processed >= self.max_object:
            assert False
        agents_history = data['agents_history'] # A, T, D 
        if not self.predict_ego_only:
            agents_interested = data['agents_interested']
        else: 
            agents_interested = np.zeros_like(data['agents_interested'])
            agents_interested[sdc_id_in_processed] = 10
        agents_future = data['agents_future'][:, :self.future_len + 1]
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        anchors = self._process(agents_type)

        # steering_action_labels = self.action_labels[scenario_id]["4s_action"][0,1] 
        # speed_action_labels = self.action_labels[scenario_id]["4s_action"][0,0] 
        action_label = self.action_labels[scenario_id]['action_label'] 

        tensors = {
            "scenario_id": scenario_id,
            "sdc_idx": sdc_id_in_processed,
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors),
            "sdc_steer_label": action_label,
            "sdc_speed_label": torch.inf,
        }
        
        return tensors

        
    def get_scenario_by_id(
        self, scenario_id,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its ID.

        Args:
            scenario_id (int): Scenario ID.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        file_path = os.path.join(self.base_path, f"scenario_{scenario_id}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
        else:
            raise ValueError("scenario_raw not found")
        
        # data_dict = self.process_scenario(
        #     scenario_raw,
        #     current_index=current_index,
        #     use_log=use_log,
        #     remove_history=remove_history,
        # )
        # scenario_id = self.data_list[idx].split("/")[-1].rstrip(".pkl").split("_")[-1]
        data_dict = self.gen_tensor(data, scenario_id)
        
        return scenario_id, scenario_raw, data_dict#, data
    
    def get_scenario_by_index(
        self, index,
        current_index: int = 10,
        use_log: bool = True,
        remove_history=False
    ):
        """
        Get a scenario by its index.

        Args:
            index (int): Scenario index.
            current_index (int, optional): Current index. Defaults to 10.
            use_log (bool, optional): Whether to use logarithmic scaling. Defaults to True.

        Returns:
            tuple: Scenario ID, scenario raw data, and tensors.
        """
        filename = self.data_list[index]
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        if 'scenario_raw' in data:
            scenario_raw = data['scenario_raw']
            scenario_id = data['scenario_id']
        elif 'scenario' in data:
            scenario_raw = data['scenario']
            scenario_id = filename.split('/')[-1].split('.')[0].split('_')[-1]
        else:
            raise ValueError("scenario_raw not found")
        
        
        # data_dict = self.process_scenario(
        #     scenario_raw,
        #     current_index=current_index,
        #     use_log=use_log,
        #     remove_history=remove_history,
        # )
        scenario_id = self.data_list[index].split("/")[-1].rstrip(".pkl").split("_")[-1]
        data_dict = self.gen_tensor(data, scenario_id)
        
        return scenario_id, scenario_raw, data_dict
    
    def __getitem__(self, idx):
        _, _, data_dict = self.get_scenario_by_index(idx)
        return data_dict       
    
    


class WaymaxComboTestDataset(Dataset):
    """
    Dataset class for Waymax data.

    Args:
        data_dir (str): Directory path where the data is stored.
        anchor_path (str, optional): Path to the anchor file. Defaults to "data/cluster_64_center_dict.pkl".
    """

    def __init__(
        self,
        data_dir,
        future_len,
        predict_ego_only,
        action_combo_path,
        anchor_path = "data/cluster_64_center_dict.pkl",
    ):
        self.data_dir = data_dir
        self.anchors = pickle.load(open(anchor_path, "rb"))

        self.future_len = future_len
        self.predict_ego_only = predict_ego_only

        self.action_combo_list = pickle.load(open(action_combo_path, "rb"))
        
        self.__collate_fn__ = data_collate_fn

    def __len__(self):
        return len(self.action_combo_list)
    
    def _process(self, types):
        """
        Process the agent types and convert them into anchor vectors.

        Args:
            types (numpy.ndarray): Array of agent types.

        Returns:
            numpy.ndarray: Array of anchor vectors.
        """
        anchors = []

        for i in range(len(types)):
            if types[i] == 1:
                anchors.append(self.anchors['TYPE_VEHICLE'])
            elif types[i] == 2:
                anchors.append(self.anchors['TYPE_PEDESTRIAN'])
            elif types[i] == 3:
                anchors.append(self.anchors['TYPE_CYCLIST'])
            else:
                anchors.append(np.zeros_like(self.anchors['TYPE_VEHICLE']))

        return np.array(anchors, dtype=np.float32)
    
    def gen_tensor(self, data, scenario_id, action):
        """
        Generate tensors from the input data.

        Args:
            data (dict): Input data dictionary.

        Returns:
            dict: Dictionary of tensors.
        """
        scenario = data['scenario_raw']
        sdc_id = np.where(scenario.object_metadata.is_sdc)[0][0]
        sdc_id_in_processed = np.where(data["agents_id"]==sdc_id)[0][0]
        agents_history = data['agents_history'] # A, T, D 
        if not self.predict_ego_only:
            agents_interested = data['agents_interested']
        else: 
            agents_interested = np.zeros_like(data['agents_interested'])
            agents_interested[sdc_id_in_processed] = 10
        agents_future = data['agents_future'][:, :self.future_len + 1]
        agents_type = data['agents_type']
        traffic_light_points = data['traffic_light_points']
        polylines = data['polylines']
        polylines_valid = data['polylines_valid']
        relations = data['relations']
        anchors = self._process(agents_type)
        roadgraph_points = scenario.roadgraph_points

        # if self.future_len == 40:
        #     steering_action_labels = self.action_labels[scenario_id]["4s_action"][0,1] 
        #     speed_action_labels = self.action_labels[scenario_id]["4s_action"][0,0] 
        # elif self.future_len == 80:
        #     steering_action_labels = self.action_labels[scenario_id]["8s_action"][1] 
        #     speed_action_labels = self.action_labels[scenario_id]["8s_action"][0] 
        speed_action_label, steer_action_label = action

        tensors = {
            "scenario_id": scenario_id,
            "sdc_idx": sdc_id_in_processed,
            "agents_history": torch.from_numpy(agents_history),
            "agents_interested": torch.from_numpy(agents_interested),
            "agents_future": torch.from_numpy(agents_future),
            "agents_type": torch.from_numpy(agents_type),
            "traffic_light_points": torch.from_numpy(traffic_light_points),
            "polylines": torch.from_numpy(polylines),
            "polylines_valid": torch.from_numpy(polylines_valid),
            "relations": torch.from_numpy(relations),
            "anchors": torch.from_numpy(anchors),
            "sdc_steer_label": steer_action_label,
            "sdc_speed_label": speed_action_label,
        }
        
        return tensors

    def __getitem__(self, idx):
        scenario_id, action = self.action_combo_list[idx]
        file_path = os.path.join(self.data_dir, f"scenario_{scenario_id}.pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return self.gen_tensor(data, scenario_id, action)