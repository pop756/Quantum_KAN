from qiskit.compiler import transpile
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.pulse import builder, DriveChannel, Schedule, GaussianSquareDrag, Drag, Play, ScheduleBlock, Delay
from qiskit.transpiler import InstructionProperties
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2, SamplerV2, Session, Batch
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.pulse.instructions import ShiftPhase,ShiftFrequency
import numpy as np
import torch
import pickle
import pandas as pd
import uuid
import json
import copy
from functions.custom_pulse import custom_GaussianSquare


def remove_ecr_gates(circuit):
    new_circuit = QuantumCircuit(circuit.num_qubits)
    for instr, qargs, cargs in circuit.data:
        if instr.name != 'ecr':
            new_circuit.append(instr, qargs, cargs)
    return new_circuit

def check_connect(backend,init_list):
    min = np.min(np.array(init_list))
    connection_temp = []
    connection = []
    for item in backend.target['ecr']:
        if (item[0] in init_list) and (item[1] in init_list):
            connection.append(item)

    connection_temp = []
    for item in backend.target['ecr']:
        if (item[0] in init_list) and (item[1] in init_list):
            connection_temp.append((int(item[0]-min),int(item[1]-min)))

    return connection,connection_temp

    






import numpy as np
import math
class update_pulse():
    def __init__(self,
                backend,
                config):
        self.backend = backend
        self.config_list = config
    
    def update_ecr_real(self):
        """
        _Make ecr gate to stretch gate in backend_
        """
        
        backend_copy = copy.deepcopy(self.backend)
        for config in self.config_list:
            pulse_real = self.__ecr_to_schedule(config)
            backend_copy.target.update_instruction_properties(f'ecr',tuple(config["init"]),properties = InstructionProperties(calibration=(pulse_real)))
    

        return backend_copy
    
    
    
    def __ecr_to_schedule(self,config):
        initial_layout = tuple(config["init"])
        backend = copy.deepcopy(self.backend)
        x_target = backend.target['x'][(initial_layout[1],)].calibration.instructions[0][1]
        x_control = backend.target['x'][(initial_layout[0],)].calibration.instructions[0][1]
        CR_plus =  backend.target['ecr'][initial_layout].calibration.instructions[1][1]

        
        duration_width_diff = int(CR_plus.pulse.duration-CR_plus.pulse._params['width'])
        duration =  CR_plus.pulse.duration
        width = duration-duration_width_diff
        CR_plus.pulse.duration = duration
        signal_params_c = {'amp':config['cr_amp'],'width':width,'angle': config['cr_angle']}
        

        CR_plus.pulse._params.update(signal_params_c)

        my_schedule = ScheduleBlock()
        
        real_pulse = ScheduleBlock()
        #real_pulse += ShiftFrequency(config['offset'],x_target.channel)
        #real_pulse += ShiftFrequency(config['offset'],CR_plus.channel)
        real_pulse += Play(custom_GaussianSquare(duration,amp = config['cr_amp'],sigma = 32,offset = config['offset']*backend.dt,width = width, angle = config['cr_angle']),CR_plus.channel)
        real_pulse += Play(custom_GaussianSquare(duration,amp = config['amp1'],sigma = 32,offset = 0,width = width, angle = config['x_angle1']),x_target.channel)
        #real_pulse += ShiftFrequency(-config['offset'],x_target.channel)
        #real_pulse += ShiftFrequency(-config['offset'],CR_plus.channel)
        real_pulse += Delay(x_control.pulse.duration,x_target.channel)
        real_pulse += Delay(x_control.pulse.duration,CR_plus.channel)
        
        real_pulse += Delay(duration,x_control.channel)
        real_pulse += x_control
        
        
        
        signal_params_c['angle'] += np.pi
        CR_minus = copy.deepcopy(CR_plus)
        CR_minus.pulse._params.update(signal_params_c)

        #real_pulse += ShiftFrequency(config['offset'],x_target.channel)
        #real_pulse += ShiftFrequency(config['offset'],CR_plus.channel)
        real_pulse += Play(custom_GaussianSquare(duration,amp = config['cr_amp'],sigma = 32,offset = config['offset']*backend.dt,width = width, angle = config['cr_angle']+np.pi),CR_plus.channel)
        real_pulse += Play(custom_GaussianSquare(duration,amp = config['amp2'],sigma = 32,offset = 0,width = width, angle = config['x_angle2']+np.pi),x_target.channel)
        #real_pulse += ShiftFrequency(-config['offset'],x_target.channel)
        #real_pulse += ShiftFrequency(-config['offset'],CR_plus.channel)


        
        my_schedule += real_pulse
        return my_schedule
    

import re
from typing import List, Optional

class train_ZNE(update_pulse):
    def __init__(self,size=20,**kwargs):
        """_Error gate와 실제 ECR 사이의 노이즈가 얼마나 차이가 있는지 체크하는 class_

        Args:
            backend (_type_): _IBM real backend_
            stretch_list (_type_): _Error gate stretch_
            ecr_stretch (int, optional): _ecr gate stretch_. Defaults to 1.
            l (int, optional): _noise injection_. Defaults to 0.
            size (int, optional): _half number of maximum ecr(error) gate_. Defaults to 20.
            connect (list, optional): _qubit use_. Defaults to [0,1].
        """

        self.size = size
        self.class_id = str(uuid.uuid4())
        super().__init__(**kwargs)
        
        
    def make_circ(self,odd=False,axis = 'z'):
        """_해당하는 서킷을 만드는 메소드_

        Args:
            odd (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        qc_list = []
        if not(odd):
            for i in range(self.size):
                qc = QuantumCircuit(2,2)
                if axis == 'x':
                    qc.h(0)
                elif axis == 'y':
                    qc.h(0)
                    qc.s(0)
                for j in range(i):
                    qc.ecr(0,1)
                    qc.ecr(0,1)
                if axis == 'x':
                    qc.h(0)
                elif axis == 'y':
                    qc.sdg(0)
                    qc.h(0)
                qc.measure_all()
                qc_list.append(qc)
        else:
            for i in range(self.size):
                qc = QuantumCircuit(2,2)
                if axis == 'x':
                    qc.h(0)
                elif axis == 'y':
                    qc.h(0)
                    qc.s(0)
                qc.ecr(0,1)
                for j in range(i):
                    qc.ecr(0,1)
                    qc.ecr(0,1)
                if axis == 'x':
                    qc.h(0)
                elif axis == 'y':
                    qc.sdg(0)
                    qc.h(0)
                qc.measure_all()
                qc_list.append(qc)
        return qc_list

    def _run_qc_test(self,initial_layout,qc_list):
        backend_ecr = self.update_ecr_real()
        passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_ecr, initial_layout=list(initial_layout))
        qc_input_ecr = passmanager.run(qc_list)
        with Batch(backend=backend_ecr):
            sampler = SamplerV2()
            job_test = sampler.run(qc_input_ecr)
            #job = sampler.run(qc_list,shots=8000)
            job_test.update_tags([self.class_id,'ecr',f'stretch = {self.ecr_stretch}',f'l = {self.l}',f'connect = {list(initial_layout)}'])
            self.job_test.append(job_test)

    
    def _run_qc_error(self,initial_layout,qc_list,qc_list_odd):
        backend_error = self.update_ecr()
        passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_error, initial_layout=list(initial_layout))
        qc_input_ecr = passmanager.run(qc_list)
        qc_input_ecr_odd = passmanager.run(qc_list_odd)
        with Batch(backend=backend_error):
            sampler = SamplerV2()
            job_error = sampler.run(qc_input_ecr)
            job_error.update_tags([self.class_id,'even','error',f'stretch = {self.error_stretch}',f'connect = {list(initial_layout)}'])
            self.job_stretch.append(job_error)
            job_error = sampler.run(qc_input_ecr_odd)
            job_error.update_tags([self.class_id,'odd','error',f'stretch = {self.error_stretch}',f'connect = {list(initial_layout)}'])
            self.job_stretch_odd.append(job_error)

    def run(self):
        """_gate를 만들고 error 와 일반 ecr 결과를 class 내부에 만들어 주는 method_

        Args:
            axis (str, optional): _측정할 기저_. Defaults to 'z'.
        """
        self.job_stretch = []
        self.job_stretch_odd = []
        self.job_test = []
        qc_list = self.make_circ()
        qc_list_odd = self.make_circ(odd=True)
        for initial_layout in self.init_list:
            self._run_qc_test(initial_layout,qc_list)
            self._run_qc_error(initial_layout,qc_list,qc_list_odd)
    def load_data(self,class_id):
        service = self.backend.service
        self.job_stretch = service.jobs(job_tags = [class_id,'even'])
        self.job_stretch_odd = service.jobs(job_tags =[class_id,'odd'])
        self.job_test = service.jobs(job_tags =[class_id,'ecr'])
    
    def _get_result(self,index):
        result_even = self.job_stretch[index].result()
        result_odd = self.job_stretch_odd[index].result()
        result_test = self.job_test[index].result()
        return result_even,result_odd,result_test
    
    def get_data(self):
        data_odd_dict = {}
        data_even_dict = {}
        data_test_dict = {}
        
        for index in range(len(self.job_stretch)):
            result_even,result_odd,result_test = self._get_result(index)
            
            odd_key = self._extract_and_format_values((self.job_stretch_odd[index].tags))
            even_key = self._extract_and_format_values((self.job_stretch[index].tags))
            test_key = self._extract_and_format_values((self.job_test[index].tags))
            
            
            
            data_odd_dict[odd_key]= []
            data_even_dict[even_key] = []
            data_test_dict[test_key] = []
            for j in range(len(result_even)):
                data_1 = self._convert_to_expectation(result_even[j].data['meas'].get_counts())
                data_2 = self._convert_to_expectation(result_odd[j].data['meas'].get_counts())
                data_3 = self._convert_to_expectation(result_test[j].data['meas'].get_counts())
                data_odd_dict[odd_key].append(data_1)
                data_even_dict[even_key].append(data_2)
                data_test_dict[test_key].append(data_3)
                
        data_odd = pd.DataFrame(data_odd_dict)
        data_even = pd.DataFrame(data_even_dict)
        data_test = pd.DataFrame(data_test_dict)
        result = pd.concat([data_odd,data_even,data_test],axis=1)
        return result
        
    def _convert_to_expectation(self,sampling_results):
        """
        Convert quantum sampling results to expectation values based on bit interpretation.
        
        Parameters:
        sampling_results (dict): A dictionary with quantum states as keys and counts as values.
        
        Returns:
        float: The expectation value.
        """
        total_samples = sum(sampling_results.values())
        expectation_value = 0.0
        
        for state, count in sampling_results.items():
            bit_sum = sum(int(bit)*(-2)+1 for bit in state)
            probability = count / total_samples
            expectation_value += probability * bit_sum
        
        return expectation_value


    def _extract_and_format_values(self,data: List[str]) -> str:
        """
        Extracts values based on specific patterns from a list of strings and formats the output.

        Args:
            data (List[str]): The list of strings to search.

        Returns:
            str: A formatted string containing the extracted values.
        """
        results: Dict[str, Any] = {
            'connect': None,
            'odd': None,
            'even': None,
            'ecr': None
        }

        # 정규 표현식 패턴 정의
        patterns = {
            'connect': re.compile(r'connect\s*=\s*\[(\d+),\s*(\d+)\]'),
            'odd': re.compile(r'odd'),
            'even': re.compile(r'even'),
            'ecr': re.compile(r'ecr')
        }

        # 리스트를 순회하면서 패턴 매칭
        for item in data:
            for key, pattern in patterns.items():
                match = pattern.match(item)
                if match:
                    if key == 'connect':
                        results[key] = [int(match.group(1)), int(match.group(2))]
                    else:
                        results[key] = key  # 패턴의 이름을 저장
                    break

        # None인 항목 제거
        filtered_results = {k: v for k, v in results.items() if v is not None}

        # 결과 포맷팅
        formatted_results = []
        for key, value in filtered_results.items():
            if isinstance(value, list):
                formatted_results.append(f"[{', '.join(map(str, value))}]")
            else:
                formatted_results.append(value)

        return '_'.join(formatted_results)
class ZNE(update_pulse):
    def __init__(self,circ,H,train_parameters=None,valid_parameters=None,validation_size=100,train_size=100,ZNE_factor=[1,2,3,4],class_name = None,**kwargs):
        """_Error gate 로 구성된 train set와 일반 ecr로 구성된 validation set을 만들어주는 class_

        Args:
            circ (_type_): _input_circuit_
            H (_type_): _expectation_measure_basis_
            backend (_type_): _backend_
            init_list (_type_): _qubit_use_
            validation_size (int, optional): _size of validation_. Defaults to 100.
            train_size (int, optional): _size of train_. Defaults to 100.
            stretch (int, optional): _length of pulse at 1_. Defaults to 1.
            ZNE_factor (list, optional): _ZNE factor_. Defaults to [1,1.8,2.2,2.6].
        """
        
        super().__init__(**kwargs)
        
        min_value = self.__min_connect_value(self.x_amp_dict.keys())
        
        self.H = H
        self.validation_size = validation_size
        self.train_size = train_size
        self.ZNE_factor = ZNE_factor
        if class_name is None:
            self.class_id = str(uuid.uuid4())
        else:
            self.class_id = class_name
        
        circ = transpile(circ,basis_gates=['rz','sx','x','ecr'],coupling_map=[[i[0]-min_value,i[1]-min_value] for i in self.x_amp_dict.keys()],optimization_level=2,seed_transpiler=30)
        train_circ = remove_ecr_gates(circ)
        self.circ = circ
        self.train_circ = train_circ
        
        self.qubit_use = self._initial_layout()
        if train_parameters is None:
            self.train_parameters = np.random.uniform(-3.14, 3.14, [train_size,len(circ.parameters)])
            self.valid_parameters = np.random.uniform(-3.14, 3.14, [validation_size,len(circ.parameters)])
        
        else:
            self.train_parameters = train_parameters
            self.valid_parameters = valid_parameters
    
    def __min_connect_value(self,connect_seq):
        min_value = 1e5
        for connect in connect_seq:
            min_value = min(connect[0],connect[1],min_value)
        return min_value
    
    def _initial_layout(self):
        result_list = []
        for connect in self.x_amp_dict.keys():
            connect = list(connect)
            result_list += connect
        result_list = set(result_list)
        return list(result_list)
    
    def ZNE_pulse_error(self,factor):
        """_backend에 있는 ecr gate를 error gate와 stretch 된 ecr gate로 치환해주는 함수_

        Args:
            factor (_float_): _에러주입 정도(error gate가 뒤에 들감)_

        Returns:
            _type_: _backend(cal)_
        """
        self.l = factor-1
        backend_error = self.update_ecr()
        #backend_ecr = self.update_ecr_real()
        return backend_error
    
    def ZNE_pulse_ecr(self,factor):
        self.l = factor-1
        backend_error = self.update_ecr_real()
        #backend_ecr = self.update_ecr_real()
        return backend_error
    def _make_train_set(self,factor):
        backend_error = self.ZNE_pulse_error(factor)
        #Train set 만들기
        passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_error, initial_layout=self.qubit_use)
        qc_input = passmanager.run(self.circ)
        isa_observables = self.H.apply_layout(qc_input.layout)
        with Batch(backend = backend_error):
            estimator = EstimatorV2()
            job = estimator.run([(qc_input, isa_observables, self.train_parameters[i]) for i in range(self.train_size)])
            job.update_tags([self.class_id,'train_set',f"l={factor}",f"stretch={self.error_stretch}"])
            self.train_jobs.append(job)
    def _make_valid_set(self,factor):
        backend_ecr = self.ZNE_pulse_ecr(factor)
        #Validation set 만들기
        passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_ecr, initial_layout=self.qubit_use)
        qc_input = passmanager.run(self.circ)
        isa_observables = self.H.apply_layout(qc_input.layout)
        with Batch(backend = backend_ecr):
            estimator = EstimatorV2()
            job = estimator.run([(qc_input, isa_observables, self.valid_parameters[i]) for i in range(self.validation_size)])
            #job = sampler.run(qc_list,shots=8000)
            job.update_tags([self.class_id,'valid_set',f"l={factor}","x"])
            self.valid_jobs.append(job)
    
    def _make_ZNE_set(self):
        with Batch(backend = self.backend):
            passmanager = generate_preset_pass_manager(optimization_level=0, backend=self.backend, initial_layout=self.qubit_use)
            qc_input = passmanager.run(self.circ)
            isa_observables = self.H.apply_layout(qc_input.layout)
            estimator = EstimatorV2(options={"resilience_level": 2})
            job = estimator.run([(qc_input, isa_observables, self.valid_parameters[i]) for i in range(self.validation_size)])
            #job = sampler.run(qc_list,shots=8000)
            job.update_tags([self.class_id,'ZNE_set'])
            self.ZNE_jobs.append(job)    


    def make_data(self):
        """
        _데이터를 만들어주는 method, IBM에 train set 그리고 validation set 을 만들기 위한 job을 던짐_
        """
        self.ZNE_jobs = []
        self.train_jobs = []
        self.valid_jobs = []
        for factor in self.ZNE_factor:
            #Train set 만들기
            self._make_train_set(factor)

            #Validation set 만들기
            self._make_valid_set(factor)
                    
        self._make_ZNE_set()

    def make_label(self):
        """_label을 만들어주는 method_
        Returns:
            _train_label,valid_label_: _label data 결과_
        """
        estimator = StatevectorEstimator()
        job_train = estimator.run([(self.train_circ, self.H, self.train_parameters[i]) for i in range(self.train_size)])
        job_valid = estimator.run([(self.circ, self.H, self.valid_parameters[i]) for i in range(self.validation_size)])
        train_label = torch.tensor(np.array([result.data.evs for result in job_train.result()]),dtype=torch.float32)
        validation_label = torch.tensor(np.array([result.data.evs for result in job_valid.result()]),dtype=torch.float32)
        train_label = torch.reshape(train_label,[-1,1])
        validation_label = torch.reshape(validation_label,[-1,1])
        return train_label,validation_label

    def load_data(self,class_id):
        service = self.backend.service
        self.train_jobs = service.jobs(job_tags =[class_id,'train_set'])
        self.valid_jobs = service.jobs(job_tags =[class_id,'valid_set'])
        self.ZNE_jobs = service.jobs(job_tags =[class_id,'ZNE_set'])


    def _get_train_jobs(self):
        train_set = []
        for job in self.train_jobs:
            train_list = []
            res_datas = job.result()
            for res in res_datas:
                train_list.append(res.data.evs)
            train_set.append(torch.tensor(np.array(train_list),dtype=torch.float32))
        return train_set
    def _get_valid_jobs(self):
        valid_set = []
        for job in self.valid_jobs:
            train_list = []
            res_datas = job.result()
            for res in res_datas:
                train_list.append(res.data.evs)
            valid_set.append(torch.tensor(np.array(train_list),dtype=torch.float32))
        return valid_set

    def _get_ZNE_jobs(self):
        ZNE_set = []
        for job in self.ZNE_jobs:
            train_list = []
            res_datas = job.result()
            for res in res_datas:
                train_list.append(res.data.evs)
            ZNE_set.append(torch.tensor(np.array(train_list),dtype=torch.float32))
        return ZNE_set
    
    def run(self):
        """__

        Args:
            make_data (bool, optional): _데이터를 이미 만들어 놓은지 유무_. Defaults to True.
            make_label (bool, optional): _label를 만들지 유무_. Defaults to True.
        """
        import torch
        train_set = self._get_train_jobs()
        valid_set = self._get_valid_jobs()
        ZNE_set = self._get_ZNE_jobs()
        
        
        train_data = torch.stack(train_set,dim=1)
        train_data = train_data.to(torch.float32)
        validation_data = torch.stack(valid_set,dim=1)
        validation_data = validation_data.to(torch.float32)
        ZNE_data = torch.stack(ZNE_set,dim=1)
        ZNE_data = ZNE_data.to(torch.float32)

        train_label,validation_label = self.make_label()
        return train_data,validation_data,train_label,validation_label,ZNE_data
    

class ZNE_stretch(ZNE):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    
    def ZNE_pulse_error(self,factor):
        """_backend에 있는 ecr gate를 error gate와 stretch 된 ecr gate로 치환해주는 함수_

        Args:
            factor (_float_): _에러주입 정도(error gate가 뒤에 들감)_

        Returns:
            _type_: _backend(cal)_
        """
        self.error_stretch = factor
        backend_error = self.update_ecr()
        #backend_ecr = self.update_ecr_real()
        return backend_error
    
    def ZNE_pulse_ecr(self,factor):
        self.ecr_stretch = factor
        backend_error = self.update_ecr_real()
        #backend_ecr = self.update_ecr_real()
        return backend_error