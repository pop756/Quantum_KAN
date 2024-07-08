from qiskit.compiler import transpile
import copy
from qiskit_ibm_runtime.fake_provider import FakePerth
from qiskit.circuit import QuantumCircuit, Gate
from qiskit.pulse import builder, DriveChannel,Schedule,GaussianSquare,Drag,Play,ScheduleBlock,Delay
from qiskit.transpiler import InstructionProperties
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 ,SamplerV2
from qiskit_ibm_runtime import Session
import numpy as np
from qiskit.primitives import StatevectorEstimator
import torch
from qiskit.quantum_info import SparsePauliOp
import pickle
import pandas as pd
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import Batch
import uuid


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


def ecr_to_error(pulse_schedule,l,amp_rate):
    pulse_copy = copy.deepcopy(pulse_schedule)
    duration_width_diff = int(pulse_copy.instructions[0][1].pulse.duration-pulse_copy.instructions[0][1].pulse._params['width'])
    x_duration = pulse_copy.instructions[2][1].pulse.duration
    duration = round((pulse_copy.instructions[0][1].pulse.duration)/16*l)*8
    rate = (pulse_copy.instructions[0][1].pulse.duration/2)/duration
    width = duration-duration_width_diff
    amp_x = pulse_copy.instructions[0][1].pulse._params['amp']*rate*amp_rate
    amp_c = pulse_input_c_1 = pulse_copy.instructions[1][1].pulse._params['amp']*rate*amp_rate

    my_schedule = ScheduleBlock()
    signal_params_x = {'width':width,'amp':amp_x}
    signal_params_c = {'width':width,'amp':amp_c}
    for j in range(2):
        pulse_copy = copy.deepcopy(pulse_schedule)
        pulse_input_x_1 = pulse_copy.instructions[0][1]
        pulse_input_x_2 = pulse_copy.instructions[3][1]
        pulse_input_c_1 = pulse_copy.instructions[1][1]
        #pulse_input_c_1.pulse._params['angle'] = 0
        pulse_input_c_1.pulse._params.update(signal_params_c)
        pulse_input_c_1.pulse.duration = duration
        pulse_input_x_1.pulse._params.update(signal_params_x)
        pulse_input_x_1.pulse.duration = duration
        pulse_input_drag = pulse_copy.instructions[2][1]
        pulse_input_c_2 = pulse_copy.instructions[4][1]
        pulse_input_c_2.pulse.duration = duration
        pulse_input_c_2.pulse._params.update(signal_params_c)
        pulse_input_x_2.pulse._params.update(signal_params_x)
        if j == 1:
            #pulse_input_drag.pulse._params['angle'] += 3.14
            c1 = float(pulse_input_c_1.pulse._params['angle'])
            x1 = float(pulse_input_x_1.pulse._params['angle'])
            c2 = float(pulse_input_c_2.pulse._params['angle'])
            x2 = float(pulse_input_x_2.pulse._params['angle'])
            pulse_input_c_1.pulse._params['angle'] = c2
            pulse_input_x_1.pulse._params['angle'] = x2
            pulse_input_c_2.pulse._params['angle'] = c1
            pulse_input_x_2.pulse._params['angle'] = x1
            #pulse_input_c_1.pulse._params['angle'] = c1+3.14
            #pulse_input_x_1.pulse._params['angle'] = x1+3.14
            #pulse_input_c_2.pulse._params['angle'] = c2+3.14
            #pulse_input_x_2.pulse._params['angle'] = x2+3.14
        pulse_input_x_2.pulse.duration = duration
        real_pulse = ScheduleBlock()
        real_pulse += pulse_input_c_1
        real_pulse += pulse_input_x_1
        real_pulse += Delay(x_duration,pulse_input_c_1.channel)
        real_pulse += Delay(x_duration,pulse_input_x_1.channel)
        real_pulse +=  pulse_input_c_2
        real_pulse += pulse_input_x_2
        real_pulse += Delay(duration,pulse_input_drag.channel)
        real_pulse += pulse_input_drag
        real_pulse += Delay(duration,pulse_input_drag.channel)
        real_pulse += pulse_input_drag
        my_schedule += real_pulse
    return my_schedule


def ecr_to_schedule(pulse_schedule,stretch):
    pulse_copy = copy.deepcopy(pulse_schedule)
    my_schedule = ScheduleBlock()

    #변수들 값 정의
    duration_width_diff = int(pulse_copy.instructions[0][1].pulse.duration-pulse_copy.instructions[0][1].pulse._params['width'])
    duration = round(pulse_copy.instructions[0][1].pulse.duration/8*stretch)*8
    rate = pulse_copy.instructions[0][1].pulse.duration/duration
    width = duration - duration_width_diff
    x_duration = pulse_copy.instructions[2][1].pulse.duration


    pulse_input_x_1 = pulse_copy.instructions[0][1]
    pulse_input_x_2 = pulse_copy.instructions[3][1]
    pulse_input_c_1 = pulse_copy.instructions[1][1]
    pulse_input_drag = pulse_copy.instructions[2][1]
    pulse_input_c_2 = pulse_copy.instructions[4][1]
    #c1 = float(pulse_input_c_1.pulse._params['angle'])
    #x1 = float(pulse_input_x_1.pulse._params['angle'])
    #c2 = float(pulse_input_c_2.pulse._params['angle'])
    #x2 = float(pulse_input_x_2.pulse._params['angle'])
    #pulse_input_c_1.pulse._params['angle'] = c2
    #pulse_input_x_1.pulse._params['angle'] = x2
    #pulse_input_c_2.pulse._params['angle'] = c1
    #pulse_input_x_2.pulse._params['angle'] = x1
    amp_x = pulse_copy.instructions[0][1].pulse._params['amp']*rate
    amp_c = pulse_copy.instructions[1][1].pulse._params['amp']*rate

    signal_params_x = {'width':width,'amp':amp_x}
    signal_params_c = {'width':width,'amp':amp_c}
    pulse_input_c_1.pulse._params.update(signal_params_c)
    pulse_input_c_1.pulse.duration = duration
    pulse_input_x_1.pulse._params.update(signal_params_x)
    pulse_input_x_1.pulse.duration = duration
    pulse_input_c_2.pulse.duration = duration
    pulse_input_c_2.pulse._params.update(signal_params_c)
    pulse_input_x_2.pulse._params.update(signal_params_x)
    pulse_input_x_2.pulse.duration = duration
    real_pulse = ScheduleBlock()
    real_pulse += pulse_input_c_1
    real_pulse += pulse_input_x_1
    real_pulse += Delay(x_duration,pulse_input_c_1.channel)
    real_pulse += Delay(x_duration,pulse_input_x_1.channel)
    real_pulse +=  pulse_input_c_2
    real_pulse += pulse_input_x_2
    real_pulse += Delay(duration,pulse_input_drag.channel)
    real_pulse += pulse_input_drag
    my_schedule += real_pulse
    return my_schedule


def update_ecr(l,init_list,backend,stretch=1,amp_rate=1):
    """
    _Make ecr gate to error gate in backend_
    """
    backend_copy = copy.deepcopy(backend)
    for initial_layout in init_list:
        pulse_schedule = backend_copy.target['ecr'][initial_layout].calibration
        pulse_real = ecr_to_error(pulse_schedule,stretch,amp_rate=amp_rate)


        if l == 0:
            backend_copy.target.update_instruction_properties(f'ecr',initial_layout,properties = InstructionProperties(calibration=(pulse_real)))
        else:
            my_schedule = ecr_to_error(pulse_schedule,l,amp_rate=amp_rate)
            backend_copy.target.update_instruction_properties(f'ecr',initial_layout,properties = InstructionProperties(calibration=(pulse_real+my_schedule)))



    return backend_copy

def update_ecr_real(l,init_list,backend,stretch=1,amp_rate=1):
    """
    _Make ecr gate to stretch gate in backend_
    """
    backend_copy = copy.deepcopy(backend)
    layouts = {}
    for initial_layout in init_list:
        pulse_schedule = backend_copy.target['ecr'][initial_layout].calibration
        pulse_real = ecr_to_schedule(pulse_schedule,stretch)

        if l == 0:
            backend_copy.target.update_instruction_properties(f'ecr',initial_layout,properties = InstructionProperties(calibration=(pulse_real)))
        else:
            my_schedule = ecr_to_error(pulse_schedule,l,amp_rate)
            backend_copy.target.update_instruction_properties(f'ecr',initial_layout,properties = InstructionProperties(calibration=(pulse_real+my_schedule)))



    return backend_copy

class ZNE():
    def __init__(self,circ,H,backend,init_list,amp_rate,validation_size=100,train_size=100,stretch=1.6,ecr_stretch = 1.1,ZNE_factor=[1,2,3,4]):
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
        self.H = H
        self.backend = backend
        self.validation_size = validation_size
        self.train_size = train_size
        self.stretch = stretch
        self.ecr_stretch = ecr_stretch
        self.ZNE_factor = ZNE_factor
        self.amp_rate = amp_rate
        self.class_id = str(uuid.uuid4())
        connection,connection_temp = check_connect(backend,init_list)
        circ = transpile(circ,basis_gates=['rz','sx','x','ecr'],coupling_map=[list(i) for i in connection_temp],optimization_level=2,seed_transpiler=30)
        train_circ = remove_ecr_gates(circ)
        self.circ = circ
        self.train_circ = train_circ
        self.init_list = init_list
        self.connection = connection
        np.random.seed(30)
        self.train_parameters = np.random.uniform(-3.14, 3.14, [train_size,len(circ.parameters)])
        np.random.seed(60)
        self.valid_parameters = np.random.uniform(-3.14, 3.14, [validation_size,len(circ.parameters)])
    def ZNE_pulse(self,factor):
        """_backend에 있는 ecr gate를 error gate와 stretch 된 ecr gate로 치환해주는 함수_

        Args:
            factor (_float_): _에러주입 정도(error gate가 뒤에 들감)_

        Returns:
            _type_: _backend(cal)_
        """
        backend_error = update_ecr((factor-1)*self.stretch,self.connection,self.backend,stretch=self.stretch,amp_rate=self.amp_rate)
        backend_ecr = update_ecr_real((factor-1)*self.stretch,self.connection,self.backend,stretch=self.ecr_stretch,amp_rate=self.amp_rate)
        return backend_error,backend_ecr

    def make_data(self):
        """
        _데이터를 만들어주는 method, IBM에 train set 그리고 validation set 을 만들기 위한 job을 던짐_
        """
        self.train_jobs = []
        self.valid_jobs = []
        for factor in self.ZNE_factor:
            backend_error,backend_ecr = self.ZNE_pulse(factor)
            #Train set 만들기
            passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_error, initial_layout=self.init_list)
            qc_input = passmanager.run(self.circ)
            isa_observables = self.H.apply_layout(qc_input.layout)
            with Batch(backend = backend_error):
                estimator = EstimatorV2()
                job = estimator.run([(qc_input, isa_observables, self.train_parameters[i]) for i in range(self.train_size)])
                job.update_tags([self.class_id,'train_set',f"l={factor}",f"stretch={self.stretch}"])
                self.train_jobs.append(job)

            #Validation set 만들기
            passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_ecr, initial_layout=self.init_list)
            qc_input = passmanager.run(self.circ)
            isa_observables = self.H.apply_layout(qc_input.layout)
            with Batch(backend = backend_ecr):
                estimator = EstimatorV2()
                job = estimator.run([(qc_input, isa_observables, self.valid_parameters[i]) for i in range(self.validation_size)])
                #job = sampler.run(qc_list,shots=8000)
                job.update_tags([self.class_id,'valid_set',f"l={factor}","x"])
                self.valid_jobs.append(job)

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

    def get_data(self,class_id):
        service = self.backend.service
        self.train_jobs = service.jobs(job_tags =[class_id,'train_set'])
        self.valid_jobs = service.jobs(job_tags =[class_id,'valid_set'])


    def run(self,get_data = False,make_label = True,train_uuid = None,valid_uuid = None):
        """__

        Args:
            make_data (bool, optional): _데이터를 이미 만들어 놓은지 유무_. Defaults to True.
            make_label (bool, optional): _label를 만들지 유무_. Defaults to True.
        """
        if get_data:
            self.get_data(train_uuid,valid_uuid)
        import torch
        train_set = []
        valid_set = []


        
        for job in self.train_jobs:
            train_list = []
            valid_list = []
            if job.status().value != 'job has successfully run':
                print('Job is not done')
                raise
            else:
                res_datas = job.result()
                for res in res_datas:
                    train_list.append(res.data.evs)
            train_set.append(torch.tensor(np.array(train_list),dtype=torch.float32))


        for job in self.valid_jobs:
            train_list = []
            valid_list = []
            if job.status().value != 'job has successfully run':
                print('Job is not done')
            else:
                res_datas = job.result()
                for res in res_datas:
                    valid_list.append(res.data.evs)
            valid_set.append(torch.tensor(np.array(valid_list),dtype=torch.float32))
        train_data = torch.stack(train_set,dim=1)
        train_data = train_data.to(torch.float32)
        validation_data = torch.stack(valid_set,dim=1)
        validation_data = validation_data.to(torch.float32)
        if make_label:
            train_label,validation_label = self.make_label()
            return train_data,validation_data,train_label,validation_label
        return train_data,validation_data


class train_ZNE():
    def __init__(self,backend,stretch_list,ecr_stretch=1,l=0,size=20,connect = [0,1],amp_rate=1):
        """_Error gate와 실제 ECR 사이의 노이즈가 얼마나 차이가 있는지 체크하는 class_

        Args:
            backend (_type_): _IBM real backend_
            stretch_list (_type_): _Error gate stretch_
            ecr_stretch (int, optional): _ecr gate stretch_. Defaults to 1.
            l (int, optional): _noise injection_. Defaults to 0.
            size (int, optional): _half number of maximum ecr(error) gate_. Defaults to 20.
            connect (list, optional): _qubit use_. Defaults to [0,1].
        """
        self.layout,self.connect = check_connect(backend,connect)
        self.qc_connect = connect
        self.stretch_list = stretch_list
        self.size = size
        self.backend = backend
        self.l = l
        self.ecr_stretch = ecr_stretch
        self.class_id = str(uuid.uuid4())
        self.amp_rate = amp_rate
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
                    qc.ecr(*self.connect[0])
                    qc.ecr(*self.connect[0])
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
                qc.ecr(*self.connect[0])
                for j in range(i):
                    qc.ecr(*self.connect[0])
                    qc.ecr(*self.connect[0])
                if axis == 'x':
                    qc.h(0)
                elif axis == 'y':
                    qc.sdg(0)
                    qc.h(0)
                qc.measure_all()
                qc_list.append(qc)
        return qc_list

    def run(self,axis = 'z'):
        """_gate를 만들고 error 와 일반 ecr 결과를 class 내부에 만들어 주는 method_

        Args:
            axis (str, optional): _측정할 기저_. Defaults to 'z'.
        """
        self.job_stretch = {}
        self.job_stretch_odd = {}
        qc_list = self.make_circ()
        qc_list_odd = self.make_circ(odd=True)

        backend_ecr = update_ecr_real(self.l,self.layout,self.backend,self.ecr_stretch,amp_rate=self.amp_rate)
        passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_ecr, initial_layout=self.qc_connect)
        qc_input_ecr = passmanager.run(qc_list)
        with Batch(backend=backend_ecr):
            sampler = SamplerV2()
            job_test = sampler.run(qc_input_ecr)
            #job = sampler.run(qc_list,shots=8000)
            job_test.update_tags([self.class_id,'ecr',f'stretch = {self.ecr_stretch}',f'l = {self.l}'])
            self.job_test_list = job_test

        for stretch in self.stretch_list:
            backend_test = update_ecr(self.l,self.layout,self.backend,stretch,amp_rate=self.amp_rate)
            passmanager = generate_preset_pass_manager(optimization_level=0, backend=backend_test, initial_layout=self.qc_connect)
            qc_input_ecr = passmanager.run(qc_list)
            qc_input_ecr_odd = passmanager.run(qc_list_odd)
            with Batch(backend=backend_test):
                sampler = SamplerV2()
                job_test = sampler.run(qc_input_ecr)
                job_test.update_tags([self.class_id,'even','error',f'stretch = {stretch}',f'l = {self.l}',axis])
                self.job_stretch[stretch] = job_test
                job_test = sampler.run(qc_input_ecr_odd)
                job_test.update_tags([self.class_id,'odd','error',f'stretch = {stretch}',f'l = {self.l}',axis])
                self.job_stretch_odd[stretch] = job_test
    def get_data(self,class_id):
        service = self.backend.service
        self.job_stretch = service.jobs(job_tags = [class_id,'even'])
        self.job_stretch_odd = service.jobs(job_tags =[class_id,'odd'])
        self.job_test_list = service.jobs(job_tags =[class_id,'ecr'])