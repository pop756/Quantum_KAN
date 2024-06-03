import pennylane as qml
from pennylane import numpy as np
import torch

def extra_polation(circ,H,theta,noise_factor=[1,3,5],p1=0.01,p2=0.02):
    """게이트 폴딩으로 노이즈를 만들고 그 결과 출력

    Args:
        circ (qnode): _들어갈 서킷_
        H (qml.Hamiltonian): _측정할 기저_
        theta (list): _서킷에 임의로 넣을 변수값_
        noise_factor (list, optional): _게이트 폴딩 노이즈 팩터(2n+1)_. Defaults to [1,3,5].
        p1 (float, optional): _CNOT dephasing 확률_. Defaults to 0.01.
        p2 (float, optional): _CNOT bitflip 확률_. Defaults to 0.02.

    Returns:
        res(list),real_value(float): 노이즈 팩터 리스트,실제 서킷 값 
    """
    circ(theta)
    dev = circ.device
    ops = circ.qtape.operations
    ops_inv = ops[::-1]
    noise_factor1 = (np.array(noise_factor)+1)/2-1
    noise_factor = np.asarray(noise_factor1,dtype=int)
    
    if (noise_factor != noise_factor1).all():
        print("input is not suitable(2n+1)")
        raise
    
    @qml.qnode(dev)
    def real_circ():
        tensor = torch.tensor
        
        for op in ops:
            eval(f'qml.{op}')
        return qml.expval(H)
    
    def noise_circ():
        tensor = torch.tensor
        for op in ops:
            eval(f'qml.{op}')
            if len(op.wires)>1:
                for wire in op.wires:
                    qml.DepolarizingChannel(p1, wires=wire)
                    qml.BitFlip(p2, wires=wire)
            else:
                qml.DepolarizingChannel(p1, wires=op.wires) 
                qml.BitFlip(p2, wires=op.wires)
    def noise_circ_inv():
        tensor = torch.tensor
        for op in ops_inv:
            eval(f'qml.adjoint(qml.{op})')
            if len(op.wires)>1:
                for wire in op.wires:
                    qml.DepolarizingChannel(p1, wires=wire)
                    qml.BitFlip(p2, wires=wire)
            else:
                qml.DepolarizingChannel(p1, wires=op.wires) 
                qml.BitFlip(p2, wires=op.wires)
    
    @qml.qnode(dev,interface='torch')
    def extra_polation(factor):
        for i in range(factor):
            noise_circ()
            noise_circ_inv()
        noise_circ()
        return qml.expval(H)
    




    res = []
    real_value = real_circ()
    for factor in noise_factor:
        res.append(extra_polation(factor).detach().numpy())
    return res,real_value

def extra_polation_time(circ,H,theta,noise_factor=[1,1.5,2],p1=0.01,p2=0.02):
    """노이즈를 키워 서킷을 만들고 그 노이즈 서킷의 결과 출력

    Args:
        circ (qnode): _들어갈 서킷_
        H (qml.Hamiltonian): _측정할 기저_
        theta (list): _서킷에 임의로 넣을 변수값_
        noise_factor (list, optional): _time 노이즈 팩터_. Defaults to [1,1.5,2].
        p1 (float, optional): _CNOT dephasing 확률_. Defaults to 0.01.
        p2 (float, optional): _CNOT bitflip 확률_. Defaults to 0.02.

    Returns:
        res(list),real_value(float): 노이즈 팩터 리스트,실제 서킷 값 
    """
    circ(theta)
    dev = circ.device
    ops = circ.qtape.operations
    @qml.qnode(dev)
    def real_circ():
        tensor = torch.tensor
        
        for op in ops:
            eval(f'qml.{op}')
        return qml.expval(H)
            
    @qml.qnode(dev)
    def noise_circ(factor):
        tensor = torch.tensor
        
        for op in ops:
            if len(op.wires)>1:
                """
                for wire in op.wires:
                    qml.DepolarizingChannel(p1*factor, wires=wire)
                    qml.BitFlip(p2*factor, wires=wire)"""

                eval(f'qml.{op}')
                for wire in op.wires:
                    qml.DepolarizingChannel(p1*factor, wires=wire)
                    qml.BitFlip(p2*factor, wires=wire)
            else:
                eval(f'qml.{op}')
        return qml.expval(H)
    
    real_value = real_circ()
    res = []
    for factor in noise_factor:
        res.append(noise_circ(factor).detach().numpy())
    return res,real_value
    






import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 선형 함수 정의
def linear_func(x, a, b):
    return a * x + b

# 지수 함수 정의
def exp_func(x, a, b, c):
    return b*a **(x) + c

# 데이터 피팅 및 외삽 함수
def extrapolate_values(y_data,x_data = np.array([1, 3, 5, 7])):
    # 선형 함수 피팅
    params_linear, _ = curve_fit(linear_func, x_data, y_data)
    a_linear, b_linear = params_linear
    y_linear_0 = linear_func(0, a_linear, b_linear)

    # 지수 함수 피팅 초기 추정값 및 범위 설정
    initial_guess = [0.8, 1, 1]
    bounds = ([0, -5, -np.inf], [1, 5, np.inf])

    params_exp, _ = curve_fit(exp_func, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=10000)
    a_exp, b_exp, c_exp = params_exp
    y_exp_0 = exp_func(0, a_exp, b_exp, c_exp)

    return y_linear_0, y_exp_0, params_linear, params_exp

def curve_plot(x,y):
    # 선형 함수 정의
    def linear_func(x, a, b):
        return a * x + b

    # 지수 함수 정의
    def exp_func(x, a, b, c):
        return b*a **(x) + c
    
    # 선형 함수 피팅
    params_linear, _ = curve_fit(linear_func, x, y)

    # 지수 함수 피팅 초기 추정값 및 범위 설정
    initial_guess = [0.8, 1, 1]
    bounds = ([0, -5, -np.inf], [1, 5, np.inf])

    params_exp, _ = curve_fit(exp_func, x, y, p0=initial_guess, bounds=bounds, maxfev=10000)
    
    # 맞춘 함수 플롯
    x_fit = np.linspace(0, max(x), 100)
    y_fit_linear = linear_func(x_fit, *params_linear)
    y_fit_exp = exp_func(x_fit, *params_exp)
    
    
    plt.plot(x_fit, y_fit_linear, label='Linear fitted function')
    plt.plot(x_fit, y_fit_exp , label='Exp fitted function')