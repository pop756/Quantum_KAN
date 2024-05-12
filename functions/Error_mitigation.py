import pennylane as qml
from pennylane import numpy as np



def extra_polation(circ,theta,noise_factor=[0,1,2,3],p1=0.01,p2=0.02):
    
    real_vaule = circ(theta)
    dev = circ.device
    ops = circ.qtape.operations
    ops_inv = ops[::-1]


    meas = circ.qtape.measurements[0]
    def noise_circ():
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
        for op in ops_inv:
            eval(f'qml.adjoint(qml.{op})')
            if len(op.wires)>1:
                for wire in op.wires:
                    qml.DepolarizingChannel(p1, wires=wire)
                    qml.BitFlip(p2, wires=wire)
            else:
                qml.DepolarizingChannel(p1, wires=op.wires) 
                qml.BitFlip(p2, wires=op.wires)
    
    @qml.qnode(dev)
    def extra_polation(factor):
        Z = qml.PauliZ
        Y = qml.PauliY
        X = qml.PauliX
        for i in range(factor):
            noise_circ()
            noise_circ_inv()
        noise_circ()
        return eval(f"qml.{meas}")
    




    res = []
    for factor in noise_factor:
        res.append(extra_polation(factor))
    return res,real_vaule