# Import the necessary libraries
from qiskit import QuantumCircuit, transpile, Aer, execute
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from qiskit.quantum_info import DensityMatrix
import torch
import pickle #For exporting the variables

from CST import Shadow


def initialize_all_zeros(nr_qubits):
    # Create a quantum circuit with N qubits
    qc = QuantumCircuit(nr_qubits)
    
    # Initialize all qubits in the |0⟩ state
    for qubit in range(nr_qubits):
        qc.initialize([1, 0], qubit)
    
    return qc

def apply_cnot_chain(qc, control_qubit, nr_qubits):
    # Apply CNOT gates from the control qubit to the list of target qubits
    for target_qubit in range(nr_qubits-1):
        qc.cx(control_qubit, target_qubit+1)
        
def measurement_bases_N(nr_qubits):
    # Generate nr_qubits random numbers from the set {x=1, y=2, z=3}
    # random_bases = np.random.choice([1, 2, 3], size=nr_qubits) # X=1 Y=2 Z=3

    random_bases = np.random.choice([1, 2, 3], size=nr_qubits)
    return random_bases

def run_circuit_and_obtain_shadows(nr_qubits,n_sample):
    # Initialize a list to store the rows
    rows = []

    # Initialize an empty list to store the measured result and bases in the correct format for the AI model
    measurement_results_in_specific_format = np.zeros((n_sample,nr_qubits), dtype=int)
    measurement_bases_in_specific_format = []

    for _ in range(n_sample):
                
        # Create a quantum circuit with N qubits
        qc = initialize_all_zeros(nr_qubits)

        control_qubit = 0 #Usually just the first qubit. Hardcoded because we don't need the control


        # Apply a Hadamard gate to qubit 0
        qc.h(control_qubit)

        apply_cnot_chain(qc, control_qubit, nr_qubits)

        # Choose random measurement basis for each qubit
        measurement_bases = measurement_bases_N(nr_qubits)
        
        # measurement_bases = np.array([1, 1, 1, 3]) #To check a specific case
        # print('measurement_bases', measurement_bases)

        # Store the measurement bases in the list    
        measurement_bases_in_specific_format.append(measurement_bases)
        

        # Apply the measurement bases to the qubits
        for qubit in range(nr_qubits):
            if measurement_bases[qubit] == 1:
                qc.h(qubit)
            elif measurement_bases[qubit] == 2:
                qc.sdg(qubit)
                qc.h(qubit)

        # Add measurements for all qubits in the Z basis
        qc.measure_all()
        
        # Simulate the circuit and get measurement results
        simulator = Aer.get_backend('qasm_simulator') #Priya is going to change these backends
        job = execute(qc, simulator, shots=1)
        result = job.result()
        counts = result.get_counts(qc)

        # Iterate through the qubits and add their measurement results to the list
        for qubit in range(nr_qubits):
            basis = measurement_bases[qubit]
            result = int(list(counts.keys())[0][nr_qubits - 1 - qubit])  # Extract the result
            rows.append(pd.DataFrame({"Measurement Basis": [basis], "Measured Result": [result]}))
            
            measurement_results_in_specific_format[_][qubit] = result
        
        # print(measurement_results_in_specific_format[_]) #To compare them to the ones in the paper

        # Concatenate the rows into the DataFrame and reset the index
        df = pd.concat(rows, ignore_index=True)

        

    # Display the DataFrame
    # print('df',df)

    obs_before_tensor = measurement_bases_in_specific_format
    # print('obs_before_tensor', obs_before_tensor)

    out_before_tensor = [np.array(row) for row in measurement_results_in_specific_format]
    # print('out_before_tensor', out_before_tensor)
    
    return obs_before_tensor, out_before_tensor

def ghz_shadow_qiskit_generated(nr_qubits,n_sample):
    obs_before_tensor, out_before_tensor = run_circuit_and_obtain_shadows(nr_qubits,n_sample)
    obs = torch.tensor(np.stack(obs_before_tensor))
    out = torch.tensor(np.stack(out_before_tensor))
    
    return Shadow(obs, out)
