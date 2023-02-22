    #def random_circuit (int:num_qubits, int:depth, list:basis_gates):
    #     This is qiskit random.random_circuit function 
    #     https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/circuit/random/utils.py
    #     added basis_gate list argument
    #     removed arguments max_operands, measure, conditional, reset, seed 
    #     removed 3 and 4 gates
    #     num_qubits : integer value that is the number of qubits.
    #     depth: integer value that is the depth for the random circuit.
    #     basis_gates : ilist that contains the basis gates to generate the quantum circuit.
    #     Return the quantum circuit

    import numpy as np

    from qiskit.circuit import ClassicalRegister, QuantumCircuit, CircuitInstruction
    from qiskit.circuit import Reset
    from qiskit.circuit.library import standard_gates
    from qiskit.circuit.exceptions import CircuitError

    def random_circuit(
        num_qubits, depth, basis_gates
    ):
        if num_qubits == 0:
            return QuantumCircuit()
        gates = create_basis_gate_list(basis_gates)    
        gates = np.array(
            gates, dtype=[("class", object), ("num_qubits", np.int64), ("num_params", np.int64)]
        )
        gates = np.array(gates, dtype=gates.dtype)

        qc = QuantumCircuit(num_qubits)

        #if measure or conditional:
        cr = ClassicalRegister(num_qubits, "c")
        qc.add_register(cr)

        #if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
        rng = np.random.default_rng(seed)

        qubits = np.array(qc.qubits, dtype=object, copy=True)

        # Apply arbitrary random operations in layers across all qubits.
        for _ in range(depth):
            # We generate all the randomness for the layer in one go, to avoid many separate calls to
            # the randomisation routines, which can be fairly slow.

            # This reliably draws too much randomness, but it's less expensive than looping over more
            # calls to the rng. After, trim it down by finding the point when we've used all the qubits.
            gate_specs = rng.choice(gates, size=len(qubits))
            cumulative_qubits = np.cumsum(gate_specs["num_qubits"], dtype=np.int64)
            # Efficiently find the point in the list where the total gates would use as many as
            # possible of, but not more than, the number of qubits in the layer.  If there's slack, fill
            # it with 1q gates.
            max_index = np.searchsorted(cumulative_qubits, num_qubits, side="right")
            gate_specs = gate_specs[:max_index]
            slack = num_qubits - cumulative_qubits[max_index - 1]
            if slack:
                gate_specs = np.hstack((gate_specs, rng.choice(gates, size=slack)))

            # For efficiency in the Python loop, this uses Numpy vectorisation to pre-calculate the
            # indices into the lists of qubits and parameters for every gate, and then suitably
            # randomises those lists.
            q_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            p_indices = np.empty(len(gate_specs) + 1, dtype=np.int64)
            q_indices[0] = p_indices[0] = 0
            np.cumsum(gate_specs["num_qubits"], out=q_indices[1:])
            np.cumsum(gate_specs["num_params"], out=p_indices[1:])
            parameters = rng.uniform(0, 2 * np.pi, size=p_indices[-1])
            rng.shuffle(qubits)


            is_conditional = rng.random(size=len(gate_specs)) < 0.1
            condition_values = rng.integers(
                0, 1 << min(num_qubits, 63), size=np.count_nonzero(is_conditional)
            )
            c_ptr = 0
            for gate, q_start, q_end, p_start, p_end, is_cond in zip(
                gate_specs["class"],
                q_indices[:-1],
                q_indices[1:],
                p_indices[:-1],
                p_indices[1:],
                is_conditional,
            ):
                operation = gate(*parameters[p_start:p_end])
                if is_cond:
                    operation.condition = (cr, condition_values[c_ptr])
                    c_ptr += 1
                qc._append(CircuitInstruction(operation=operation, qubits=qubits[q_start:q_end]))



        qc.measure(qc.qubits, cr)

        return qc

    def create_basis_gate_list (basis_gates): 

        gates = []
        if 'R' in basis_gates:
            gates.append((Reset, 1, 0))
        if 'I' in basis_gates:
            gates.append((standard_gates.IGate, 1, 0))
        if 'SX' in basis_gates:
            gates.append((standard_gates.SXGate, 1, 0))
        if 'X' in basis_gates:
            gates.append((standard_gates.XGate, 1, 0))
        if 'RZ' in basis_gates:
            gates.append((standard_gates.RZGate, 1, 1))
        if 'R' in basis_gates:
            gates.append((standard_gates.RGate, 1, 2)) 
        if 'H' in basis_gates:
            gates.append((standard_gates.HGate, 1, 0))
        if 'P' in basis_gates:
            gates.append((standard_gates.PhaseGate, 1, 1))
        if 'RX' in basis_gates:
            gates.append((standard_gates.RXGate, 1, 1))
        if 'RY' in basis_gates:
            gates.append((standard_gates.RYGate, 1, 1))
        if 'S' in basis_gates:
            gates.append((standard_gates.SGate, 1, 0)) 
        if 'Sdg' in basis_gates:
            gates.append((standard_gates.SdgGate, 1, 0))
        if 'SXdg' in basis_gates:
            gates.append((standard_gates.SXdgGate, 1, 0))
        if 'T' in basis_gates:
            gates.append((standard_gates.TGate, 1, 0))
        if 'Tdg' in basis_gates:
            gates.append((standard_gates.TdgGate, 1, 0))
        if 'U' in basis_gates:
            gates.append((standard_gates.UGate, 1, 3)) 
        if 'U1' in basis_gates:
            gates.append((standard_gates.U1Gate, 1, 1))
        if 'U2' in basis_gates:
            gates.append((standard_gates.U2Gate, 1, 2))
        if 'U3' in basis_gates:
            gates.append((standard_gates.U3Gate, 1, 3))
        if 'Y' in basis_gates:
            gates.append((standard_gates.YGate, 1, 0))
        if 'Z' in basis_gates:
            gates.append((standard_gates.ZGate, 1, 0)) 
        if 'CX' in basis_gates:
            gates.append((standard_gates.CXGate, 2, 0))
        if 'DCX' in basis_gates:
            gates.append((standard_gates.DCXGate, 2, 0))
        if 'CH' in basis_gates:
            gates.append((standard_gates.CHGate, 2, 0))
        if 'CP' in basis_gates:
            gates.append((standard_gates.CPhaseGate, 2, 1))
        if 'CRX' in basis_gates:
            gates.append((standard_gates.CRXGate, 2, 1))
        if 'CRY' in basis_gates:
            gates.append((standard_gates.CRYGate, 2, 1))
        if 'CRZ' in basis_gates:
            gates.append((standard_gates.CRZGate, 2, 1))
        if 'CSX' in basis_gates:
            gates.append((standard_gates.CSXGate, 2, 0))
        if 'CU' in basis_gates:        
            gates.append((standard_gates.CUGate, 2, 4))
        if 'CU1' in basis_gates:
            gates.append((standard_gates.CU1Gate, 2, 1))
        if 'CU3' in basis_gates:        
            gates.append((standard_gates.CU3Gate, 2, 3))
        if 'CY' in basis_gates:        
            gates.append((standard_gates.CYGate, 2, 0))
        if 'CZ' in basis_gates:        
            gates.append((standard_gates.CZGate, 2, 0))
        if 'RXX' in basis_gates:        
            gates.append((standard_gates.RXXGate, 2, 1))
        if 'RYY' in basis_gates:        
            gates.append((standard_gates.RYYGate, 2, 1))
        if 'RZZ' in basis_gates:        
            gates.append((standard_gates.RZZGate, 2, 1))
        if 'RZX' in basis_gates:        
            gates.append((standard_gates.RZXGate, 2, 1))
        if 'XX-YY' in basis_gates:        
            gates.append((standard_gates.XXMinusYYGate, 2, 2))
        if 'XX+YY' in basis_gates:   
            gates.append((standard_gates.XXPlusYYGate, 2, 2))
        if 'ECR' in basis_gates:
            gates.append((standard_gates.ECRGate, 2, 0))
        if 'CS' in basis_gates:
            gates.append((standard_gates.CSGate, 2, 0))
        if 'CSdg' in basis_gates:
            gates.append((standard_gates.CSdgGate, 2, 0))
        if 'Swap' in basis_gates:
            gates.append((standard_gates.SwapGate, 2, 0))
        if 'iSwap' in basis_gates:        
            gates.append((standard_gates.iSwapGate, 2, 0))
        #print(gates)
        return gates
        
 basis_gate_list = ['R','I','SX','X','RZ','R','H','P','RX','RY','S','Sdg','SXdg',
'T','Tdg','U','U1','U2','U3','Y','Z','CX','DCX','CH','CP','CRX','CRY','CRZ','CSX',
'CU','CU1','CU3','CY','CZ','RXX','RYY','RZZ','RZX','XX-YY','XX+YY','ECR','CS','CSdg'
,'Swap','iSwap']
#basis_gate_list=['I','X','Z','CX','Swap']
circuit=random_circuit(2, 8,basis_gate_list )
circuit.draw()
