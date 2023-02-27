# QOSF-screening-tasks
# overwritten qiskit random.random_circuit located here https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/circuit/random/utils.py by doing the following
#     added basis_gate list argument
#     removed arguments max_operands, measure, conditional, reset, seed 
#     removed 3 and 4 gates
#     created a function create_basis_gate_list to set the basis gates for roandom_circuit function
