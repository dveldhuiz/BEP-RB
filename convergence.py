#Import general libraries 
import numpy as np

#Import Qiskit classes
import qiskit
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, kraus_error, pauli_error
from qiskit.quantum_info import Pauli
#Import the RB Functions
import qiskit.ignis.verification.randomized_benchmarking as rb

#Import my own modules
import modules.rb_fitter as rb_fitter
import modules.rb_execute as rb_execute
import modules.rb_qiskit_fit_plot as rb_qiskit_fit_plot


import copy
from itertools import product

# from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
# from qiskit.transpiler.passes import BasisTranslator
# from qiskit.circuit import QuantumCircuit
# from qiskit.converters import circuit_to_dag, dag_to_circuit
import os



#%%


class Experiment:
    
    """
    For each experiment (e.g. benchmarking the pattern [[0],[1,2]]) I create an instance of Experiment
    """
    def __init__(self, nQ, rb_opts, basis_gates, noise_model=None, shots = 10000, backend = qiskit.Aer.get_backend('qasm_simulator'), complete=False):
        self.nQ=nQ 
        self.rb_opts=rb_opts
        self.basis_gates=basis_gates
        self.noise_model=noise_model
        self.shots=shots
        self.backend=backend
        self.result_list=[]
        self.transpile_list=[]
        self.rb_circs, self.xdata = rb.randomized_benchmarking_seq(**self.rb_opts)
        if complete == True:
            """ If complete is true each qubit is coupled to every other qubit"""
            self.coupling_map = qiskit.transpiler.CouplingMap.from_full(nQ)
        else: 
            """Create a coupling map that couples only nearest neighbours, make is symmetric (apparently this is necessary) and sets it"""
            coupling=qiskit.transpiler.CouplingMap([[i, i+1] for i in range(nQ-1)])
            coupling.make_symmetric()
            self.coupling_map = coupling
    
    def execute(self):
        """Does the transpilation as well as execution, see modules.rb_execute"""
        self.result_list, self.transpile_list = rb_execute.execute(self.rb_circs, self.basis_gates, 
                                                                   self.noise_model, self.shots, self.backend,
                                                                   coupling_map=self.coupling_map)   
        return self.result_list, self.transpile_list 
    
    def transpile(self):
        """Does tranpilation only the get the gates per clifford for n>2 benchmarking, this code is nearly identical to that of the qiskit RB tuto"""
        transpile_list = []
        for rb_seed,rb_circ_seed in enumerate(self.rb_circs):
            print('Compiling seed %d'%rb_seed)
            rb_circ_transpile = qiskit.transpile(rb_circ_seed, basis_gates=self.basis_gates, coupling_map=self.coupling_map)
            transpile_list.append(rb_circ_transpile)
        print("Finished Transpiling")
        self.transpile_list = transpile_list
    
    def rb_fit(self):
        """Fits using qiskits RBFitter"""
        if len(self.result_list) != 0:
            self.rbfit = rb.fitters.RBFitter(self.result_list, self.xdata, self.rb_opts['rb_pattern'])
            
            # self.pattern_params={}
            # for pttrn in self.rbfit.fit:
            #    self.pattern_params.append(pttrn['params'][1])
            
            """Formats the result of the fit as a dict {(0,): alpha_0, (1,2): alpha_12} for the pattern [[0],[1,2]]"""  
            self.pattern_params=dict(zip([tuple(pttrn) for pttrn in self.rb_opts['rb_pattern']],[pttrn['params'][1] for pttrn in self.rbfit.fit]))
            
            params_err=[np.array([p['params'][1],p['params'][1]-p['params_err'][1], p['params'][1]+p['params_err'][1]]) for p in self.rbfit.fit]
            self.pattern_params_err=dict(zip([tuple(pttrn) for pttrn in self.rb_opts['rb_pattern']],params_err))

        
            return self.rbfit    
        
    def plot_fit(self):
        """Plots the RBfit"""
        rb_qiskit_fit_plot.plot_fit(self.result_list, self.rbfit, self.rb_opts)
        
    def plot_own_fit(self):
        """ My own fitter (probably broken) and plotter"""
        survival_prob=rb_fitter.get_survival_prob(self.result_list, self.rb_opts['length_vector'])
        popt, pcov = rb_fitter.get_fit_param(survival_prob, nCliffs)
        rb_fitter.plot_fit(popt, pcov, self.result_list, self.rb_opts['length_vector'])
    
    def get_gpc(self, qubit_i, pattern_j):
        """Wrapper for qiskits gates_per_clifford"""
        return rb.rb_utils.gates_per_clifford( transpiled_circuits_list=self.transpile_list, 
        clifford_lengths=self.xdata[pattern_j], basis=self.basis_gates, qubits=[qubit_i])
    
    def get_sum_gpc(self):
        
        """Counts the number of 1 qubit basis gates per 1 qubit clifford, 
                                1 qubit basis gates per 2 qubit clifford, 
                                2 qubit basis gates per 2 qubit clifford
        Loops through the subpatterns, 
            if the subpattern is a 1 qubit benchmark it will increase N_1_per_1, 
                it also tries to count 2 qubit gates due to swaps but this is not relevant with my current implementation. 
            if the subpattern is a 2 qubit benchmark it will increase N_1_per_2 and N_2_per_2
            N_2_per_2 are devided by 2 because they are counted double.
        """
        N_1_per_1=0
        N_1_per_2=0
        N_2_per_2=0
        
        for j, sub_pattern in enumerate(self.rb_opts['rb_pattern']):
            if len(sub_pattern)==1:
                for i in sub_pattern:
                   for gate, value in self.get_gpc(i,j)[i].items():
                       if gate in ['u1', 'u2', 'u3']:
                           N_1_per_1+=value
                       elif gate in ['cx']:
                           N_2_per_2+=value/2.
            if len(sub_pattern)==2:
                for i in sub_pattern:
                   for gate, value in self.get_gpc(i,j)[i].items():
                       if gate in ['u1', 'u2', 'u3']:
                           N_1_per_2+=value
                       elif gate in ['cx']:
                           N_2_per_2+=value/2.
        return N_1_per_1, N_1_per_2, N_2_per_2 
    
    def get_gpc2(self):
        """Wrapper for qiskits gates_per_clifford"""
        return rb.rb_utils.gates_per_clifford( transpiled_circuits_list=self.transpile_list, 
        clifford_lengths=self.xdata, basis=self.basis_gates, qubits=[i for i in range(self.nQ)])
    
    def get_sum_gpc2(self):
        
        """Counts the number of 1 qubit basis gates per 1 qubit clifford, 
                                1 qubit basis gates per 2 qubit clifford, 
                                2 qubit basis gates per 2 qubit clifford
        Loops through the subpatterns, 
            if the subpattern is a 1 qubit benchmark it will increase N_1_per_1, 
                it also tries to count 2 qubit gates due to swaps but this is not relevant with my current implementation. 
            if the subpattern is a 2 qubit benchmark it will increase N_1_per_2 and N_2_per_2
            N_2_per_2 are devided by 2 because they are counted double.
        """
        N_1_per_1=0
        N_1_per_2=0
        N_2_per_2=0
        
        for sub_pattern in self.rb_opts['rb_pattern']:
            if len(sub_pattern)==1:
                for i in range(self.nQ):
                   for gate, value in self.get_gpc2()[i].items():
                       if gate in ['u1', 'u2', 'u3']:
                           N_1_per_1+=value
                       elif gate in ['cz']:
                           N_2_per_2+=value/2.
            if len(sub_pattern)==2:
                for i in range(self.nQ):
                   for gate, value in self.get_gpc2()[i].items():
                       if gate in ['u1', 'u2', 'u3']:
                           N_1_per_2+=value
                       elif gate in ['cz']:
                           N_2_per_2+=value/2.
        return N_1_per_1, N_1_per_2, N_2_per_2 
    
    
    def get_3q_gpc(self):
        N_1_per_3=0
        N_2_per_3=0

        """ counts the number of 1, 2 qubit basis gates per 3 qubit clifford 
        N_2_per_3 are devided by 2 because they are counted double."""
        for j, sub_pattern in enumerate(self.rb_opts['rb_pattern']):
            for i in sub_pattern:
                   for gate, value in self.get_gpc(i,j)[i].items():
                       if gate in ['u1', 'u2', 'u3']:
                           N_1_per_3+=value
                       elif gate in ['cz']:
                           N_2_per_3+=value/2.
        return N_1_per_3, N_2_per_3
        
    @staticmethod
    def get_epc(alpha, n):
        return (2.**n-1)*(1-alpha)/(2.**n)
        
               
class FullExperiment: 
    
    """A full experiment is basically a list of single experiments that 
    combined perform 1 and 2 qubit benchmarking on each qubit at least once
    The experiments are simply stored as a list within this object
    Also added some convenient methods for executing, fitting and plotting"""
    def __init__(self, experiments=None):
        if experiments==None:
            self.experiments=[]
        else:
            self.experiments=experiments
       
    def add_exp(self, experiment):
        if experiment not in self.experiments:
            print(experiment.rb_opts['rb_pattern'])
            self.experiments.append(experiment)
            
    def remove_exp(self, experiment):
        if experiment in self.experiments:
            self.experiments.remove(experiment)
            
    def execute_all(self):
        for exp in self.experiments:
            exp.execute()
            
    def fit_all(self):
        for exp in self.experiments:
            exp.rb_fit() 
    
    def plot_all(self):
        for exp in self.experiments:
            exp.plot_fit()          
   


class FullExperimentNQ(FullExperiment):
    
    """Subclass of FullExperiment (this a bit pointless but I haven't changed it yet)"""
    def __init__(self, experiments=None):
       super().__init__(experiments)
       
    def get_1q_basis_alphas(self):
        """Calculates the alpha parameter for 1 qubit basis gates on each qubit using equation 9 in the Three Qubit RB supplement"""
        alphas_1q = {}
        for experiment in self.experiments:
            # N_1_per_1, N_1_per_2, N_2_per_2 = experiment.get_sum_gpc()
            N_1_per_1 = 0.959808
            # N_1_per_1=1.5
            """pattern params is a dict such that {(0,): alpha_0_c, (1,2): alpha_12_c} for the pattern [[0],[1,2]]"""
            pattern_params_err=experiment.pattern_params_err
            """Creates a dictioray with the calculated 1q basis alpha for each qubit eg. {0: alpha_0, 1: alpha_1, 2: alpha_2} """
            for qubits, val in pattern_params_err.items():
                 if len(qubits)==1:
                     alphas_1q[qubits[0]]=val**(1./N_1_per_1)
        return alphas_1q
                     
    def get_2q_basis_alphas(self):
        """Calculates the alpha parameters for 2 qubit basis gates on each qubit using equation 12 in the Three Qubit RB supplement"""
        alphas_2q = {}
        alphas_1q = self.get_1q_basis_alphas()
        for experiment in self.experiments:
            # N_1_per_1, N_1_per_2, N_2_per_2 = experiment.get_sum_gpc()
            N_1_per_2, N_2_per_2 = [4.42531,	1.5079]
            # N_1_per_2=1.6
            # N_2_per_2=1.3
            pattern_params_err=experiment.pattern_params_err
            """Creates a dictioray with the calculated 2q basis alpha for each benchmarked qubit pair eg. {(0,1): alpha_01, (1,2): alpha_12} """
            for qubits, val in pattern_params_err.items():
                 if len(qubits)==2:
                     a_01_c=val
                     a_0=alphas_1q[qubits[0]]**(N_1_per_2/2.)
                     a_1=alphas_1q[qubits[1]]**(N_1_per_2/2.)
                     alphas_2q[tuple(qubits)]=(5*a_01_c/(a_0+a_1+3*a_0*a_1))**(1./N_2_per_2)                 
        return alphas_2q
    
    

def generate_paulis(n):
    """returns a itertools.product object containing each n-qubit pauli e.g. for n=2 returns II, IX, IY, IZ, XX, etc. """
    return product('IXYZ', repeat=n)

def get_nq_alpha_c(n, alphas_1q, alphas_2q, N1, N2):
    """Estimates the n-qbubit alpha from the 1 and 2 basis alphas and the number of these per n-qubit Cliffor: N1, N2"""
    #initialise dictionary to story diagonal matrices as arrays
    arrays={}
    #loop over all individual qubits q_i with respective basis alpha
    for q_i, alpha in alphas_1q.items():
        diag=np.outer( (alpha**(N1/n)) , np.ones(4**n) )# initialise the diagonal as all alphas to the power N1/n
        for pauli_i, pauli in enumerate(generate_paulis(n)): #loop over the enumerated list of paulis
            if pauli[q_i] == 'I': #if the single qubit (sub)pauli acting on qubit q_i is the identity
                diag[:,pauli_i] = 1 #change its diagonal element to 1
        arrays[q_i]=diag 
    
    n_2q=len(alphas_2q) #count the number of 2 qubit pairs being benchmarked 
    #loop over all benchmarked qubit pairs q_pair with respective basis alpha
    for q_pair, alpha in alphas_2q.items(): 
        diag=np.outer( (alpha**(N2/n_2q)) , np.ones(4**n) )# initialise the diagonal as all alphas to the power N2/n_2q
        for pauli_i, pauli in enumerate(generate_paulis(n)):
            if pauli[q_pair[0]] == 'I' and pauli[q_pair[1]] == 'I': #if the 2 qubit (sub)pauli acting on pair q_pair is the identity
                diag[:,pauli_i] = 1 #change its diagonal element to 1
        arrays[q_pair]=diag  
        
    """First convert the dict of arrays to a list of arrays. Then vstack these arrays, turning them into a 2 dimensional array (matrix), then multiply the matrix across the correct axis.
    This has the result of multiplying all the arrays componentwise, in practise this multiplies diagonal Pauli transfer matrices.
    It then sums the entries, effectively taking the trace. Substract the first entry (which is always 1) and devide by the correct factor"""
    result=np.zeros(3)
   
    for i in range(3):
        single_arrays=[array[i] for array in list(arrays.values())]
        result[i]= np.divide((np.sum(np.prod(np.vstack(single_arrays), axis=0))-1),(4**n-1))
    return result

def get_random_patterns(n):
    pairs=[[i,i+1] for i in range(1,n)]
    singlets=[[i] for i in range(1,n+1)]
    total=[None]*(len(pairs)+len(singlets))
    total[::2]=singlets
    total[1::2]=pairs
    total=[0,0]+total+[0,0]
    
    reference=np.array(total,dtype=list)
    
    array=np.array(total,dtype=list)
    patterns=[]
    pattern=[]
    len_multiplier=[]
    while len(reference.nonzero()[0])!=0:   
        if len(array.nonzero()[0])!=0:
            indexis=array.nonzero()[0]
            choice=np.random.choice(indexis, replace=False)
            subpattern=array[choice]
            if len(subpattern)==1:
                len_multiplier.append(3)
                array[choice-1:choice+2]=0
            if len(subpattern)==2:
                len_multiplier.append(1)
                array[choice-2:choice+3]=0  
            reference[choice]=0
            pattern.append([i-1 for i in subpattern])
        else:
            patterns.append([pattern,len_multiplier])
            pattern=[]
            len_multiplier=[]
            array=copy.copy(reference)
    patterns.append([pattern,len_multiplier])     
    return patterns

def get_random_patterns2(n):
    reference=np.triu(np.ones((n,n)))
    
    
    matrix=np.triu(np.ones((n,n)))
    patterns=[]
    pattern=[]
    len_multiplier=[]
    while len(reference[reference.nonzero()])!=0:   
        if len(matrix[matrix.nonzero()])!=0:
            indexis=matrix.nonzero()
            choice=np.random.choice(range(len(indexis[0])))
            c1=indexis[0][choice]
            c2=indexis[1][choice]
            
            if c1==c2:
                subpattern=[c1]
                len_multiplier.append(3)
            else:
                subpattern=[c1,c2]
                len_multiplier.append(1)
                
            matrix[c1,:]=np.zeros(n)
            matrix[c2,:]=np.zeros(n)
            matrix[:,c1]=np.zeros(n)
            matrix[:,c2]=np.zeros(n)
            reference[c1,c2]=0
            pattern.append(subpattern)
        else:
            patterns.append([pattern,len_multiplier])
            pattern=[]
            len_multiplier=[]
            matrix=copy.copy(reference)
    patterns.append([pattern,len_multiplier])     
    return patterns     

     
# %%



#Number of Cliffords in the sequence (start, stop, steps)
nCliffs = np.arange(1,401,20)

basis_gates = ['u1','u2','u3','cz']

    
# %%


#Defining the noise model
noise_model = NoiseModel()
p1Q = 0.002
p2Q = 0.01
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u1')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u2')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), 'u3')
noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), 'cz')
#'rx(pi/2)', u, cz

# Dephasing noise
noise_model2 = NoiseModel()
K_0=np.sqrt(1-(p1Q/2.))*Pauli('I').to_matrix()
K_1=np.sqrt(p1Q/2.)*Pauli('Z').to_matrix()
dephasing_error=kraus_error([K_0,K_1])
dephasing_error2q=pauli_error([('ZZ',p2Q),('II',1-p2Q)])
noise_model2.add_all_qubit_quantum_error(dephasing_error, ['u1', 'u2', 'u3'])
noise_model2.add_all_qubit_quantum_error(dephasing_error2q, 'cz')



# Kraus depolarizing noise
noise_model3 = NoiseModel()
K_0=np.sqrt(1-(3*p1Q/4.))*Pauli('I').to_matrix()
K_1=np.sqrt(p1Q/4.)*Pauli('X').to_matrix()
K_2=np.sqrt(p1Q/4.)*Pauli('Y').to_matrix()
K_3=np.sqrt(p1Q/4.)*Pauli('Z').to_matrix()
depolar_error=kraus_error([K_0,K_1,K_2,K_3])
noise_model3.add_all_qubit_quantum_error(depolar_error, ['u1', 'u2', 'u3'])

for nQ in [4]:
    for nseeds in [30]:
        rb_opts = {}
        rb_opts['length_vector'] = nCliffs
        rb_opts['nseeds'] = nseeds
        
        patterns=get_random_patterns2(nQ)
        #get_random_patterns2(nQ)
        exp1=FullExperimentNQ()
        for rb_pattern in patterns: # adds individual experiments to a FullExperiment(uses copy.deepcopy to avoid weird behaviour)
            rb_opts_copy=copy.deepcopy(rb_opts)
            rb_opts_copy['rb_pattern'] = rb_pattern[0]
            rb_opts_copy['length_multiplier'] = rb_pattern[1]
            print(rb_opts_copy)
            exp1.add_exp(Experiment(nQ, rb_opts_copy, basis_gates, noise_model))
        
        exp2=FullExperimentNQ()
        for rb_pattern in patterns: # adds individual experiments to a FullExperiment(uses copy.deepcopy to avoid weird behaviour)
            rb_opts_copy=copy.deepcopy(rb_opts)
            rb_opts_copy['rb_pattern'] = rb_pattern[0]
            rb_opts_copy['length_multiplier'] = rb_pattern[1]
            print(rb_opts_copy)
            exp2.add_exp(Experiment(nQ, rb_opts_copy, basis_gates, noise_model2))
        
        exps=[exp1, exp2]
        
        # Creates a 3Q RB experiment 
        rb_opts_copy=copy.deepcopy(rb_opts)
        if nQ == 3:
            len_vec=np.arange(1,90,3)
        if nQ == 4:
            len_vec=np.arange(1,60,2)
        if nQ == 5:
            len_vec=np.arange(1,30,1)
        if nQ == 6:
            len_vec=np.arange(1,30,1)    
        rb_opts_copy['length_vector']=len_vec
        rb_opts_copy['length_multiplier'] = [1]
        rb_opts_copy['rb_pattern'] = [[i for i in range(nQ)]]
        
        
        three_q1=Experiment(nQ, rb_opts_copy, basis_gates, noise_model, shots=10000) 
        three_q2=Experiment(nQ, rb_opts_copy, basis_gates, noise_model2, shots=10000)
        
        fulls=[three_q1, three_q2]
        
        data=np.zeros((2,4))
        for i, exp in enumerate(exps):
            #Execute all Experiments in exp
            exp.execute_all()
            exp.fit_all()
            
            #Execute the 3Q experiment
           
            fulls[i].execute()
            #fit and plot all Experiments in exp
           
            
            # fit and plot the 3Q experiment
            fulls[i].rb_fit()
            
            nq_result=fulls[i].rbfit.fit[0]['params'][1]
            nq_conf=fulls[i].rbfit.fit[0]['params_err'][1]
        
    
            #Calculate the 1q and 2q basis alphas 
            q1=exp.get_1q_basis_alphas()
            q2=exp.get_2q_basis_alphas()
            
            N1, N2 = fulls[i].get_3q_gpc()
    
            #Calculate the predicted 3Q RB Clifford alpha
            predictions=get_nq_alpha_c(nQ, q1, q2, N1, N2)
            prediction=predictions[1]
            prediction_conf=predictions[0]-predictions[1]
            
           
            
            data[i]=np.array([nq_result,nq_conf,prediction,prediction_conf])
            
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_path = f"{nQ}q_{nseeds}seeds.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        np.savetxt(abs_file_path,data, fmt='%2.5f')
            



    



    