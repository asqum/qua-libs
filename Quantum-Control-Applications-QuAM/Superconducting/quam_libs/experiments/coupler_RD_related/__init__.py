from qualibrate import QualibrationNode
from qm.type_hinting import DictQuaConfig
from qm import QuantumMachinesManager, QuantumMachine
from quam_libs.components import QuAM
from qm.program.program import Program
from quam_libs.components.macros.coupler_RD_decoding import CouplerReadoutDecoder
from abc import ABC, abstractmethod
from time import time, sleep
from xarray import Dataset, concat



def load_data_only(node:QualibrationNode):
    ''' The coupler's name and load_data_id must be in the node.parameters '''
    node = node.load_from_id(node.parameters.load_data_id)
    ds = node.results["ds"] 
    machine = node.machine
    CRD = CouplerReadoutDecoder(machine, node.parameters.coupler, coupler_arbi_LO_manual=None, simulate=False, load_data_id=node.parameters.load_data_id)

    return node, machine, ds, CRD


class ExpTemplate(ABC):
    def __init__(self):
        self.load_data:bool = False
        self.progress_bar_display:bool = True
        self.node:QualibrationNode = None
        self.CRD:CouplerReadoutDecoder = None
        self.machine:QuAM = None
        self.config:DictQuaConfig = None
        self.qmm: QuantumMachinesManager = None
        self.qm: QuantumMachine = None
        self.variables: dict = {}
        self.participant_num:int = None
        self.qua_prog:Program = None
        self.analyzed_items:dict = {}

    @abstractmethod
    def participants_stand_by(self,*args,**kwargs):
        ''' check qubits or couplers for this experiment'''
        pass

    @abstractmethod
    def exp_variable_arangement(self,*args,**kwargs):
        ''' count all the exp variables '''
        pass

    @abstractmethod
    def qua_composer(self,*args,**kwargs):
        ''' compose your QUA program here '''
        pass
    
    @abstractmethod
    def qua_executor(self,*args,**kwargs)->Dataset:
        ''' running your QUA '''
        pass
    
    @abstractmethod
    def data_catcher(self,*args,**kwargs)->Dataset:
        ''' Data fetch '''
        pass

    @abstractmethod
    def analyze(self,*args,**kwargs):
        ''' Data analyze '''
        pass

    @abstractmethod
    def visualize(self,*args,**kwargs):
        ''' Results visualizations '''
        pass

    @abstractmethod
    def state_management(self,*args,**kwargs):
        ''' save state and results '''
        pass
    
    def _check_coupler_num_(self):
        assert len(self.node.parameters.coupler) == 1, "Currently we support 1 coupler in a single run only !"

    def _sort_variables_(self, order_in_list:list):
        for key_name in order_in_list:
            if key_name not in self.variables:
                raise KeyError(f"Key '{key_name}' missing when we tried to sort the variables !")
        return {k: self.variables[k] for k in order_in_list}


    def easy_preparation(self):
        """ Simplfied preparations. Variables, qubits/couplers and QUA program are included. """  
        
        self.participants_stand_by()
        self.exp_variable_arangement()
        self.qua_composer()

    def serialize_debug_file(self, file_name:str='debug'):
        ''' 
        Serialize your QUA program into py script.\n 
        args:
        - File_name: An arbitrary name **without File Extension** like "debug" as default.
        '''

        from qm import generate_qua_script
        file_name += '.py'

        if self.config is not None and self.qua_prog is not None:
            sourceFile = open(file_name, 'w')
            print(generate_qua_script(self.qua_prog, self.config), file=sourceFile) 
            sourceFile.close()

    def iteratively_run(self, target_iterations:int):
        '''
        Running iteratively, generated new dim = "iteration".\n
        If `target_iterations` = 1, the dim "iteration" will be dropped anyway.
        '''
        dss = []
        start = time()
        current_success = 0
        if target_iterations > 1:
            max_retries = target_iterations + 5
        else:
            max_retries = 1
            
        attempts = 0
        if target_iterations>4:
            self.progress_bar_display = False

        while current_success < target_iterations and attempts < max_retries:
            attempts += 1
            try:
                ds = self.qua_executor()

                dss.append(ds)
                current_success += 1
                print(f"Counts: {current_success} (Total attempts: {attempts})")
            
            except Exception as e:
                print(f"Attempt {attempts} failed: {e}. Close qm and skipping...")
                try:
                    if self.qm is not None:
                        self.qm.close()
                    sleep(2)
                except Exception as ee:
                    
                    print(f"Got errors when close qm:\n{ee}\n")
        
                if (attempts - current_success) > 5:
                    print("Too many consecutive failures. Stopping experiment.")
                    break

        
        end = time()
        print(f"Total {round(end-start,1)} sec for {target_iterations} counts")
        ds = concat(dss, dim='iteration')


        if target_iterations > 1:
            self.node.results = {"ds": ds}
        else:
            self.node.results = {"ds": ds.isel(iteration=0, drop=True)}
        

            
                