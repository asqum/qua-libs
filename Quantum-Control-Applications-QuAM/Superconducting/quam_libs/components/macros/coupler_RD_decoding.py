from quam_libs.components import Transmon, TransmonPair
from quam_libs.components import QuAM
from qualang_tools.units import unit
from typing import Dict, Literal
u = unit(coerce_to_integer=True)

class CouplerReadoutDecoder:
    """ For the future to scale up """
    def __init__(self, machine:QuAM, coupler_names:list, coupler_arbi_LO_manual:float=None, simulate:bool=False, load_data_id:int|None=None):
        if len(coupler_names) > 1:
            raise ValueError("Currently we only support you measure one coupler within a single run !")
        self.all_couplers = coupler_names
        self.machine = machine
        self.original_lo = {}

        # Step 1. Collecting all the dive and readout q for couplers
        self.rd_decoder()

        if not simulate and load_data_id is None:
            # Step 2. Chech aSWAP applied target
            self.aswap_supplier_instruciton()
            # Step 3. Readout Strategy checking
            self.readout_strategy_assigner()
            # Step 4. Driving LO settings
            self.lo_prepare(lo_GHz = coupler_arbi_LO_manual)


    def rd_decoder(self):
        """ Equip driving Q and readout Q for every coupler """
        self.paired_elements = {}
        for a_coupler in self.all_couplers:
            c: TransmonPair = self.machine.qubit_pairs[a_coupler]
            drive_q: Transmon = self.machine.qubits[c.extras["RD"]["driven_q"]]
            readout_q: Transmon = self.machine.qubits[c.extras["RD"]["readout_q"]]
        
            self.paired_elements[c.name] = {"coupler":c, "drive_q":drive_q, "readout_q":readout_q}


    def lo_prepare(self, lo_GHz:float=None):
        """
        Setting drive_q's LO for coupler's driving.\n
        Currently, All couplers with the same LO.
        
        """
        # Check key name first
        if lo_GHz is not None:
            if float(lo_GHz) > 8.0 or float(lo_GHz) < 1.0:
                raise ValueError(f"An inappropriate LO frequency = {lo_GHz} was given. We see it in the range (1 GHz, 8GHz) is reasonable !")
        
        for c_name in self.paired_elements:
            c: TransmonPair = self.paired_elements[c_name]['coupler']
            drive_q: Transmon = self.paired_elements[c_name]['drive_q']
            # memo original driving LO for the drive q
            self.original_lo[drive_q.name] = drive_q.xy.opx_output.upconverter_frequency
            # Set it to coupler's LO



            if lo_GHz is None:
                drive_q.xy.opx_output.upconverter_frequency = c.extras["RD"]["LO"]
            else:
                drive_q.xy.opx_output.upconverter_frequency = float(lo_GHz) * u.GHz


    def obtain_coupler_lo(self)->Dict:
        """ Get coupler's driving LO """
        LOs = {}
        for c_name in self.paired_elements:
            c: TransmonPair = self.paired_elements[c_name]['coupler']
            LOs[c_name] = c.extras["RD"]["LO"]

        return LOs
          

    def readout_strategy_assigner(self):
        """ Check readout strategy for every coupler. May support ZZ method. in the future. """
        for c_name in self.paired_elements:
            c: TransmonPair = self.paired_elements[c_name]['coupler']
            if 'strategy' in c.extras["RD"]:
                self.paired_elements[c_name]["readout_method"] = c.extras["RD"]["strategy"]
            else:
                self.paired_elements[c_name]["readout_method"] = 'aswap'

    
    def aswap_supplier_instruciton(self):
        """ Check aSWAP applied target and check direction of the aSWAP """
        for c_name in self.paired_elements:
            c: TransmonPair = self.paired_elements[c_name]['coupler']
            detector_q: Transmon = self.paired_elements[c_name]['readout_q']
            if 'aswap_supplier' in c.extras["RD"]:
                if c.extras["RD"]["aswap_supplier"].lower() == 'c':
                    print("*** aSWAP is applied on coupler itself !")
                    if not hasattr(c.coupler.operations, "aSWAP"):
                        raise  LookupError(f"aSWAP operation now is not in {c.name}.coupler.operation, please add it to unlock the ability for coupler's measurement!")
                    self.paired_elements[c_name]["aswap_supplier"] = c
                    c.coupler.operations['aSWAP'].slope_direction = c.extras["RD"]["swap_direction"]
                else:
                    self.paired_elements[c_name]["aswap_supplier"] = None
                    if 'swap_direction' in c.extras["RD"]:
                        detector_q.z.operations['aSWAP'].slope_direction = c.extras["RD"]["swap_direction"]

            else:
                self.paired_elements[c_name]["aswap_supplier"] = None
                if 'swap_direction' in c.extras["RD"]:
                        detector_q.z.operations['aSWAP'].slope_direction = c.extras["RD"]["swap_direction"]


    def get_obj_with_type(self, c_name, indicate_type:Literal["readout_q", "drive_q", "coupler"])->Transmon|TransmonPair:
        if indicate_type == 'coupler':
            ans: TransmonPair = self.paired_elements[c_name][indicate_type]
        else:
            ans: Transmon = self.paired_elements[c_name][indicate_type]

        return ans
    

    def get_all_TransmonPairs(self)->list[TransmonPair]:
        return [self.paired_elements[c_name]['coupler'] for c_name in self.paired_elements]
    
    def get_all_Transmons(self, catagory: Literal['readout_q', 'drive_q']='readout_q')->list[Transmon]:
        return [self.paired_elements[c_name][catagory] for c_name in self.paired_elements]
    


if __name__ == "__main__":
    machine = QuAM.load()
    couplers = ['coupler_q3_q4']
    CRD = CouplerReadoutDecoder(machine, couplers)


    x = CRD.get_obj_with_type(couplers[0], "coupler")

    print(CRD.paired_elements['coupler_q3_q4'].keys())
    print(CRD.obtain_coupler_lo())