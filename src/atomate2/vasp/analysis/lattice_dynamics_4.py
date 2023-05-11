import json
import os
import subprocess
import shlex
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from copy import copy, deepcopy

from monty.dev import requires
from monty.serialization import dumpfn, loadfn
from monty.json import jsanitize
from pymongo import ReturnDocument

import phonopy as phpy

from jobflow import job

from atomate2.utils.utils import env_chk, get_logger
# from atomate.vasp.database import VaspCalcDb
# from atomate.vasp.drones import VaspDrone
from atomate2.vasp.files import copy_non_vasp_outputs
from atomate2.vasp.analysis.lattice_dynamics_2 import (
    T_QHA,
    T_KLAT,
    fit_force_constants,
    harmonic_properties,
    anharmonic_properties,
    run_renormalization,
    # setup_TE_iter,
    get_cutoffs
)

from typing import Dict, List, Optional, Union
from fireworks import FiretaskBase, FWAction, explicit_serialize

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonon_band_structure_from_fc, \
    get_phonon_dos_from_fc, get_phonon_band_structure_symm_line_from_fc
from pymatgen.io.shengbte import Control
from pymatgen.transformations.standard_transformations import (
    SupercellTransformation,
)

try:
    import hiphive
    from hiphive import ForceConstants, ClusterSpace
    from hiphive.utilities import get_displacements
except ImportError:
    logger.info('Could not import hiphive!')
    hiphive = False


__author__ = "Alex Ganose, Junsoo Park"
__email__ = "aganose@lbl.gov, jsyony37@lbl.gov"

logger = get_logger(__name__)


@job
@explicit_serialize
# class RunHiPhive(FiretaskBase):
def RunHiPhive(
    cutoffs: Optional[list[list]] = None,
    separate_fit: bool = None,
    fit_method: str = None,
    disp_cut: float = None,
    bulk_modulus: float = None,
    temperature_qha: float = None,
    imaginary_tol: float = None,
    # rattled_structures: json = None,
    # forces: json = None,
    # structure_data: json = None,
    prev_dir_struct: str = None,
):    
    """
    Fit force constants using hiPhive.
    Requires "perturbed_structures.json", "perturbed_forces.json", and
    "structure_data.json" files to be present in the current working directory.
    Optional parameters:
        cutoffs (Optional[list[list]]): A list of cutoffs to trial. If None,
            a set of trial cutoffs will be generated based on the structure
            (default).
        separate_fit: If True, harmonic and anharmonic force constants are fit
            separately and sequentially, harmonic first then anharmonic. If
            False, then they are all fit in one go. Default is False.
        disp_cut: if separate_fit=True, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting  
        imaginary_tol (float): Tolerance used to decide if a phonon mode
            is imaginary, in THz.
        fit_method (str): Method used for fitting force constants. This can
            be any of the values allowed by the hiphive ``Optimizer`` class.
    """

    logger.info('Separate Fit is {}'.format(separate_fit))

    # os.chdir('/global/homes/h/hrushi99/atomate2_workflows/hiphive/Ba/555')

    # print(os.getcwd())
    # os.chdir('..')
    # print(os.getcwd())

    copy_non_vasp_outputs(prev_dir_struct)

    all_structures = loadfn("perturbed_structures.json")
    all_forces = loadfn("perturbed_forces_new.json")
    structure_data = loadfn("structure_data.json")

    # all_structures = rattled_structures
    # all_forces = forces
    # structure_data = structure_data

    print(f'structures: {all_structures}')
    print(f'all_forces: {all_forces}')
    print(f'structure_data: {structure_data}')


    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_matrix = np.array(structure_data["supercell_matrix"])


    separate_fit = separate_fit
    disp_cut = disp_cut

    if cutoffs is None:
        print("###################### No cutoffs found. Please provide a list of cutoffs.")
        cutoffs = get_cutoffs(supercell_structure)
    
    else:
        cutoffs = cutoffs
    # cutoffs = [[7, 4.5, 3.5], [7.75, 4.5, 4.4], [8.5, 5.625, 4.4]]



    # if cutoffs is None:
    #     print("###################### No cutoffs found. Please provide a list of cutoffs.")

    if temperature_qha is not None:
        T_qha = temperature_qha
    else: 
        T_qha = T_QHA
    T_qha.sort()
    imaginary_tol = imaginary_tol
    bulk_modulus = bulk_modulus
    fit_method = fit_method

    structures = []
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    for structure, forces in zip(all_structures, all_forces):
        atoms = AseAtomsAdaptor.get_atoms(structure)
        displacements = get_displacements(atoms, supercell_atoms)
        atoms.new_array("displacements", displacements)
        atoms.new_array("forces", forces)
        atoms.positions = supercell_atoms.get_positions()
        structures.append(atoms)
    
    print(f'Fit method is {fit_method}')
    print(f'disp_cut is {disp_cut}')
    print(f'separate_fit is {separate_fit}')
    print(f'cutoffs is {cutoffs}')

    all_cutoffs = cutoffs
    fcs, param, cs, fitting_data, fcp = fit_force_constants(
        parent_structure,
        supercell_matrix,
        structures,
        # cutoffs,
        all_cutoffs,
        separate_fit,
        disp_cut,
        imaginary_tol,
        fit_method,
    )

    if fcs is None:
        # fitting failed for some reason
        raise RuntimeError(
            "Could not find a force constant solution"
        )
    
    # fcs.write("best_fit_force_constants_{}_{}.fcp".format(cutoffs, fit_method))
    # # fcp.write("force_constants_{}_{}.fcs".format(cutoffs, fit_method))

    # np.savetxt("best_fit_rmse_test_{}_{}.csv".format(cutoffs, fit_method), 
    #     fitting_data["rmse_test"], delimiter = ", ", fmt ='% s')  


    thermal_data, phonopy = harmonic_properties(
        parent_structure, supercell_matrix, fcs, T_qha, imaginary_tol
    )

    # print("############# This should be bulk_modulus")
    # print(bulk_modulus)

    anharmonic_data, phonopy = anharmonic_properties(
        phonopy, fcs, T_qha, thermal_data["heat_capacity"],
        thermal_data["n_imaginary"], bulk_modulus
    )

    phonopy.save("phonopy_params.yaml")
    fitting_data["n_imaginary"] = thermal_data.pop("n_imaginary")
    thermal_data.update(anharmonic_data)
    dumpfn(fitting_data, "fitting_data.json")
    dumpfn(thermal_data, "thermal_data.json")

    logger.info("Writing cluster space and force_constants")
    logger.info("{}".format(type(fcs)))
    fcp.write("force_constants.fcp")
    fcs.write("force_constants.fcs")
    np.savetxt('parameters.txt',param)
    cs.write('cluster_space.cs')

    if fitting_data["n_imaginary"] == 0:
        logger.info("No imaginary modes! Writing ShengBTE files")
        atoms = AseAtomsAdaptor.get_atoms(parent_structure)
        fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD", atoms, order=3)
        fcs.write_to_phonopy("FORCE_CONSTANTS_2ND", format="text")
    else:
        logger.info("ShengBTE files not written due to imaginary modes.")
        logger.info("You may want to perform phonon renormalization.")


    current_dir = os.getcwd()    

    # return thermal_data, anharmonic_data, fitting_data, param
    return [thermal_data, anharmonic_data, fitting_data, param, current_dir]


@job
@explicit_serialize
# class RunHiPhiveRenorm(FiretaskBase):
def RunHiPhiveRenorm(
    # renorm_temp=temperature,
    # renorm_method=renorm_method,
    # nconfig=renorm_nconfig,
    # conv_thresh=renorm_conv_thresh,
    # max_iter=renorm_max_iter,
    # renorm_TE_iter=renorm_TE_iter,
    # bulk_modulus=bulk_modulus,
    renorm_temp,
    renorm_method,
    nconfig,
    conv_thresh,
    max_iter,
    renorm_TE_iter,
    bulk_modulus,
    prev_dir_hiphive,
    prev_dir_struct
):
    """
    Perform phonon renormalization to obtain temperature-dependent force constants
    using hiPhive. Requires "structure_data.json" to be present in the current working
    directory.
    Required parameters:
   
    Optional parameter:
        renorm_temp (List): list of temperatures to perform renormalization - defaults to T_RENORM
        renorm_with_te (bool): if True, perform outer iteration over thermally expanded volumes
        bulk_modulus (float): input bulk modulus - required for thermal expansion iterations
    """

    optional_params = ["renorm_method","renorm_temp","nconfig","conv_thresh","max_iter",
                       "renorm_TE_iter","bulk_modulus","imaginary_tol"]

    # os.chdir('/global/homes/h/hrushi99/atomate2_workflows/hiphive_trial/MgO/555')
    
    # print(os.getcwd())
    # # os.chdir('..')
    # os.chdir('/global/homes/h/hrushi99/atomate2_workflows/hiphive/BP/555/StaticCalc/configs_10/job_2023-05-11-06-05-54-150813-12737')
    # print(os.getcwd())

    # @requires(hiphive, "hiphive is required for lattice dynamics workflow")
    # def run_task(self, fw_spec):

    copy_non_vasp_outputs(prev_dir_hiphive)
    copy_non_vasp_outputs(prev_dir_struct)


    cs = ClusterSpace.read('cluster_space.cs')
    fcs = ForceConstants.read('force_constants.fcs')
    param = np.loadtxt('parameters.txt')
    fitting_data = loadfn("fitting_data.json")
    structure_data = loadfn("structure_data.json")
    phonopy_orig = phpy.load("phonopy_params.yaml")
    
    thermal_data = loadfn("thermal_data.json")
    thermal_data = thermal_data["heat_capacity"]

    
    cutoffs = fitting_data["cutoffs"]
    fit_method = fitting_data["fit_method"]
    n_imaginary = fitting_data["n_imaginary"]
    imaginary_tol = fitting_data["imaginary_tol"]

    parent_structure = structure_data["structure"]
    supercell_structure = structure_data["supercell_structure"]
    supercell_atoms = AseAtomsAdaptor.get_atoms(supercell_structure)
    supercell_matrix = np.array(structure_data["supercell_matrix"])

    # renorm_temp = np.array(self.get("renorm_temp"))
    # renorm_method = self.get("renorm_method")
    # nconfig = self.get("nconfig")
    # conv_thresh = self.get("conv_thresh")
    # max_iter = self.get("max_iter")
    # renorm_TE_iter = self.get("renorm_TE_iter")
    # bulk_modulus = self.get("bulk_modulus")

    renorm_temp = np.array(renorm_temp)
    renorm_temp.sort()
    renorm_method = renorm_method
    nconfig = nconfig
    conv_thresh = conv_thresh
    max_iter = max_iter
    renorm_TE_iter = renorm_TE_iter
    bulk_modulus = bulk_modulus

        # Renormalization with DFT lattice 

        # Parallel version
#        renorm_data = Parallel(n_jobs=len(renorm_temp), backend="multiprocessing")(delayed(run_renormalization)(
#            parent_structure, supercell_atoms, supercell_matrix, cs, fcs, param, T, nconfig, max_iter,
#            conv_thresh, renorm_method, fit_method, bulk_modulus, phonopy_orig) for t, T in enumerate(renorm_temp))

        # Serial version - likely better because it allows parallelization over A matrix construction,
        # which takes most of the time during renormalization, and leaves no process idling around when 
        # low temperature renormalizations finish early
    renorm_data = []
    for t, T in enumerate(renorm_temp):
        data_T = run_renormalization(thermal_data, n_imaginary, parent_structure, supercell_atoms, supercell_matrix, cs, fcs, param, T, nconfig, 
                            max_iter, conv_thresh, renorm_method, fit_method, bulk_modulus, phonopy_orig)
        renorm_data.append(data_T)

    # Additional renormalization with thermal expansion - optional - just single "iteration" for now
    if renorm_TE_iter:
        n_TE_iter = 1            
        for i in range(n_TE_iter):
            logger.info("Renormalizing with thermally expanded lattice - iteration {}".format(i))

            if i==0: # first iteration - pull only cases where phonon is all real and order by temperature
                T_real = []
                dLfrac_real = []
                param_real = []
                for result in renorm_data:
                    if result is None: # failed or incomplete 
                        continue 
                    elif result["n_imaginary"] < 0: # still imaginary
                        continue
                    else:
                        T_real = T_real + result["temperature"]
                        dLfrac_real = dLfrac_real + result["expansion_ratio"]
                        param_real = param_real + result["param"]
                if len(T_real)==0: 
                    logger.info("No cases with real phonons - cannot do thermal expansion for any temperature")
                    break
            else:
                for t,result in enumerate(renorm_data):
                    dLfrac_real[t] = result["expansion_ratio"]  
                    param_real[t] = result["param"]

            parent_structure_TE, cs_TE, fcs_TE = setup_TE_iter(cutoffs,parent_structure,T_real,dLfrac_real)
            param_TE = copy(param_real)
            prim_TE_atoms = AseAtomsAdaptor.get_atoms(parent_structure_TE)
            prim_phonopy_TE = PhonopyAtoms(symbols=prim_TE_atoms.get_chemical_symbols(), 
                                            scaled_positions=prim_TE_atoms.get_scaled_positions(), cell=prim_TE_atoms.cell)
            phonopy_TE = Phonopy(prim_phonopy_TE, supercell_matrix=scmat, primitive_matrix=None)

            # Parallel
#                renorm_data = Parallel(n_jobs=len(T_real), backend="multiprocessing")(delayed(run_renormalization)(
#                    parent_structure_TE[t], supercell_atoms_TE[t], supercell_matrix, cs_TE[t], fcs_TE[t], param_TE[t],
#                    T, nconfig, max_iter, conv_thresh, renorm_method, fit_method, bulk_modulus, phonopy_TE
#                    ) for t, T in enumerate(T_real)
#                )

            # Serial
            renorm_data_TE = []
            for t, T in enumerate(T_real):
                data_T = run_renormalization(parent_structure_TE[t], supercell_atoms_TE[t], supercell_matrix, 
                                                cs_TE[t], fcs_TE[t], param_TE[t], T, nconfig, max_iter, conv_thresh,
                                                renorm_method, fit_method, bulk_modulus, phonopy_TE)
                renorm_data_TE.append(data_T)
                
        if len(T_real) > 0:
            for t, result in enumerate(renorm_data_TE):
                temp_index = np.where(renorm_temp==T_real[t])[0][0]
                renorm_data[temp_index] = result

    # write results
    logger.info("Writing renormalized results")
    # thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
    # "gruneisen","thermal_expansion","expansion_ratio","free_energy_correction"]
    thermal_keys = ["temperature","free_energy","entropy","heat_capacity",
    "gruneisen","thermal_expansion"]
    renorm_thermal_data = {key: [] for key in thermal_keys}
    for t, result in enumerate(renorm_data):
        logger.info("DEBUG: ",result)
        T = result["temperature"]
        fcs = result["fcs"]
        fcs.write("force_constants_{}K.fcs".format(T))
        np.savetxt('parameters_{}K.txt'.format(T),result["param"])
        for key in thermal_keys:
            renorm_thermal_data[key].append(result[key])
        if result["n_imaginary"] > 0:
            logger.info("n_imaginary = {} @ temperature = {}".format(result["n_imaginary"], T))
            logger.warning('Imaginary modes exist for {} K!'.format(T))
            logger.warning('ShengBTE files not written')
            logger.warning('No renormalization with thermal expansion')
        else:
            logger.info("No imaginary modes! Writing ShengBTE files")
            fcs.write_to_phonopy("FORCE_CONSTANTS_2ND_{}K".format(T), format="text")
            atoms = AseAtomsAdaptor.get_atoms(parent_structure)
            fcs.write_to_shengBTE("FORCE_CONSTANTS_3RD_{}K".format(T), atoms, order=3)

    # renorm_thermal_data.pop("n_imaginary")                    
    # dumpfn(thermal_data, "renorm_thermal_data.json")
    dumpfn(renorm_thermal_data, "renorm_thermal_data.json")

    return renorm_thermal_data

        
# @explicit_serialize
# class ForceConstantsToDb(FiretaskBase):
#     """
#     Add force constants, phonon band structure and density of states
#     to the database.
#     Assumes you are in a directory with the force constants, fitting
#     data, and structure data written to files.
#     Required parameters:
#         db_file (str): Path to DB file for the database that contains the
#             perturbed structure calculations.
#     Optional parameters:
#         renormalized (bool): Whether FC resulted from original fitting (False)
#             or renormalization process (True) determines how data are stored. 
#             Default is False.
#         mesh_density (float): The density of the q-point mesh used to calculate
#             the phonon density of states. See the docstring for the ``mesh``
#             argument in Phonopy.init_mesh() for more details.
#         additional_fields (dict): Additional fields added to the document, such
#             as user-defined tags, name, ids, etc.
#     """

#     required_params = ["db_file"]
#     optional_params = ["renormalized","mesh_density", "additional_fields"]

#     @requires(hiphive, "hiphive is required for lattice dynamics workflow")
#     def run_task(self, fw_spec):

#         db_file = env_chk(self.get("db_file"), fw_spec)
#         mmdb = VaspCalcDb.from_db_file(db_file, admin=True)
#         renormalized = self.get("renormalized", False)
#         mesh_density = self.get("mesh_density", 100.0)

#         structure_data = loadfn("structure_data.json")
#         forces = loadfn("perturbed_forces.json")
#         structures = loadfn("perturbed_structures.json")
        
#         structure = structure_data["structure"]
#         supercell_structure = structure_data["supercell_structure"]
#         supercell_matrix = structure_data["supercell_matrix"]

#         if not renormalized:
            
#             fitting_data = loadfn("fitting_data.json")
#             thermal_data = loadfn("thermal_data.json")
#             fcs = ForceConstants.read("force_constants.fcs")
            
#             dos_fsid, uniform_bs_fsid, lm_bs_fsid, fc_fsid = _get_fc_fsid(
#                 structure, supercell_matrix, fcs, mesh_density, mmdb
#                 )
        
#             data = {
#                 "created_at": datetime.utcnow(),            
#                 "tags": fw_spec.get("tags", None),
#                 "formula_pretty": structure.composition.reduced_formula,            
#                 "structure": structure.as_dict(),
#                 "supercell_matrix": supercell_matrix,
#                 "supercell_structure": supercell_structure.as_dict(),
#                 "perturbed_structures": [s.as_dict() for s in structures],
#                 "perturbed_forces": [f.tolist() for f in forces],
#                 "fitting_data": fitting_data,
#                 "thermal_data": thermal_data,
#                 "force_constants_fs_id": fc_fsid,
#                 "phonon_dos_fs_id": dos_fsid,
#                 "phonon_bandstructure_uniform_fs_id": uniform_bs_fsid,
#                 "phonon_bandstructure_line_fs_id": lm_bs_fsid,
#                 }
#             data.update(self.get("additional_fields", {}))

#             # Get an id for the force constants
#             fitting_id = _get_fc_fitting_id(mmdb)
#             metadata = {"fc_fitting_id": fitting_id, "renormalization_dir": os.getcwd()}
#             data.update(metadata)
#             data = jsanitize(data,strict=True,allow_bson=True)
            
#             mmdb.db.lattice_dynamics.insert_one(data)
            
#             logger.info("Finished inserting force constants and phonon data")

#         else:
#             renorm_thermal_data = loadfn("renorm_thermal_data.json")
#             temperature = renorm_thermal_data["temperature"]

#             # pushing data for individual temperature  
#             for t, T in enumerate(temperature):
#                 fcs = ForceConstants.read("force_constants_{}K.fcs".format(T))
#                 phonopy_fc = fcs.get_fc_array(order=2)
#                 dos_fsid, uniform_bs_fsid, lm_bs_fsid, fc_fsid = _get_fc_fsid(
#                     structure, supercell_matrix, phonopy_fc, mesh_density, mmdb
#                     )
                
#                 data_at_T = {
#                     "created_at": datetime.utcnow(),
#                     "tags": fw_spec.get("tags", None),
#                     "formula_pretty": structure.composition.reduced_formula,
#                     "renormalized": renormalized,
#                     "temperature": T,
#                     "force_constants_fs_id": fc_fsid,
#                     "thermal_data": renorm_thermal_data[t],
#                     "phonon_dos_fs_id": dos_fsid,
#                     "phonon_bandstructure_uniform_fs_id": uniform_bs_fsid,
#                     "phonon_bandstructure_line_fs_id": lm_bs_fsid,
#                     }
#                 data_at_T.update(self.get("additional_fields", {}))
        
#                 # Get an id for the force constants
#                 fitting_id = _get_fc_fitting_id(mmdb)
#                 metadata = {"fc_fitting_id": fitting_id, "fc_fitting_dir": os.getcwd()}
#                 data_at_T.update(metadata)
#                 data_at_T = jsanitize(data,strict=True,allow_bson=True)

#                 mmdb.db.renormalized_lattice_dynamics.insert_one(data_at_T)

#                 logger.info("Finished inserting renormalized force constants and phonon data at {} K".format(T))

#         return FWAction(update_spec=metadata)        


@job    
@explicit_serialize
# class RunShengBTE(FiretaskBase):
def RunShengBTE(
    shengbte_cmd,
    renormalized,
    temperature,
    structure_data
    # control_kwargs
):
    """
    Run ShengBTE to calculate lattice thermal conductivity. Presumes
    the FORCE_CONSTANTS_3RD and FORCE_CONSTANTS_2ND, and a "structure_data.json"
    file, with the keys "structure", " and "supercell_matrix" is in the current
    directory.
    Required parameters:
        shengbte_cmd (str): The name of the shengbte executable to run. Supports
            env_chk.
    Optional parameters:
        renormalized: boolean to denote whether force constants are from
            phonon renormalization (True) or directly from fitting (False)  
        temperature (float or dict): The temperature to calculate the lattice
            thermal conductivity for. Can be given as a single float, or a
            dictionary with the keys "t_min", "t_max", "t_step".
        control_kwargs (dict): Options to be included in the ShengBTE control
            file.
    """

    required_params = ["shengbte_cmd"]
    optional_params = ["renormalized","temperature", "control_kwargs"]


    print(os.getcwd())
    os.chdir('..')
    print(os.getcwd())

    ## Create a symlink to ShengBTE
    
    ShengBTE = "ShengBTE"
    src = "/global/homes/h/hrushi99/code/shengbte_new/shengbte/ShengBTE"
    dst = os.path.join(os.getcwd(), ShengBTE)

    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass

    # def run_task(self, fw_spec):
    # structure_data = loadfn("structure_data.json")
    structure_data = structure_data
    structure = structure_data["structure"]
    supercell_matrix = structure_data["supercell_matrix"]
    # temperature = self.get("temperature", T_KLAT)
    # renormalized = self.get("renormalized", False)
    # temperature = self.get("temperature", T_KLAT)
    if temperature is not None:
        temperature = temperature
    else: 
        temperature = T_KLAT
    # renormalized = self.get("renormalized", False)
    if renormalized is not None:
        renormalized = renormalized
    else: 
        renormalized = False

    if renormalized:
        assert isinstance(temperature, (int, float))
        # self["t"] = temperature
        t = temperature
    else:
        if isinstance(temperature, (int, float)):
            # self["t"] = temperature
            t = temperature
        elif isinstance(temperature, dict):
            # self["t_min"] = temperature["t_min"]
            # self["t_max"] = temperature["t_max"]
            # self["t_step"] = temperature["t_step"]
            t_min = temperature["t_min"]
            t_max = temperature["t_max"]
            t_step = temperature["t_step"]
        else:
            raise ValueError("Unsupported temperature type, must be float or dict")
    
    control_dict = {
        # "scalebroad": 0.5,
        "scalebroad": 1.1,
        "nonanalytic": False,
        "isotopes": False,
        "temperature": temperature,
        "scell": np.diag(supercell_matrix).tolist(),
    }
    # control_kwargs = control_kwargs or {}
    # control_dict.update(control_kwargs)
    control = Control().from_structure(structure, **control_dict)
    control.to_file()

    # shengbte_cmd = env_chk(self["shengbte_cmd"], fw_spec)
    # shengbte_cmd = env_chk(shengbte_cmd, fw_spec)

    if isinstance(shengbte_cmd, str):
        shengbte_cmd = os.path.expandvars(shengbte_cmd)
        shengbte_cmd = shlex.split(shengbte_cmd)

    shengbte_cmd = list(shengbte_cmd)
    logger.info("Running command: {}".format(shengbte_cmd))

    with open("shengbte.out", "w") as f_std, open(
        "shengbte_err.txt", "w", buffering=1
    ) as f_err:
        # use line buffering for stderr
        return_code = subprocess.call(
            shengbte_cmd, stdout=f_std, stderr=f_err
        )
    logger.info(
        "Command {} finished running with returncode: {}".format(
            shengbte_cmd, return_code
        )
    )

    if return_code == 1:
        raise RuntimeError(
            "Running ShengBTE failed. Check '{}/shengbte_err.txt' for "
            "details.".format(os.getcwd())
        )
    


# @explicit_serialize
# class ShengBTEToDb(FiretaskBase):
#     """
#     Add lattice thermal conductivity results to database.
#     Assumes you are in a directory with the ShengBTE results in.
#     Required parameters:
#         db_file (str): Path to DB file for the database that contains the
#             perturbed structure calculations.
#     Optional parameters:
#         additional_fields (dict): Additional fields added to the document.
#     """

#     required_params = ["db_file"]
#     optional_params = ["additional_fields"]

#     def run_task(self, fw_spec):
#         db_file = env_chk(self.get("db_file"), fw_spec)
#         mmdb = VaspCalcDb.from_db_file(db_file, admin=True)

#         control = Control().from_file("CONTROL")
#         structure = control.get_structure()
#         supercell_matrix = np.diag(control["scell"])

#         if Path("BTE.KappaTotalTensorVsT_CONV").exists():
#             filename = "BTE.KappaTotalTensorVsT_CONV"
#         elif Path("BTE.KappaTensorVsT_CONV").exists():
#             filename = "BTE.KappaTensorVsT_CONV"
#         elif Path("BTE.KappaTensorVsT_RTA").exists():
#             filename = "BTE.KappaTensorVsT_RTA"
#         else:
#             raise RuntimeError("Could not find ShengBTE output files.")

#         bte_data = np.loadtxt(filename)
#         if len(bte_data.shape) == 1:
#             # pad extra axis to make compatible with multiple temperatures
#             bte_data = bte_data[None, :]

#         temperatures = bte_data[:, 0].tolist()
#         kappa = bte_data[:, 1:10].reshape(-1, 3, 3).tolist()

#         data = {
#             "structure": structure.as_dict(),
#             "supercell_matrix": supercell_matrix.tolist(),
#             "temperatures": temperatures,
#             "lattice_thermal_conductivity": kappa,
#             "control": control.as_dict(),
#             "tags": fw_spec.get("tags", None),
#             "formula_pretty": structure.composition.reduced_formula,
#             "shengbte_dir": os.getcwd(),
#             "fc_fitting_id": fw_spec.get("fc_fitting_id", None),
#             "fc_fitting_dir": fw_spec.get("fc_fitting_dir", None),
#             "renormalization_dir": fw_spec.get("renormalization_dir", None),
#             "created_at": datetime.utcnow(),
#         }
#         data.update(self.get("additional_fields", {}))

#         mmdb.collection = mmdb.db["lattice_thermal_conductivity"]
#         mmdb.collection.insert(data)


# def _get_fc_fitting_id(mmdb: VaspCalcDb) -> int:
#     """Helper method to get a force constant fitting id."""
#     fc_id = mmdb.db.counter.find_one_and_update(
#         {"_id": "fc_fitting_id"},
#         {"$inc": {"c": 1}},
#         return_document=ReturnDocument.AFTER,
#     )
#     if fc_id is None:
#         mmdb.db.counter.insert({"_id": "fc_fitting_id", "c": 1})
#         fc_id = 1
#     else:
#         fc_id = fc_id["c"]

#     return fc_id


# def _get_fc_fsid(structure, supercell_matrix, fcs, mesh_density, mmdb):
#     phonopy_fc = fcs.get_fc_array(order=2)
    
#     logger.info("Getting uniform phonon band structure.")
#     uniform_bs = get_phonon_band_structure_from_fc(
#         structure, supercell_matrix, phonopy_fc
#     )
    
#     logger.info("Getting line mode phonon band structure.")
#     lm_bs = get_phonon_band_structure_symm_line_from_fc(
#         structure, supercell_matrix, phonopy_fc
#     )
    
#     logger.info("Getting phonon density of states.")
#     dos = get_phonon_dos_from_fc(
#         structure, supercell_matrix, phonopy_fc, mesh_density=mesh_density
#     )
    
#     logger.info("Inserting phonon objects into database.")
#     dos_fsid, _ = mmdb.insert_gridfs(
#         dos.to_json(), collection="phonon_dos_fs"
#     )
#     uniform_bs_fsid, _ = mmdb.insert_gridfs(
#         uniform_bs.to_json(), collection="phonon_bandstructure_fs"
#     )
#     lm_bs_fsid, _ = mmdb.insert_gridfs(
#         lm_bs.to_json(), collection="phonon_bandstructure_fs"
#     )
    
#     logger.info("Inserting force constants into database.")
#     fc_json = json.dumps(
#         {str(k): v.tolist() for k, v in fcs.get_fc_dict().items()}
#     )
#     fc_fsid, _ = mmdb.insert_gridfs(
#         fc_json, collection="phonon_force_constants_fs"
#     )

#     return dos_fsid, uniform_bs_fsid, lm_bs_fsid, fc_fsid