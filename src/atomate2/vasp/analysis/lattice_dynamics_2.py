from itertools import product
from joblib import Parallel, delayed
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy as sp
import os
import psutil
from copy import copy
import time
import csv

from atomate2.utils.utils import get_logger

from hiphive import (ForceConstants, ForceConstantPotential,
                     enforce_rotational_sum_rules, ClusterSpace,
                     StructureContainer)

from hiphive.force_constant_model import ForceConstantModel
from hiphive.cutoffs import is_cutoff_allowed, estimate_maximum_cutoff
from hiphive.fitting import Optimizer
# from trainstation import Optimizer
from hiphive.renormalization import Renormalization
from hiphive.utilities import get_displacements
from hiphive.run_tools import _clean_data, FE_correction, construct_fit_data

from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.phonopy import get_phonopy_structure


from ase.atoms import Atoms
from ase.cell import Cell

from phonopy import Phonopy
from phono3py.phonon3.gruneisen import Gruneisen


__author__ = "Alex Ganose, Rees Chang, Junsoo Park"
__email__ = "aganose@lbl.gov, rc564@cornell.edu, jsyony37@lbl.gov"

logger = get_logger(__name__)

# Define some shared constants
IMAGINARY_TOL = 0.025  # in THz
MESH_DENSITY = 100.0  # should always be a float 

# Temperature for straight-up phonopy calculation of thermodynamic properties (free energy etc.)
T_QHA = [i*100 for i in range(21)]
# Temperature at which renormalization is to be performed
# T_RENORM = [0,50,100,200,300,500,700,1000,1500]#[i*100 for i in range(0,16)]
T_RENORM = [0,300, 700, 1000, 1500]#[i*100 for i in range(0,16)]
# T_RENORM = [0,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[i*100 for i in range(0,16)]
# Temperature at which lattice thermal conductivity is calculated
# If renormalization is performed, T_RENORM overrides T_KLAT for lattice thermal conductivity
T_KLAT = {"t_min":100,"t_max":1500,"t_step":100} #[i*100 for i in range(0,11)]
# T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#[i*100 for i in range(0,16)]
T_THERMAL_CONDUCTIVITY = [0, 100, 200, 300]#[i*100 for i in range(0,16)]

FIT_METHOD = "rfe" 
RENORM_METHOD = 'pseudoinverse'
RENORM_NCONFIG = 5 # Changed from 50
RENORM_CONV_THRESH = 0.1 # meV/atom
RENORM_MAX_ITER = 30 # Changed from 20
DISP_CUT = 0.02
SEPERATE_FIT = True

eV2J = sp.constants.elementary_charge
hbar = sp.constants.hbar # J-s
kB = sp.constants.Boltzmann # J/K

def get_cutoffs(supercell_structure: Structure):
    """
    Get a list of trial cutoffs based on a supercell structure for grid search.
    An initial guess for the lower bound of the cutoffs is made based on the
    average period (row) of the elements in the structure, according to:
    ====== === === ===
    .        Cutoff
    ------ -----------
    Period 2ND 3RD 4TH
    ====== === === ===
     1     5.0 3.0 2.5
     2     6.0 3.5 3.0
     3     7.0 4.5 3.5
     4     8.0 5.5 4.0
     5     9.0 6.0 4.5
     6     10.0 6.5 5.0
     7     11.0 7.0 5.5
    ====== === === ===
    The maximum cutoff for each order is determined by the minimum cutoff and
    the following table. A full grid of all possible cutoff combinations is
    generated based on the step size in the table below times a row factor
    ====== ==== =====
    Cutoff Max  Step
    ====== ==== =====
    2ND    +2.0 1.0
    3RD    +1.5 0.75
    4TH    +0.6 0.6
    ====== ==== =====
    Finally, the max cutoff size is determined by the supercell lattice dimensions.
    Cutoffs which result in multiple of the same orbits being populated will be
    discounted.
    Args:
        supercell_structure: A structure.
    Returns:
        A list of trial cutoffs.
    """
    # indexed as min_cutoffs[order][period]
    # DO NOT CHANGE unless you know what you are doing
    min_cutoffs = {
        2: {1: 5.0, 2: 6.0, 3: 7.0, 4: 8.0, 5: 9.0, 6: 10.0, 7: 11.0},
        3: {1: 3.0, 2: 3.5, 3: 4.5, 4: 5.5, 5: 6.0, 6: 6.5, 7: 7.0},
        4: {1: 2.5, 2: 3.0, 3: 3.5, 4: 4.0, 5: 4.5, 6: 5.0, 7: 5.5},
    }
    inc = {2: 2, 3: 1.5, 4: 0.6}
    steps = {2: 1, 3: 0.75, 4: 0.6}

    row = int(np.around(np.array([s.row for s in supercell_structure.species]).mean(),0))
    factor = row/4
    mins = {
        2: min_cutoffs[2][row], 3: min_cutoffs[3][row], 4: min_cutoffs[4][row]
    }

    range_two = np.arange(mins[2], mins[2] + factor*(inc[2]+steps[2]), factor*steps[2])
    range_three = np.arange(mins[3], mins[3] + factor*(inc[3]+steps[3]), factor*steps[3])
    range_four = np.arange(mins[4], mins[4] + factor*(inc[4]+steps[4]), factor*steps[4])

    cutoffs = np.array(list(map(list, product(range_two, range_three, range_four))))
    max_cutoff = estimate_maximum_cutoff(AseAtomsAdaptor.get_atoms(supercell_structure))
    cutoffs[cutoffs>max_cutoff] = max_cutoff-0.2
    logger.info('CUTOFFS \n {}'.format(cutoffs))
    logger.info('MAX_CUTOFF \n {}'.format(max_cutoff))    
    good_cutoffs = np.all(cutoffs < max_cutoff-0.1, axis=1)
    logger.info('GOOD CUTOFFS \n{}'.format(good_cutoffs))
    return cutoffs[good_cutoffs].tolist()


def fit_force_constants(
    parent_structure: Structure,
    supercell_matrix: np.ndarray,
    structures: List["Atoms"],
    all_cutoffs: List[List[float]],
    separate_fit: bool,
    disp_cut: float,
    imaginary_tol: float,
    fit_method: str,
    n_jobs: int = -1,
    fit_kwargs: Optional[Dict] = None
) -> Tuple["SortedForceConstants", np.ndarray, ClusterSpace, Dict]:
    """
    Fit force constants using hiphive and select the optimum cutoff values.
    The optimum cutoffs will be determined according to:
    1. Number imaginary modes < ``max_n_imaginary``.
    2. Most negative imaginary frequency < ``max_imaginary_freq``.
    3. Least number of imaginary modes.
    4. Lowest free energy at 300 K.
    If criteria 1 and 2 are not satisfied, None will be returned as the
    force constants.
    Args:
        parent_structure: Initial input structure.
        supercell_matrix: Supercell transformation matrix.
        structures: A list of ase atoms objects with "forces" and
            "displacements" arrays added, as required by hiPhive.
        all_cutoffs: A nested list of cutoff values to trial. Each set of
            cutoffs contains the radii for different orders starting with second
            order.
        separate_fit: Boolean to determine whether harmonic and anharmonic fitting
            are to be done separately (True) or in one shot (False)
        disp_cut: if separate_fit true, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.
        max_n_imaginary: Maximum number of imaginary modes allowed in the
            the final fitted force constant solution. If this criteria is not
            reached by any cutoff combination this FireTask will fizzle.
        max_imaginary_freq: Maximum allowed imaginary frequency in the
            final fitted force constant solution. If this criteria is not
            reached by any cutoff combination this FireTask will fizzle.
        fit_method: Method used for fitting force constants. This can be
            any of the values allowed by the hiphive ``Optimizer`` class.
        n_jobs: Number of processors to use for fitting coefficients. -1 means use all
            processors.
        fit_kwargs: Additional arguements passed to the hiphive force constant
            optimizer.
    Returns:
        A tuple of the best fitted force constants as a hiphive
        ``SortedForceConstants`` object, array of parameters, cluster space,
        and a dictionary of information on the fitting results.
    """
    logger.info("Starting force constant fitting.")

    fitting_data = {
        "cutoffs": [],
        "rmse_test": [],
    #    "n_imaginary": [],
#        "min_frequency": [],
#        "temperature": [],
#        "free_energy": [],
#        "entropy": [],
#        "heat_capacity": [],
        "fit_method": fit_method,
        "separate_fit": separate_fit,
        "disp_cut": disp_cut,
        "imaginary_tol": imaginary_tol,
#        "max_n_imaginary": max_n_imaginary,
    }

    best_fit = {
        "n_imaginary": np.inf,
        "rmse_test": np.inf,
        "cluster_space": None,
        "force_constants": None,
        "parameters":None,
        "cutoffs": None,
    }
    n_cutoffs = len(all_cutoffs)

    fit_kwargs = fit_kwargs if fit_kwargs else {}
    if fit_method == "rfe" and n_jobs == -1:
        fit_kwargs["n_jobs"] = 1
    # elif fit_method == "lasso":
    #     fit_kwargs['lasso'] = dict(max_iter=1000)  
    elif fit_method == "elasticnet":
    # if fit_method in ['elasticnet', 'lasso']:
        fit_kwargs = {"max_iter": 100000}
        # fit_kwargs = {"max_iter": 10000}
        # fit_kwargs = {"tol": 0.001, "max_iter": 200}

    logger.info('Separate Fit is {}'.format(separate_fit))
    logger.info('CPU COUNT: {}'.format(os.cpu_count()))
    # cutoff_results = Parallel(n_jobs=12, backend="multiprocessing")(delayed(_run_cutoffs)(
    #     i, cutoffs, n_cutoffs, parent_structure, structures, supercell_matrix, fit_method,
    #     separate_fit, disp_cut, imaginary_tol, fit_kwargs) for i, cutoffs in enumerate(all_cutoffs))
    
    for i, cutoffs in enumerate(all_cutoffs):
        
        print(disp_cut)
        print("THIS IS DISP_____CUT")

        print(fit_kwargs)
        print("THESE ARE FIT_KWARGS_______")
        start_time = time.time()
        cutoff_results = _run_cutoffs(
            i, cutoffs, n_cutoffs, parent_structure, structures, supercell_matrix, fit_method,
            separate_fit, disp_cut, imaginary_tol, fit_kwargs)
        time_taken = time.time() - start_time
        logger.info('Time taken for cutoffs {} is {} seconds'.format(cutoffs, time_taken))

        # with open('timings_{}_{}.csv'.format(cutoffs, fit_method), mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([cutoffs, time_taken, cutoff_results["rmse_test"]])

        print(f'cuttoffs: {cutoffs}')
        print(f'time taken: {time_taken}')
        print(f'rmse_test: {cutoff_results["rmse_test"]}')
        print(f'fit_method: {fit_method}')
        with open('timings_{}_{}.csv'.format(cutoffs, fit_method), mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([cutoffs, time_taken, cutoff_results["rmse_test"]])



        cutoff_results["force_constants_potential"].write("force_constants_potential_{}_{}.fcp".format(cutoffs, fit_method))
        cutoff_results["force_constants"].write("force_constants_potential_{}_{}.fcs".format(cutoffs, fit_method))   


        print("###################This should be the RMSE")
        print(fit_method)
        print(cutoff_results["cutoffs"])
        print(cutoff_results["rmse_test"])
        print(cutoff_results["parameters"])
        # print(rmse_test_results)
        # print(parameters_results)
        if (cutoff_results["rmse_test"] < best_fit["rmse_test"]):
            best_fit["rmse_test"] = cutoff_results["rmse_test"]
            best_fit["cluster_space"] = cutoff_results["cluster_space"]
            best_fit["force_constants"] = cutoff_results["force_constants"]
            best_fit["parameters"] = cutoff_results["parameters"]
            best_fit["cutoffs"] = cutoff_results["cutoffs"]
            best_fit["force_constants_potential"] = cutoff_results["force_constants_potential"]
            # fitting_data["best"] = result["cutoffs"]

    logger.info('CUTOFF RESULTS \n {}'.format(cutoff_results))
    
    for result in cutoff_results:
        if result is None:
            continue

        fitting_data["cutoffs"].append(cutoff_results["cutoffs"])
        fitting_data["rmse_test"].append(cutoff_results["rmse_test"])
        # fitting_data["n_imaginary"].append(result["n_imaginary"])
        # fitting_data["min_frequency"].append(result["min_frequency"])

#         if (
#             result["rmse_test"] < best_fit["rmse_test"]
# #            and result["min_frequency"] > -np.abs(max_imaginary_freq)
# #            and result["n_imaginary"] <= max_n_imaginary
# #            and result["n_imaginary"] < best_fit["n_imaginary"]
#         ):
#             best_fit.update(result)
#             fitting_data["best"] = result["cutoffs"]

    logger.info("Finished fitting force constants.")

    # np.savetxt("rmse_test_{}_{}.csv".format(all_cutoffs, fit_method), 
    #     fitting_data["rmse_test"], delimiter = ", ", fmt ='% s')  

    return best_fit["force_constants"], best_fit["parameters"], best_fit["cluster_space"], fitting_data, best_fit["force_constants_potential"]
    # return cutoff_results["force_constants"], cutoff_results["parameters"], cutoff_results["cluster_space"], fitting_data, cutoff_results["force_constants_potential"]


def get_fit_data(
        cs: ClusterSpace,
        supercell: Atoms,
        structures: List["Atoms"],
        separate_fit: bool,
        disp_cut: float,
        ncut: int,
        param2: np.ndarray = None
) -> "List":
    """ Constructs structure container of atom-displaced supercells
    Args: 
    cs: ClusterSpace
    supercell: Atoms
      Original undisplaced supercell
    structures: A list of ase atoms objects with the "forces" and
      "displacements" arrays included.
    separate_fit: Boolean to determine whether harmonic and anharmonic fitting
      are to be done separately (True) or in one shot (False)
    disp_cut: if separate_fit true, determines the mean displacement of perturbed
      structure to be included in harmonic (<) or anharmonic (>) fitting
    ncut: the parameter index where fitting separation occurs
      param2: previously fit parameter array (harmonic only for now, hence 2)
    Returns:
    fit_data: List[A_mat,f_vec] 
    """

    saved_structures = []
    fcm = ForceConstantModel(supercell,cs)
    natom = supercell.get_global_number_of_atoms()
    ndim = natom*3
    nrow_all = ndim*len(saved_structures)
    A_mat = np.zeros((nrow_all,cs.n_dofs))
    f_vec = np.zeros(nrow_all)

    for i, structure in enumerate(structures):
        displacements = structure.get_array('displacements')
        mean_displacements = np.linalg.norm(displacements,axis=1).mean()
        logger.info('Mean displacements: {}'.format(mean_displacements))
        if not separate_fit: # fit all
            saved_structures.append(structure)
        else: # fit separately
            if param2 is None: # for harmonic fitting
                if mean_displacements < disp_cut:
                    saved_structures.append(structure)
            else: # for anharmonic fitting
                if mean_displacements >= disp_cut:
                    saved_structures.append(structure)

    fit_data_tmp = Parallel(n_jobs=min(os.cpu_count(),max(1,len(saved_structures))))(#,prefer="threads")(
        delayed(construct_fit_data)(fcm,structure,s)
        for s,structure in enumerate(saved_structures)
    )
    for s,data in enumerate(fit_data_tmp):
        logger.info('DEBUG: {}, {}'.format(s,data[0]))
        A_mat[s*ndim:(s+1)*ndim] = data[0]
        f_vec[s*ndim:(s+1)*ndim] = data[1]

    if param2 is not None:
        f_vec -= np.dot(A_mat[:,:ncut],param2) # subtract harmonic force from total
    logger.info('Fit_data dimensions: {}'.format(A_mat.shape))
    fit_data = _clean_data(A_mat,f_vec,ndim)
    return fit_data


def _run_cutoffs(
    i,
    cutoffs,
    n_cutoffs,
    parent_structure,
    structures,
    supercell_matrix,
    fit_method,
    separate_fit,
    disp_cut,
    imaginary_tol,
    fit_kwargs
) -> Dict:

    # from collections import defaultdict
    # fit_kwargs = defaultdict(dict)
    # fit_kwargs['lasso'] = dict(max_iter=1000)   


    logger.info(
        "Testing cutoffs {} out of {}: {}".format(i+1, n_cutoffs, cutoffs)
    )
    supercell_atoms = structures[0]

    if not is_cutoff_allowed(supercell_atoms, max(cutoffs)):
        logger.info(
            "Skipping cutoff due as it is not commensurate with supercell size"
        )
        return None

    cs = ClusterSpace(supercell_atoms, cutoffs, symprec=1e-3, acoustic_sum_rules=True)
    logger.debug(cs.__repr__())
    n2nd = cs.get_n_dofs_by_order(2)
    nall = cs.n_dofs
    
    if separate_fit:
        logger.info('Fitting harmonic force constants separately')
#        fit_data = get_fit_data(cs, supercell_atoms, structures, separate_fit,
#                                disp_cut, ncut=n2nd, param2=None)
        sc = get_structure_container(cs, structures, separate_fit, disp_cut,
                                     ncut=n2nd, param2=None)
    
        opt = Optimizer(sc.get_fit_data(),
                        fit_method,
                        [0,n2nd],
                        **fit_kwargs)
        opt.train()
        param_harmonic = opt.parameters # harmonic force constant parameters
        
        logger.info('Fitting anharmonic force constants separately')
#        fit_data = get_fit_data(cs, supercell_atoms, structures, separate_fit,
#                                disp_cut, ncut=n2nd, param2=param_harmonic)
        sc = get_structure_container(cs, structures, separate_fit, disp_cut,
                                     ncut=n2nd, param2=param_harmonic)
        opt = Optimizer(sc.get_fit_data(),
                        fit_method,
                        [n2nd,nall],
                        **fit_kwargs)
        opt.train()
        param_anharmonic = opt.parameters # anharmonic force constant parameters

        parameters = np.concatenate((param_harmonic,param_anharmonic)) # combine 
        assert(nall==len(parameters))
        logger.info('Training complete for cutoff: {}, {}'.format(i,cutoffs))
        
    else:
        logger.info('Fitting all force constants in one shot')
#        fit_data = get_fit_data(cs, supercell_atoms, structures, separate_fit,
#                                disp_cut=None, ncut=None, param2=None)
        sc = get_structure_container(cs, structures, separate_fit, disp_cut=None,
                                     ncut=None, param2=None)
        opt = Optimizer(sc.get_fit_data(),
                        fit_method,
                        [0,nall],
                        **fit_kwargs)
        opt.train()
        parameters = opt.parameters
        logger.info('Training complete for cutoff: {}, {}'.format(i,cutoffs))

    logger.info('Memory use: {} %'.format(psutil.virtual_memory().percent))
    parameters = enforce_rotational_sum_rules(
        cs, parameters, ["Huang", "Born-Huang"]
    )
    fcp = ForceConstantPotential(cs, parameters)
    fcs = fcp.get_force_constants(supercell_atoms)
    logger.info('FCS generated for cutoff {}, {}'.format(i,cutoffs))

    
    try:
        return {
            "cutoffs": cutoffs,
            "rmse_test": opt.rmse_test,
            "cluster_space": sc.cluster_space,
            "parameters": parameters,
            "force_constants": fcs,
            "force_constants_potential": fcp
        }
    except Exception:
        return None


def get_structure_container(
        cs: ClusterSpace,
        structures: List["Atoms"],
        separate_fit: bool,
        disp_cut: float,
        ncut: int,
        param2: np.ndarray
) -> "StructureContainer":
    """
    Get a hiPhive StructureContainer from cutoffs and a list of atoms objects.
    Args:
        cs: ClusterSpace 
        structures: A list of ase atoms objects with the "forces" and
            "displacements" arrays included.
        separate_fit: Boolean to determine whether harmonic and anharmonic fitting
            are to be done separately (True) or in one shot (False)
        disp_cut: if separate_fit true, determines the mean displacement of perturbed
            structure to be included in harmonic (<) or anharmonic (>) fitting 
        ncut: the parameter index where fitting separation occurs
        param2: previously fit parameter array (harmonic only for now, hence 2)
    Returns:
        A hiPhive StructureContainer.
    """

    sc = StructureContainer(cs)
    saved_structures = []
    for i, structure in enumerate(structures):
        displacements = structure.get_array('displacements')
        mean_displacements = np.linalg.norm(displacements,axis=1).mean()
        logger.info('Mean displacements: {}'.format(mean_displacements))
        if not separate_fit: # fit all
            sc.add_structure(structure)
        else: # fit separately
            if param2 is None: # for harmonic fitting
                if mean_displacements < disp_cut:
                    print("We are in harmonic fitting if statement")
                    print(mean_displacements)
                    
                    sc.add_structure(structure) 
            else: # for anharmonic fitting
                if mean_displacements >= disp_cut:

                    print("We are in anharmonic fitting if statement")
                    print(mean_displacements)
                    sc.add_structure(structure) 
                    saved_structures.append(structure) 
    
    print("We have completed anharmonic fitting")
    print(sc.get_fit_data())

    if separate_fit and param2 is not None: # do after anharmonic fitting
        A_mat = sc.get_fit_data()[0] # displacement matrix
        f_vec = sc.get_fit_data()[1] # force vector
        f_vec -= np.dot(A_mat[:,:ncut],param2) # subtract harmonic forces
        sc.delete_all_structures()
        for i, structure in enumerate(saved_structures):
            natoms = structure.get_global_number_of_atoms()
            ndisp = natoms*3
            structure.set_array('forces',f_vec[i*ndisp:(i+1)*ndisp].reshape(natoms,3))
            sc.add_structure(structure)

    logger.debug(sc.__repr__())

    return sc

def harmonic_properties(
    structure: Structure,
    supercell_matrix: np.ndarray,
    fcs: ForceConstants,
    T: List,
    imaginary_tol: float = IMAGINARY_TOL
    # imaginary_tol: float = 0.025
) -> Tuple[Dict,Phonopy]:
    """
    Uses the force constants to extract phonon properties. Used for comparing
    the accuracy of force constant fits.
    Args:
        structure: The parent structure.
        supercell_matrix: The supercell transformation matrix.
        force_constants: The force constants in numpy format.
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.
    Returns:
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity
    """

    logger.info('Evaluating harmonic properties...')
    fcs2 = fcs.get_fc_array(2)
    fcs3 = fcs.get_fc_array(3)
    parent_phonopy = get_phonopy_structure(structure)
    phonopy = Phonopy(parent_phonopy, supercell_matrix=supercell_matrix)
    natom = phonopy.primitive.get_number_of_atoms()
    mesh = supercell_matrix.diagonal()*4
    
    phonopy.set_force_constants(fcs2)
    phonopy.set_mesh(mesh,is_eigenvectors=True,is_mesh_symmetry=False) #run_mesh(is_gamma_center=True)
    phonopy.run_thermal_properties(temperatures=T)
    logger.info('Thermal properties successfully run!')

    _, free_energy, entropy, Cv = phonopy.get_thermal_properties()
    free_energy *= 1000/sp.constants.Avogadro/eV2J/natom # kJ/mol to eV/atom
    entropy *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom
    Cv *= 1/sp.constants.Avogadro/eV2J/natom # J/K/mol to eV/K/atom

    freq = phonopy.mesh.frequencies # in THz
    # find imaginary modes at gamma
#    phonopy.run_qpoints([0, 0, 0])
#    gamma_eigs = phonopy.get_qpoints_dict()["frequencies"]

    print("############# This should be imaginary_tol")
    imaginary_tol = 0.025
    print(imaginary_tol)
    n_imaginary = int(np.sum(freq < -np.abs(imaginary_tol)))
    min_freq = np.min(freq)

    if n_imaginary == 0:
        logger.info('No imaginary modes!')
    else: # do not calculate these if imaginary modes exist
        logger.warning('Imaginary modes found!')

    return {
        "temperature": T,
        "free_energy": free_energy,
        "entropy": entropy,
        "heat_capacity": Cv,
        "n_imaginary": n_imaginary
        }, phonopy


def anharmonic_properties(
    phonopy: Phonopy,
    fcs: ForceConstants,
    T: List,
    Cv: np.ndarray,
    n_imaginary: float,
    # bulk_modulus: float = None
    bulk_modulus: float
) -> Tuple[Dict, Phonopy]:

    if n_imaginary == 0:
        logger.info('Evaluating anharmonic properties...')
        fcs2 = fcs.get_fc_array(2)
        fcs3 = fcs.get_fc_array(3)
        print("############# This should be bulk_modulus")
        print(bulk_modulus)
        grun, cte = gruneisen(phonopy,fcs2,fcs3,T,Cv,bulk_modulus=bulk_modulus)

        print("############# This should be T & cte")
        print(T)
        print(cte)


        if type(bulk_modulus) is float or int:
            # dLfrac = thermal_expansion(T,cte)  ## Fix this
            pass
        else:
            logger.warning('Thermal expansion cannot be calculated without bulk modulus input. Set to 0.')
            cte = np.zeros((len(T),3))
            dLfrac = np.zeros((len(T),3))
    else: # do not calculate these if imaginary modes exist
        logger.warning('Gruneisen and thermal expansion cannot be calculated with imaginary modes. All set to 0.')
        grun = np.zeros((len(T),3))
        cte = np.zeros((len(T),3))
        dLfrac = np.zeros((len(T),3))

    return {
        "temperature": T,
        "gruneisen": grun,
        "thermal_expansion": cte,
        # "expansion_ratio": dLfrac,
        }, phonopy


def get_total_grun(
        omega: np.ndarray,
        grun: np.ndarray,
        kweight: np.ndarray,
        T: float
) -> np.ndarray:
    total = 0
    weight = 0
    nptk = omega.shape[0]
    nbands = omega.shape[1]
    omega = abs(omega)*1e12*2*np.pi
    if T==0:
        total = np.zeros((3,3))
        grun_total_diag = np.zeros(3)
    else:
        for i in range(nptk):
            for j in range(nbands):
                x = hbar*omega[i,j]/(2.0*kB*T)
                dBE = (x/np.sinh(x))**2
                weight += dBE*kweight[i]
                total += dBE*kweight[i]*grun[i,j]
        total = total/weight
        grun_total_diag = np.array([total[0,2],total[1,1],total[2,0]])

        def percent_diff(a,b):
            return abs((a-b)/b)
        # This process preserves cell symmetry upon thermal expansion, i.e., it prevents
        # symmetry-identical directions from inadvertently expanding by different ratios
        # when the Gruneisen routine returns slighlty different ratios for those directions
        avg012 = np.mean((grun_total_diag[0],grun_total_diag[1],grun_total_diag[2]))
        avg01 = np.mean((grun_total_diag[0],grun_total_diag[1]))
        avg02 = np.mean((grun_total_diag[0],grun_total_diag[2]))
        avg12 = np.mean((grun_total_diag[1],grun_total_diag[2]))
        if percent_diff(grun_total_diag[0],avg012) < 0.1:
            if percent_diff(grun_total_diag[1],avg012) < 0.1:
                if percent_diff(grun_total_diag[2],avg012) < 0.1: # all siilar
                    grun_total_diag[0] = avg012
                    grun_total_diag[1] = avg012
                    grun_total_diag[2] = avg012
                elif percent_diff(grun_total_diag[2],avg02) < 0.1: # 0 and 2 similar
                    grun_total_diag[0] = avg02
                    grun_total_diag[2] = avg02
                elif percent_diff(grun_total_diag[2],avg12) < 0.1: # 1 and 2 similar
                    grun_total_diag[1] = avg12
                    grun_total_diag[2] = avg12
                else:
                    pass
            elif percent_diff(grun_total_diag[1],avg01) < 0.1: # 0 and 1 similar
                grun_total_diag[0] = avg01
                grun_total_diag[1] = avg01
            elif percent_diff(grun_total_diag[1],avg12) < 0.1: # 1 and 2 similar
                grun_total_diag[1] = avg12
                grun_total_diag[2] = avg12
            else:
                pass
        elif percent_diff(grun_total_diag[0],avg01) < 0.1: # 0 and 1 similar
            grun_total_diag[0] = avg01
            grun_total_diag[1] = avg01
        elif percent_diff(grun_total_diag[0],avg02) < 0.1: # 0 and 2 similar
            grun_total_diag[0] = avg02
            grun_total_diag[2] = avg02
        else: # nothing similar
            pass
        
    return grun_total_diag


def gruneisen(
        phonopy: Phonopy,
        fcs2: np.ndarray,
        fcs3: np.ndarray,
        temperature: List,
        Cv: np.ndarray, # in eV/K/atom
        # bulk_modulus: float = None # in GPa
        bulk_modulus: float # in GPa
) -> Tuple[List,List]:

    gruneisen = Gruneisen(fcs2,fcs3,phonopy.supercell,phonopy.primitive)
    gruneisen.set_sampling_mesh(phonopy.mesh_numbers,is_gamma_center=True)
    gruneisen.run()
    grun = gruneisen.get_gruneisen_parameters() # (nptk,nmode,3,3)
    omega = gruneisen._frequencies
    qp = gruneisen._qpoints
    kweight = gruneisen._weights
    grun_tot = list()
    for temp in temperature:
        grun_tot.append(get_total_grun(omega,grun,kweight,temp))
    grun_tot = np.nan_to_num(np.array(grun_tot))
    
    # linear thermal expansion coefficient
    if bulk_modulus is None:
        cte = None
    else:
        Cv *= eV2J*phonopy.primitive.get_number_of_atoms() # eV/K/atom to J/K 
        vol = phonopy.primitive.get_volume()
        cte = grun_tot*(Cv.repeat(3).reshape((len(Cv),3)))/(vol/10**30)/(bulk_modulus*10**9)/3
        cte = np.nan_to_num(cte)    
        logger.info('Gruneisen: \n {}'.format(grun_tot))
        logger.info('CTE: \n {}'.format(cte))    
    return grun_tot, cte


def thermal_expansion(
        temperature: List,
        cte: List,
        T: Optional[float]=None
) -> np.ndarray:
    print(temperature)
    print(cte)

    print(len(temperature))
    print(len(cte))

    assert len(temperature)==len(cte)
    if 0 not in temperature:
        temperature = [0] + temperature
        cte = [[0,0,0]] + cte
    temperature = np.array(temperature)
    ind = np.argsort(temperature)
    temperature = temperature[ind]
    cte = np.array(cte)[ind]
    # linear expansion fraction
    dLfrac = copy(cte)
    for t in range(len(temperature)):
        dLfrac[t,:] = np.trapz(cte[:t+1,:],temperature[:t+1],axis=0)
    dLfrac = np.nan_to_num(dLfrac)
    logger.info('dLfrac: \n {}'.format(dLfrac))
    if T is None:
        return dLfrac
    else:
        try:
            T_ind = np.where(temperature==T)[0][0]
            return np.array([dLfrac[T_ind]])
        except:
            raise ValueError('Designated T does not exist in the temperature array!')


def run_renormalization(
        thermal_data,
        n_imaginary,
        structure: Structure,
        supercell: Atoms,
        supercell_matrix: np.ndarray,
        cs: ClusterSpace,
        fcs: ForceConstants,
        param: np.ndarray,
        T: float,
        nconfig: int,
        max_iter: int,
        conv_tresh: float,
        renorm_method: str,
        fit_method: str,
        # bulk_modulus: float = None,
        bulk_modulus: float,
        # phonopy_orig: Phonopy = None,
        phonopy_orig,
        imaginary_tol: float = IMAGINARY_TOL,
) -> Dict:
    """
    Uses the force constants to extract phonon properties. Used for comparing
    the accuracy of force constant fits.
    Args:
        structure: pymatgen Structure
            The parent structure.
        supercell : ase Atoms
            Original supercell object  
        supercell_matrix: The supercell transformation matrix.
        fcs: ForceConstants from previous fitting or renormalization
        imaginary_tol: Tolerance used to decide if a phonon mode is imaginary,
            in THz.
    Returns:
        A tuple of the number of imaginary modes at Gamma, the minimum phonon
        frequency at Gamma, and the free energy, entropy, and heat capacity
    """
    renorm = Renormalization(cs,supercell,fcs,param,T,renorm_method,fit_method)
    fcp, fcs, param = renorm.renormalize(nconfig,conv_tresh)

    renorm_data, phonopy = harmonic_properties(
        structure, supercell_matrix, fcs, [T], imaginary_tol
    )

    # if renorm_data["n_imaginary"] == 0:
    #     logger.info('Renormalized phonon is completely real at T = {} K!'.format(T))
    #     anharmonic_data, phonopy = anharmonic_properties(
    #         phonopy, fcs, [T], thermal_data["heat_capacity"], n_imaginary, bulk_modulus=bulk_modulus
	# )
    if renorm_data["n_imaginary"] == 0:
        logger.info('Renormalized phonon is completely real at T = {} K!'.format(T))
        anharmonic_data, phonopy = anharmonic_properties(
            phonopy, fcs, [T], thermal_data, n_imaginary, bulk_modulus=bulk_modulus
	)    
    else:
        anharmonic_data = dict()
        anharmonic_data["gruneisen"] = np.array([[0,0,0]])
        anharmonic_data["thermal_expansion"] = np.array([[0,0,0]])

    print("#######This is the original & new phonopy object")
    print(phonopy_orig)
    print(phonopy_orig.mesh)
    # print(phonopy_orig.mesh.frequencies)
    # print(phonopy)
    # print(phonopy.mesh)
    # print(phonopy.mesh.frequencies)

    # omega0 = phonopy_orig.mesh.frequencies # THz
    omega_TD = phonopy.mesh.frequencies # THz
#    natom = phonopy.primitive.get_number_of_atoms()
    # correction_S, correction_4 = FE_correction(omega0,omega_TD,T) # eV/atom

    renorm_data.update(anharmonic_data)
    # renorm_data["free_energy_correction_S"] = correction_S[0]
    renorm_data["fcp"] = fcp
    renorm_data["fcs"] = fcs
    renorm_data["param"] = param
    
    return renorm_data



# def setup_TE_iter(cutoffs,parent_structure,temperatures,dLfracs):
#     parent_structure_TE = []
#     cs_TE = []
#     fcs_TE = []
#     for t, (T,dLfrac) in enumerate(zip(temperatures,dLfracs)):
#         new_atoms = AseAtomsAdaptor.get_atoms(parent_structure)
#         new_cell = Cell(np.transpose([new_atoms.get_cell()[:,i]*(1+dLfrac[0,i]) for i in range(3)]))
#         new_atoms.set_cell(new_cell,scale_atoms=True)
#         new_parent_structure = AseAtomsAdaptor.get_structure(new_atoms)
#         new_supercell_atoms = AseAtomsAdaptor.get_atoms(new_parent_structure*supercell_matrix)
#         new_cutoffs = [i*(1+np.linalg.norm(dLfrac)) for i in cutoffs]
#         while True:
#             new_cs = ClusterSpace(atoms,new_cutoffs,1e-3,acoustic_sum_rules=True)
#             if cs_TD.n_dofs == cs.n_dofs:
#                 break
#             elif cs_TD.n_dofs > cs.n_dofs:
#                 new_cutoffs = [i*0.999 for i in new_cutoffs]
#             elif cs_TD.n_dofs < cs.n_dofs:
#                 new_cutoffs = [i*1.001 for i in new_cutoffs]
#         cs_TE.append(new_cs)
#         parent_structure_TE.append(new_parent_structure)
#         new_fcp = ForceConstantsPotential(new_cs,param_real[t])
#         fcs_TE.append(new_fcp.get_force_constants(new_supercell_atoms))

#         return parent_structure_TE, cs_TE, fcs_TE