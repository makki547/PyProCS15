import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial.distance import cdist
import Bio.PDB
import Bio.PDB.vectors
from enum import Enum, auto

from pathlib import Path

AROMATIC_RESIDUES = ['HIS', 'PHE', 'TRP', 'TYR']


class HBDonorType(Enum):
            
    Ammonium = auto()
    Amide = auto()
    AlphaHydrogen = auto()
    Indole = auto()
    Guanidinium = auto()

class HBAcceptorType(Enum):
    Carboxylate = auto()
    Alcohol = auto()
    Amide = auto()

DIHEDRAL_CONTRIBUTION_ORDERS = [('CA', 'ca'), ('CB', 'cb'), ('C', 'co'), ('N', 'nh'), ('H', 'hn'), ('HA', 'ha')]

PROCS15_CONTRIBUTIONS = np.array(
    # CA, CB, C, N, HN, HA
    [
        [1, 1, 1, 1, 1, 1], #Dihedral
        [0, 0, 0, 1, 1, 1], #HN hydrogen bond, primary
        [0, 0, 0, 0, 1, 1], #HA hydrogen bond, primary
        [1, 0, 1, 1, 1, 1], #HN hydrogen bond, secondary
        [1, 0, 1, 1, 1, 1], #HA hydrogen bond, secondary
        [0, 0, 0, 0, 1, 1], #Ring current
    ]
)

class PyProCS15:
    
    def __init__(self, structure_file, procs_param_file_dir = None):
        
        from os import getenv
        
        self.procs_param_file_dir = procs_param_file_dir
        if self.procs_param_file_dir is None:
            self.procs_param_file_dir = getenv('PROCS15DIR')

        if self.procs_param_file_dir is None:
            raise ProCS15ParameterFileException()
        
        self.procs_dataset = self._ProCS15_dataset(self.procs_param_file_dir)
        self.load_structure(structure_file)
        

    def load_structure(self, filename):
        
        self.structure = self._get_structure_object(filename)
        self.hbond_dist_cache = self._Hydrogen_bond_distance_cache()        
        self._prepare()

    def _get_structure_object(self, filename):
        
        import re
        
        pdb_pattern = re.compile(r'.*\.(pdb|ent)')
        cif_pattern = re.compile(r'.*\.cif')

        if pdb_pattern.match(filename.lower()) is not None:
            parser = Bio.PDB.PDBParser()
        elif cif_pattern.match(filename.lower()) is not None:
            parser = Bio.PDB.MMCIFParser()
        else:
            raise InvalidFileTypeException(filename)
        
        return parser.get_structure(filename, filename)
        
    def _prepare(self):
        
        #add attributes for chemical shift calculation to Bio.PDB.Residue
        self.all_residues = []
        for model in self.structure:
            
            model_id = model.get_id()
            for chain in model:
                
                chain_id = chain.get_id()
                for residue in chain:
                    
                    
                    hetflag, resid, _ = residue.get_id()
                    if hetflag.strip() != '':
                        residue.hetatm = True
                        continue
                    
                    residue.model = model_id
                    residue.chain = chain_id
                    residue.resid = resid
                    
                    self.all_residues.append( (model_id, chain_id, resid) )
                    
                    resname = residue.get_resname()
                    
                    if resname == 'HID' or resname == 'HIE':
                        residue.resname = resname = 'HIS'
                        
                    
                    try:
                        residue.next = self.structure[model_id][chain_id][resid + 1]
                    except KeyError:
                        residue.next = None
                    except Exception as e:
                        raise e
                    
                    try:
                        residue.prev = self.structure[model_id][chain_id][resid - 1]
                    except KeyError:
                        residue.prev = None
                    except Exception as e:
                        raise e
                    
                                        
                    residue.hetatm = False
        
        
        #collect molecular properties for chemical shift calculation    
        for model in self.structure:
            
            model_id = model.get_id()
            for chain in model:
                
                chain_id = chain.get_id()
                for residue in chain:
                    
                    if residue.hetatm: continue
                    
                    resname = residue.get_resname()
                    
                    residue.dihedral_contribution = self._Dihedral_contribution(residue)
                    
                    try:
                        residue.dihedral_contribution.calc_dihedral_angles()    
                    except AtomMissingException:
                        print(f'Atom not found in {resid}-th residue, this and the adjacent residues will be ignored', file = sys.stderr)
                    
                    
                    residue.hydrogen_bonds = self._Hydrogen_bond(residue, self.hbond_dist_cache)
           
                    
                    if resname in AROMATIC_RESIDUES:
                        residue.ring_current_donor = self._RingCurrentDonor(residue)
                    else:
                        residue.ring_current_donor = None
                        
    def _calc_ring_contribution(self, model_id, chain_id, resid):

        #only hydrogen atoms are calculated
        NATOM = len(DIHEDRAL_CONTRIBUTION_ORDERS)
        HA_ID = 5
        HN_ID = 4

        atoms = [(HA_ID, 'HA'), (HN_ID, 'H')]
        gly_atoms = [(HA_ID, 'HA2'), (HA_ID, 'HA3'), (HN_ID, 'H')]
        gly12_atoms = [(HA_ID, 'HA2'), (HA_ID, 'HA3'), (HN_ID, 'H')]
        pro_atoms = [(HA_ID, 'HA')]

        contrib = np.zeros(NATOM)

        residue = self.structure[model_id][chain_id][resid]
        if residue.resname == 'GLY':
            if 'HA2' in residue and 'HA3' in residue:
                atoms = gly_atoms
            elif 'HA1' in residue and 'HA2' in residue:
                atoms = gly12_atoms
            else:
                AtomMissingException('Backbone hydrogens are missing')
                
        elif residue.resname == 'PRO':
            atoms = pro_atoms

        for i, atom in atoms:
            try:
                hydrogen = residue[atom]
                for residue in self.structure.get_residues():
                    if residue.hetatm:
                        continue
                    if residue.model != model_id:
                        continue

                    if (residue.chain == chain_id) and (np.abs(residue.resid - resid) < 2):
                        continue

                    if residue.ring_current_donor is None:
                        continue

                    contrib[i] += residue.ring_current_donor.get_shielding(hydrogen)

            except KeyError:
                AtomMissingException('Backbone hydrogens are missing')
            except Exception as e:
                raise e
            
        return contrib


                    
                    
    def calc_shieldings(self, targets = None):
        
        if targets is None:
            targets_ = self.all_residues
        else:
            targets_ = targets

        contribs = np.zeros( tuple([len(targets_)]) + PROCS15_CONTRIBUTIONS.shape ) 
        

        
        for i, (model_id, chain_id, resid) in enumerate(targets_):
            try:
                dihedral_contrib = self.structure[model_id][chain_id][resid].dihedral_contribution.get_contribution(self.procs_dataset)

                hn_hb_primary_contrib, ha_hb_primary_contrib = self.structure[model_id][chain_id][resid].hydrogen_bonds.get_primary_contribution(self.structure[model_id].get_residues(), self.procs_dataset, self.hbond_dist_cache)
            
                hn_hb_secondary_contrib, ha_hb_secondary_contrib = self.structure[model_id][chain_id][resid].hydrogen_bonds.get_secondary_contribution(self.structure[model_id].get_residues(), self.procs_dataset, self.hbond_dist_cache)
            
                ring_current_contrib = self._calc_ring_contribution(model_id, chain_id, resid)

                contribs[i,0,:] = dihedral_contrib
                contribs[i,1,:] = hn_hb_primary_contrib
                contribs[i,2,:] = ha_hb_primary_contrib
                contribs[i,3,:] = hn_hb_secondary_contrib
                contribs[i,4,:] = ha_hb_secondary_contrib
                contribs[i,5,:] = ring_current_contrib
            except AtomMissingException as e:
                print(e, file=sys.stderr)
            except Exception as e:
                raise e

        contribs = np.sum(contribs * PROCS15_CONTRIBUTIONS, axis = 1)
        if targets is not None:
            return contribs
        else:
            return targets_, contribs 
    
    class _ProCS15_dataset:
        
        class InterpolationType(Enum):
            Cubic = auto()
            Nearest = auto()
            
        
        
        AMINO_ACIDS = [
            ('ALA', InterpolationType.Cubic),
            ('CYS', InterpolationType.Cubic),
            ('ASP', InterpolationType.Nearest),
            ('GLU', InterpolationType.Nearest),
            ('PHE', InterpolationType.Nearest),
            ('GLY', InterpolationType.Cubic),
            ('HIS', InterpolationType.Nearest),
            ('ILE', InterpolationType.Nearest),
            ('LYS', InterpolationType.Nearest),
            ('LEU', InterpolationType.Nearest),            
            ('MET', InterpolationType.Nearest),
            ('ASN', InterpolationType.Nearest),
            ('PRO', InterpolationType.Cubic),
            ('GLN', InterpolationType.Nearest),
            ('ARG', InterpolationType.Nearest),
            ('SER', InterpolationType.Cubic),
            ('THR', InterpolationType.Nearest),
            ('VAL', InterpolationType.Cubic),
            ('TRP', InterpolationType.Nearest),
            ('TYR', InterpolationType.Nearest) \
            ]
        
        ATOM_TYPES = ['ca', 'cb', 'co', 'nh', 'hn', 'ha']
        
        DIHEDRAL_DATASET_NAMES = [(f'{aa}_{atom}.npy', itp_type)  for aa, itp_type in AMINO_ACIDS for atom in ['ca', 'cb', 'co', 'nh', 'hn', 'ha']]
        
        HBOND_DATASET_NAMES = [(HBAcceptorType.Amide, 'delta_hbond_251_91_361_12.npy'), (HBAcceptorType.Alcohol, 'delta_hbond_Alcohol_251_91_361_6.npy'), (HBAcceptorType.Carboxylate, 'delta_hbond_Carboxylate_251_91_361_6.npy')]
        HABOND_DATASET_NAMES = [(HBAcceptorType.Alcohol, 'delta_halphabond_alcohol_221_91_361_6.npy'), (HBAcceptorType.Carboxylate, 'delta_halphabond_carboxy_221_91_361_6.npy'), (HBAcceptorType.Amide, 'delta_halphabond_oxygen_221_91_361_12.npy')]
        
        HBOND_R_MIN = 1.5
        HBOND_R_DELTA = 0.01
        HBOND_TETHA_MIN = 90.0
        HBOND_TETHA_DELTA = 1.0
        HBOND_RHO_MIN = 0.0
        HBOND_RHO_DELTA = 1.0

        HABOND_R_MIN = 1.8
        HABOND_R_DELTA = 0.01
        HABOND_TETHA_MIN = 90.0
        HABOND_TETHA_DELTA = 1.0
        HABOND_RHO_MIN = 0.0
        HABOND_RHO_DELTA = 1.0
        
        
        
        def __init__(self, path_to_dataset_dir):
            
            if isinstance(path_to_dataset_dir, Path):
                self.path = path_to_dataset_dir
            else:
                self.path = Path(path_to_dataset_dir)
                
            self.dihedral_dataset = {resname: {atomname: self._setup_dihedral_dataset(self.path / Path(f'{resname}_{atomname}.npy'), itp) for atomname in self.ATOM_TYPES} for resname, itp in self.AMINO_ACIDS}
            self.hbond_dataset , self.habond_dataset = self._setup_hydrogen_bond_dataset(self.path)

            self.dihedral_ala_std_prev = np.zeros( len(DIHEDRAL_CONTRIBUTION_ORDERS) )
            self.dihedral_ala_std_next = np.zeros( len(DIHEDRAL_CONTRIBUTION_ORDERS) )
            for i, atom in enumerate(self.ATOM_TYPES):
                self.dihedral_ala_std_prev[i] = self.dihedral_dataset['ALA'][atom][2]([-120, 140])[0]
                self.dihedral_ala_std_next[i] = self.dihedral_dataset['ALA'][atom][0]([-120, 140])[0]

        def _setup_dihedral_dataset(self, filepath, interpolation_type):
            
            data = np.lib.format.open_memmap(filepath)
            
            angles = tuple([np.linspace(-180, 180, num = num) for num in data.shape[1:]])
            
            if interpolation_type == self.InterpolationType.Nearest:
                method = 'nearest'
            elif interpolation_type == self.InterpolationType.Cubic:
                method = 'cubic'
            else:
                method = None
            
            dataset = []
            
            for i in range(3):
                
                dataset.append( RegularGridInterpolator(angles, data[i,...], method = method) )
            
            return dataset
        
        def _setup_hydrogen_bond_dataset(self, filepath):
            
            def make_regular_grids(dataset_names, bond_min_delta):
                dataset = {}
                for (name, filename) in dataset_names:
                    d = np.lib.format.open_memmap(filepath / filename)
                    shape = d.shape

                    indices = tuple(map(lambda i: bond_min_delta[i][0] + np.arange(shape[i])*bond_min_delta[i][1]  , range(3)))
                    dataset[name] = {}

                    max_dist = indices[0][-1]

                    for i, atom in enumerate(self.ATOM_TYPES):
                        dataset[name][atom] = RegularGridInterpolator(indices, d[...,i], method = 'nearest', bounds_error = False, fill_value = 0.0)
                return dataset, max_dist

            
            
            hbond_min_delta = [
                (self.HBOND_R_MIN, self.HBOND_R_DELTA),
                (self.HBOND_TETHA_MIN, self.HBOND_TETHA_DELTA),
                (self.HBOND_RHO_MIN, self.HBOND_RHO_DELTA) \
                ]
            
            habond_min_delta = [
                (self.HABOND_R_MIN, self.HABOND_R_DELTA),
                (self.HABOND_TETHA_MIN, self.HABOND_TETHA_DELTA),
                (self.HABOND_RHO_MIN, self.HABOND_RHO_DELTA) \
                ]
            
            hbond_dataset, self.HBOND_R_MAX = make_regular_grids(self.HBOND_DATASET_NAMES, hbond_min_delta)
            habond_dataset, self.HABOND_R_MAX = make_regular_grids(self.HABOND_DATASET_NAMES, habond_min_delta)

            return hbond_dataset, habond_dataset

                
                
                
            
            
            
    
    class _Dihedral_contribution:
        
        def __init__(self, residue):
            self.phi = 0.0
            self.psi = 0.0
            self.chi = []
            self.ignoring = False
            self.residue = residue
            self.resname = residue.get_resname()
            
            
            
    
        def calc_dihedral_angles(self):
            # see http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html
            # also see the Phaistos-1.0 source codes, src/protein/atom.cpp, Atom::init
            
            CHI_ATOMS = {\
                        'GLY': [],
                        'ALA': [],
                        'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                        'GLU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
                        'HIS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'ND1']],
                        'THR': [['N', 'CA', 'CB', 'OG1'], ['CA', 'CB', 'OG1', 'HG1']],
                        'SER': [['N', 'CA', 'CB', 'OG']],
                        'CYS': [['N', 'CA', 'CB', 'SG']],
                        'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD1']],
                        'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                        'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                        'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                        'PHE': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                        'PRO': [],
                        'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                        'LYS': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
                        'ARG': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
                        'GLN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
                        'MET': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
                        'VAL': [['CG1', 'CA', 'CB', 'CG2']], #Special case
                        }

            
            
            
            
            
            if self.residue.hetatm:
                return
            
            res_cur = self.residue
            
            try:
                res_pre = self.residue.prev
                
                if res_pre is None:
                    raise KeyError
                
                if res_pre.hetatm:
                    raise KeyError
                
                self.phi = np.rad2deg( 
                                        Bio.PDB.vectors.calc_dihedral(
                                            res_pre['C'].get_vector(),
                                            res_cur['N'].get_vector(),
                                            res_cur['CA'].get_vector(),
                                            res_cur['C'].get_vector()
                                            )
                                        )
            except KeyError:
                self.ignoring = True
                raise AtomMissingException()
            except Exception as e:
                raise e
            
            try:
                res_pro = self.residue.next
                if res_pro is None:
                    raise KeyError
                
                if res_pro.hetatm:
                    raise KeyError
                
                self.psi = np.rad2deg( 
                                        Bio.PDB.vectors.calc_dihedral(
                                            res_cur['N'].get_vector(),
                                            res_cur['CA'].get_vector(),
                                            res_cur['C'].get_vector(),
                                            res_pro['N'].get_vector()
                                            )
                                        )
            except KeyError:
                self.ignoring = True
                raise AtomMissingException()
            except Exception as e:
                raise e
            
            
            chi_atoms = CHI_ATOMS[self.resname]

            if len(chi_atoms) == 0:
                return
            
            try:
                if self.resname == 'VAL':

                    cg1cg2 = Bio.PDB.vectors.calc_dihedral( *tuple(map(lambda a:res_cur[a].get_vector(), chi_atoms[0]) ) )
                    if cg1cg2 > 0:
                        chi_atoms = [['N', 'CA', 'CB', 'CG1']]
                    else:
                        chi_atoms = [['N', 'CA', 'CB', 'CG2']]

                self.chi = [ np.rad2deg( Bio.PDB.vectors.calc_dihedral( *tuple(map(lambda a:res_cur[a].get_vector(), atoms)  ) ) ) for atoms in chi_atoms]
            except KeyError:
                self.ignoring = True
                raise AtomMissingException()
            except Exception as e:
                raise e
        
        def get_contribution(self, dataset):

            

            if self.residue.prev is None or self.residue.next is None:
                raise AtomMissingException()
            

            if self.ignoring or self.residue.prev.dihedral_contribution.ignoring or self.residue.next.dihedral_contribution.ignoring:
                raise AtomMissingException()
            
            contribs = np.zeros(len(DIHEDRAL_CONTRIBUTION_ORDERS))
            
            for i, (pdbname, procsname) in enumerate(DIHEDRAL_CONTRIBUTION_ORDERS):

                contrib = dataset.dihedral_dataset[self.resname][procsname][1]([self.phi, self.psi] + self.chi)[0]
                contrib_prev = \
                    dataset.dihedral_dataset[self.residue.prev.resname][procsname][2]([self.residue.prev.dihedral_contribution.phi, self.residue.prev.dihedral_contribution.psi] + self.residue.prev.dihedral_contribution.chi)[0] \
                     - dataset.dihedral_ala_std_prev[i]
                
                contrib_next = \
                    dataset.dihedral_dataset[self.residue.next.resname][procsname][0]([self.residue.next.dihedral_contribution.phi, self.residue.next.dihedral_contribution.psi] + self.residue.next.dihedral_contribution.chi)[0] \
                     - dataset.dihedral_ala_std_next[i]
                
                contribs[i] = contrib + contrib_next + contrib_prev

            return contribs
        
    class _RingCurrentDonor:
        
        def __init__(self, residue):
            
            BFACTOR = 30.42 # A^3, from 10.1021/ct2002607, Point-dipole model
            INTENSITIES = {'PHE': (1.0,), 'TYR': (0.81,), 'HIS': (0.69,), 'TRP': (0.57, 1.02)}  # from 10.1021/ct2002607, Point-dipole model, deprotonated HIS was not used, see lib_definitions.h in ProCS15 
                        
            self.residue = residue
            self.resname = residue.get_resname()
            
            assert self.resname in AROMATIC_RESIDUES, 'Non-aromatic residue was selected.'
            
            self.ring_atoms = self._get_ring_atoms()
            self.ring_coms = self._get_ring_coms()
            self.ring_normal_vectors = self._get_normal_vectors()
            
            self.intensities = BFACTOR * np.array(INTENSITIES[self.resname])
            
            
        def _get_ring_atoms(self):
            
            if self.resname == 'PHE' or self.resname == 'TYR':
                return [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']]
            elif self.resname == 'HIS':
                return [['CG', 'ND1', 'CD2', 'CE1', 'NE2']]
            elif self.resname == 'TRP':
                return [['CG', 'CD1', 'CD2', 'NE1', 'CE2'], ['CD1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']]
            else:
                return None
            
        def _get_ring_coms(self):
            
            coms = np.zeros((len(self.ring_atoms), 3))
            
            for i, atoms in enumerate(self.ring_atoms):
                
                coords = np.zeros((len(atoms), 3))
                for j, atom in enumerate(atoms):
                    coords[j,:] = np.array(self.residue[atom].get_coord())
                    
                coms[i,:] = np.mean(coords, axis = 0)
                
            return coms
                    
            
        def _get_normal_vectors(self):
            
            norm_vecs = np.zeros((len(self.ring_atoms), 3))
            
            for i, atoms in enumerate(self.ring_atoms):
                p1 = np.array(self.residue[atoms[0]].get_coord()) - self.ring_coms[i,:]
                p2 = np.array(self.residue[atoms[1]].get_coord()) - self.ring_coms[i,:]
                
                nv = np.cross(p1, p2)
                norm_vecs[i,:] = nv / np.linalg.norm(nv)
                
            return norm_vecs
        
        def get_shielding(self, atom):
            CUTOFF = 8.0
            atom_vec = np.array(atom.get_coord()) - self.ring_coms
            
            dist = np.linalg.norm(atom_vec)
            if dist > CUTOFF:
                return 0.0
            
            cos_term = np.sum(atom_vec * self.ring_normal_vectors, axis = -1)/dist
            
            return np.sum(self.intensities * (1 - 3*cos_term**2)/dist**3)
            
            
            
    class _Hydrogen_bond:
        

        
        def __init__(self, residue, distance_cache):
            
            self.donors = [] # list of tuples, tuples should be like (Hydrogen atom obj, second atom obj, DonorType)
            self.acceptors = [] # list of tuples, tuples should be like (Oxygen atom obj, second atom obj, third atom obj, AcceptorType)
            
            self.residue = residue

            self._add_donors(residue)
            self._add_acceptors(residue)

            for donor in self.donors:
                distance_cache.add_donor_atom(donor[0])

            for acceptor in self.acceptors:
                distance_cache.add_acceptor_atom(acceptor[0])

            pass
            
        def _add_donor_atoms(self, destination, first_residue,  first_atom_names,  second_residue, second_atom_names, hb_type):
            
            try:
                if type(first_atom_names) is list and type(second_atom_names) is list:
                    destination += [(first_residue[first_atom], second_residue[second_atom], hb_type) for first_atom, second_atom in zip(first_atom_names, second_atom_names)]
                elif type(first_atom_names) is list:
                    destination += [(first_residue[atom], second_residue[second_atom_names], hb_type) for atom in first_atom_names]
                else:
                    destination.append( (first_residue[first_atom_names], second_residue[second_atom_names], hb_type) )
                    
            except KeyError:
                print(f'!!Warning!! Hydrogen not found, {first_residue} {second_residue}', file = sys.stderr)
            except Exception as e:
                raise e
            
        def _add_acceptor_atoms(self, destination, first_residue,  first_atom_names,  second_residue, second_atom_names, third_residue, third_atom_names, hb_type):
            
            try:
                if type(first_atom_names) is list and type(third_atom_names) is list:
                    destination += [(first_residue[first_atom], second_residue[second_atom_names], third_residue[third_atom], hb_type) for first_atom, third_atom in zip(first_atom_names, third_atom_names)]
                elif type(first_atom_names) is list:
                    destination += [(first_residue[atom], second_residue[second_atom_names], third_residue[third_atom_names], hb_type) for atom in first_atom_names]
                else:
                    destination.append( (first_residue[first_atom_names], second_residue[second_atom_names], third_residue[third_atom_names], hb_type) )
                    
            except KeyError:
                print('!!Warning!! Oxygen not found', file = sys.stderr)
            except Exception as e:
                raise e
            
        def _add_donors(self, residue):
            
            resname = residue.get_resname()
            
            if 'H1' in residue:
                self._add_donor_atoms(self.donors, residue, [f'H{i}' for i in range(1, 4)], residue, 'N', HBDonorType.Ammonium)
            elif resname != 'PRO':
                self._add_donor_atoms(self.donors, residue, 'H', residue, 'N', HBDonorType.Amide)
                
            if resname == 'GLY':
                if 'HA2' in residue and 'HA3' in residue:
                    self._add_donor_atoms(self.donors, residue, [f'HA{i}' for i in range(2, 4)], residue, 'CA', HBDonorType.AlphaHydrogen)
                elif 'HA1' in residue and 'HA2' in residue:
                    self._add_donor_atoms(self.donors, residue, [f'HA{i}' for i in range(1, 3)], residue, 'CA', HBDonorType.AlphaHydrogen)
            else:
                self._add_donor_atoms(self.donors, residue, 'HA', residue, 'CA', HBDonorType.AlphaHydrogen)
                
            
            if resname == 'ASN':
                self._add_donor_atoms(self.donors, residue, [f'HD2{i}' for i in range(1, 3)], residue, 'ND2', HBDonorType.Amide)
            elif resname == 'GLN':
                self._add_donor_atoms(self.donors, residue, [f'HE2{i}' for i in range(1, 3)], residue, 'NE2', HBDonorType.Amide)
            elif resname == 'HIS':
                if 'HD1' in residue:
                    self._add_donor_atoms(self.donors, residue, 'HD1', residue, 'ND1', HBDonorType.Indole)
                if 'HE2' in residue:
                    self._add_donor_atoms(self.donors, residue, 'HE2', residue, 'NE2', HBDonorType.Indole)
            elif resname == 'TRP':
                self._add_donor_atoms(self.donors, residue, 'HE1', residue, 'NE1', HBDonorType.Indole)
            elif resname == 'LYS':
                self._add_donor_atoms(self.donors, residue, [f'HZ{i}' for i in range(1, 3)], residue, 'NZ', HBDonorType.Ammonium)
            elif resname == 'ARG':
                self._add_donor_atoms(self.donors, residue, ['HH11', 'HH12', 'HH21', 'HH22', 'HE'], residue, ['NH1', 'NH1', 'NH2', 'NH2', 'NE'], HBDonorType.Guanidinium)
                
        def _add_acceptors(self, residue):
            
            resname = residue.get_resname()
            
            if 'OXT' in residue:
                self._add_acceptor_atoms(self.acceptors, residue, ['O', 'OXT'], residue, 'C', residue, ['OXT', 'O'], HBAcceptorType.Carboxylate)
            else:
                if residue.next is not None:
                    self._add_acceptor_atoms(self.acceptors, residue, 'O', residue, 'C', residue.next, 'N', HBAcceptorType.Amide)
                    
            if resname == 'SER':
                self._add_acceptor_atoms(self.acceptors, residue, 'OG', residue, 'CB', residue, 'HG', HBAcceptorType.Alcohol)
            elif resname == 'THR': 
                self._add_acceptor_atoms(self.acceptors, residue, 'OG1', residue, 'CB', residue, 'HG1', HBAcceptorType.Alcohol)
            elif resname == 'TYR':
                self._add_acceptor_atoms(self.acceptors, residue, 'OH', residue, 'CZ', residue, 'HH', HBAcceptorType.Alcohol)
            elif resname == 'ASN':
                self._add_acceptor_atoms(self.acceptors, residue, 'OD1', residue, 'CG', residue, 'ND2', HBAcceptorType.Amide)
            elif resname == 'GLN':
                self._add_acceptor_atoms(self.acceptors, residue, 'OE1', residue, 'CD', residue, 'NE2', HBAcceptorType.Amide)
            elif resname == 'ASP':
                self._add_acceptor_atoms(self.acceptors, residue, ['OD1', 'OD2'], residue, 'CG', residue, 'CB', HBAcceptorType.Carboxylate)
            elif resname == 'GLU':
                self._add_acceptor_atoms(self.acceptors, residue, ['OE1', 'OE2'], residue, 'CD', residue, 'CG', HBAcceptorType.Carboxylate)
            pass

        def get_primary_contribution(self, residues, dataset, dist_cache, water_correction = True):

            atoms = PyProCS15._ProCS15_dataset.ATOM_TYPES
            HN_ID = 4
            WATER_CORR = 2.07
            hn_contrib = np.zeros(len(atoms))
            ha_contrib = np.zeros(len(atoms))

            for counter_residue in residues:


                if counter_residue.hetatm:
                    continue

                if counter_residue.model != self.residue.model:
                    continue

                #if ( counter_residue.chain == self.residue.chain ) and np.abs(counter_residue.resid - self.residue.resid) < 2:
                #    continue

                if self.residue == counter_residue:
                    continue
                
                for donor in self.donors:

                    v1 = donor[0].get_vector()
                    
                    for acceptor in counter_residue.hydrogen_bonds.acceptors:

                        dist = dist_cache.get_distance(donor[0], acceptor[0])

                        if donor[2] == HBDonorType.AlphaHydrogen:
                            dist_max = dataset.HABOND_R_MAX
                            dataset_ = dataset.habond_dataset[acceptor[3]]
                            contrib = ha_contrib
                        else:
                            dist_max = dataset.HBOND_R_MAX
                            dataset_ = dataset.hbond_dataset[acceptor[3]]
                            contrib = hn_contrib

                        if dist > dist_max:
                            continue

                        
                        v2 = acceptor[0].get_vector()
                        v3 = acceptor[1].get_vector()
                        v4 = acceptor[2].get_vector()
    

                        theta = np.rad2deg( Bio.PDB.vectors.calc_angle(v1, v2, v3) )
                        rho = np.rad2deg( Bio.PDB.vectors.calc_dihedral(v1, v2, v3, v4) )

                        for i,atom in enumerate(atoms):
                            contrib[i] += dataset_[atom]([dist, theta, rho])[0]


                        pass
                    pass

            #amide proton does not form any hydrogen bonds to the other residues
            if water_correction and (self.residue.resname != 'PRO') and np.abs(hn_contrib[HN_ID] < 1.0E-6):
                hn_contrib[HN_ID] -= WATER_CORR

            return hn_contrib, ha_contrib

        def get_secondary_contribution(self, residues, dataset, dist_cache):

            atoms = PyProCS15._ProCS15_dataset.ATOM_TYPES
            hn_contrib = np.zeros(len(atoms))
            ha_contrib = np.zeros(len(atoms))

            for counter_residue in residues:


                if counter_residue.hetatm:
                    continue

                if counter_residue.model != self.residue.model:
                    continue

                if ( counter_residue.chain == self.residue.chain ) and np.abs(counter_residue.resid - self.residue.resid) < 2:
                    continue

                
                for acceptor in self.acceptors:

                    if acceptor[0].get_name() != 'O':
                        continue

                    v2 = acceptor[0].get_vector()
                    v3 = acceptor[1].get_vector()
                    v4 = acceptor[2].get_vector()
                        
                    for donor in counter_residue.hydrogen_bonds.donors:

                        dist = dist_cache.get_distance(donor[0], acceptor[0])

                        if donor[2] == HBDonorType.AlphaHydrogen:
                            dist_max = dataset.HABOND_R_MAX
                            dataset_ = dataset.habond_dataset[acceptor[3]]
                            contrib = ha_contrib
                        else:
                            dist_max = dataset.HBOND_R_MAX
                            dataset_ = dataset.hbond_dataset[acceptor[3]]
                            contrib = hn_contrib

                        if dist > dist_max:
                            continue

                        v1 = donor[0].get_vector()
                        

                        theta = np.rad2deg( Bio.PDB.vectors.calc_angle(v1, v2, v3) )
                        rho = np.rad2deg( Bio.PDB.vectors.calc_dihedral(v1, v2, v3, v4) )

                        for i,atom in enumerate(atoms):
                            contrib[i] += dataset_[atom]([dist, theta, rho])[0]


                        pass
                    pass

            return hn_contrib, ha_contrib

    class _Hydrogen_bond_distance_cache:

        def __init__(self):
            self.donor_atoms = []
            self.last_donor_cache_id = -1

            self.acceptor_atoms = []
            self.last_acceptor_cache_id = -1


            self.distance_matrix = None

        def add_donor_atom(self, donor_atom):

            self.last_donor_cache_id += 1
            self.donor_atoms.append(donor_atom)
            donor_atom.donor_distance_cache_id = self.last_donor_cache_id
            pass

        def add_acceptor_atom(self, acceptor_atom):

            self.last_acceptor_cache_id += 1
            self.acceptor_atoms.append(acceptor_atom)
            acceptor_atom.acceptor_distance_cache_id = self.last_acceptor_cache_id
            pass

        def generate_distance_cache(self):

            donor_coords = np.array([atom.get_coord() for atom in self.donor_atoms])
            acceptor_coords = np.array([atom.get_coord() for atom in self.acceptor_atoms])

            self.distance_matrix = cdist(donor_coords, acceptor_coords, metric = 'euclidean')

        def get_distance(self, donor_atom, acceptor_atom):

            if self.distance_matrix is None:
                self.generate_distance_cache()
            
            return self.distance_matrix[donor_atom.donor_distance_cache_id, acceptor_atom.acceptor_distance_cache_id]



                
class InvalidFileTypeException(Exception):
    
    def __init__(self, arg = ''):
        self.arg = arg
        
    def __str__(self):
        
        if self.arg == '':
            return 'Use PDB or mmCIF file'
        else:
            return f'{self.arg} is not PDB or mmCIF file. Use PDB or mmCIF file.'



class AtomMissingException(Exception):
    
    def __init__(self, arg = ''):
        self.arg = arg
        
    def __str__(self):
        
        if self.arg == '':
            return 'Some atoms are missing.'
        else:
            return self.arg
        
class ProCS15ParameterFileException(Exception):
    
    def __init__(self, arg = ''):
        self.arg = arg
        
    def __str__(self):
        
        if self.arg == '':
            return 'ProCS15 parameter files cannot be loaded.\nDownload the file from: http://www.erda.dk/public/archives/YXJjaGl2ZS1pMHN3TXE=/procs/ProCSnumpyfiles.tar.bz2\nSet the directory to "PROCS15DIR" env. var. or provide to PyProCS constructor'
        else:
            return self.arg

if __name__ == '__main__':
    
    pyprocs15 = PyProCS15('1UBQ_amber.pdb')
    shieldings = pyprocs15.calc_shieldings()
    print(shieldings)
    
