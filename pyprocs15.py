import numpy as np
import Bio.PDB
import Bio.PDB.vectors

AROMATIC_RESIDUES = ['HIS', 'PHE', 'TRP', 'TYR']

class PyProCS15:
    
    def __init__(self, structure_file):
        
        
        self.structure = self._get_structure_object(structure_file)
        
        self._prepare()
        print(self.all_residues)

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
                    print(hetflag)
                    if hetflag != ' ':
                        residue.hetatm = True
                        continue
                    
                    self.all_residues.append( (model_id, chain_id, resid) )
                    
                    resname = residue.get_resname()
                                        
                    residue.hetatm = False
                    residue.phi = None
                    residue.psi = None
                    
                    residue.chi = []
                    
                    residue.hb_nho_hn = [] #for amide hydrogen bond, primary
                    residue.hb_nho_co = [] #for amide hydrogen bond, secondary
                    
                    residue.hb_cahao_ha = [] #for Halpha hydrogen bond, primary
                    residue.hb_cahao_co = [] #for Halpha hydrogen bond, secondary
                    
                    residue.ring_current_doners = []
                    
                    if resname in AROMATIC_RESIDUES:
                        residue.ring_current_donor = self._RingCurrentDonor(residue)
                    else:
                        residue.ring_current_donor = None
                    
    def calc_shieldings(self, targets = None):
        
        if targets is None:
            targets = self.all_residues
            
        for target in targets:
            self._calc_dihedral_angles_of_a_residue(target)
            model_id, chain_id, resid = target
            print(self.structure[model_id][chain_id][resid].phi)
        
        pass
    
    def _calc_dihedral_angles_of_a_residue(self, target):
        # see http://www.mlb.co.jp/linux/science/garlic/doc/commands/dihedrals.html
        CHI_ATOMS = {\
                    'GLY': [],
                    'ALA': [],
                    'ASP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                    'GLU': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
                    'HIS': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'ND1']],
                    'THR': [['N', 'CA', 'CB', 'OG1']],
                    'SER': [['N', 'CA', 'CB', 'OG']],
                    'CYS': [['N', 'CA', 'CB', 'SG']],
                    'ILE': [['N', 'CA', 'CB', 'CG1'], ['CA', 'CB', 'CG1', 'CD']],
                    'LEU': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'ASN': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'OD1']],
                    'TYR': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'PHE': [['N', 'CA', 'CB',' CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'PRO': [],
                    'TRP': [['N', 'CA', 'CB', 'CG'], ['CA', 'CB', 'CG', 'CD1']],
                    'LYS': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'CD'], ['CB', 'CG', 'CD', 'CE'], ['CG', 'CD', 'CE', 'NZ']],
                    'ARG': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'CD'], ['CB', 'CG', 'CD', 'NE'], ['CG', 'CD', 'NE', 'CZ']],
                    'GLN': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'CD'], ['CB', 'CG', 'CD', 'OE1']],
                    'MET': [['N', 'CA', 'CB',' CG'], ['CA', 'CB',' CG', 'SD'], ['CB', 'CG', 'SD', 'CE']],
                    'VAL': [['CG1', 'CA', 'CB', 'CG2']], #Special case
                    }

        model_id, chain_id, resid = target
        
        res_cur = self.structure[model_id][chain_id][resid]
        
        
        if res_cur.hetatm:
            return
        
        
        try:
            res_pre = self.structure[model_id][chain_id][resid - 1]
            if res_pre.hetatm:
                raise KeyError
            
            res_cur.phi = np.rad2deg( 
                                     Bio.PDB.vectors.calc_dihedral(
                                         res_pre['C'].get_vector(),
                                         res_cur['N'].get_vector(),
                                         res_cur['CA'].get_vector(),
                                         res_cur['C'].get_vector()
                                         )
                                     )
        except KeyError:
            pass
        except Exception as e:
            raise e
        
        try:
            res_pro = self.structure[model_id][chain_id][resid + 1]
            if res_pro.hetatm:
                raise KeyError
            
            res_cur.psi = np.rad2deg( 
                                     Bio.PDB.vectors.calc_dihedral(
                                         res_cur['N'].get_vector(),
                                         res_cur['CA'].get_vector(),
                                         res_cur['C'].get_vector(),
                                         res_pro['N'].get_vector()
                                         )
                                     )
        except KeyError:
            pass
        except Exception as e:
            raise e
        
        resname = res_cur.get_resname()

        chi_atoms = CHI_ATOMS[resname]

        if len(chi_atoms) == 0:
            return
        
        if resname == 'VAL':

            cg1cg2 = Bio.PDB.vectors.calc_dihedral( *(tuple(map(lambda a:res_cur[a].get_vector(), chi_atoms[0]) )) )
            if cg1cg2 > 0:
                chi_atoms = [['N', 'CA', 'CB', 'CG1']]
            else:
                chi_atoms = [['N', 'CA', 'CB', 'CG2']]

        res_cur.chi = [ np.rad2deg( Bio.PDB.vectors.calc_dihedral( *(tuple(map(lambda a:res_cur[a].get_vector(), atoms) ) ) ) ) for atoms in chi_atoms]            
        
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
            
            atom_vec = np.array(atom.get_coord()) - self.ring_coms
            
            dist = np.linalg.norm(atom_vec)
            cos_term = np.sum(atom_vec * self.ring_normal_vectors, axis = -1)/dist
            
            return np.sum(self.intensities * (1 - 3*cos_term**2)/dist**3)
            
            
            
            

            
        
class InvalidFileTypeException(Exception):
    
    def __init__(self, arg = ''):
        self.arg = arg
        
    def __str__(self):
        
        if self.arg == '':
            return 'Use PDB or mmCIF file'
        else:
            return f'{self.arg} is not PDB or mmCIF file. Use PDB or mmCIF file.'


if __name__ == '__main__':
    
    pyprocs15 = PyProCS15('1e12.cif')
    pyprocs15.calc_shieldings()
    
