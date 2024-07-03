import numpy as np
import Bio.PDB

AROMATIC_RESIDUES = ['HIS', 'PHE', 'TRP', 'TYR']

class PyProCS15:
    
    def __init__(self, structure_file):
        
        
        self.structure = self._get_structure_object(structure_file)
        
        self._add_attributes_to_biopdb_instances()


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
        
    def _add_attributes_to_biopdb_instances(self):
        
        
        for model in self.structure:
            
            for chain in model:
                
                for residue in chain:
                    
                    resname = residue.get_resname()
                                        
                    residue.phi = None
                    residue.psi = None
                    
                    residue.chi = []
                    
                    residue.hb_nho_hn = [] #for amide hydrogen bond, case 1
                    residue.hb_nho_co = [] #for amide hydrogen bond, case 2
                    
                    residue.hb_cahao_ha = [] #for Halpha hydrogen bond, case 1
                    residue.hb_cahao_co = [] #for Halpha hydrogen bond, case 2
                    
                    residue.ring_current_doners = []
                    
                    if resname in AROMATIC_RESIDUES:
                        residue.ring_current_donor = self._RingCurrentDonor(residue)
                    else:
                        residue.ring_current_donor = None
                    
                    

                
        
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
            
            self.intensities = BFACTOR * np.array(INTENSITIES[self.resname]).T
            
            
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

