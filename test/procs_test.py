from pyprocs15 import PyProCS15
import numpy
import scipy
import matplotlib.pyplot as plt
import pynmrstar


entry = pynmrstar.Entry.from_database(17769)

bmrb_cs = {}
bmrb_seq = {}

#loop = entry.get_loops_by_category('Atom_chem_shift')
    
elems = [cs.get_tag(['Seq_ID', 'Atom_ID', 'Val']) for cs in entry.get_loops_by_category('Atom_chem_shift')]

for seq_id, atom, shift in map(tuple, elems[0]):
    seq_id = int(seq_id)
    shift = float(shift)
    
    if seq_id not in bmrb_cs:
        bmrb_cs[seq_id] = {}
        bmrb_seq[seq_id] = atom

    bmrb_cs[seq_id][atom] = shift

pyprocs15 = PyProCS15('1UBQ_amber.pdb')
targets, shieldings = pyprocs15.calc_shieldings()

ca_procs = []
cb_procs = []
co_procs = []
n_procs = []
hn_procs = []
ha_procs = []

ca_bmrb = []
cb_bmrb = []
co_bmrb = []
n_bmrb = []
hn_bmrb = []
ha_bmrb = []

for target, shielding in zip(targets, shieldings):

    if target[0] < 1.0E-6:
        continue

    resid = target[2]

    ca_procs.append(target[0])
    cb_procs.append(target[1])
    co_procs.append(target[2])
    n_procs.append(target[3])
    hn_procs.append(target[4])
    ha_procs.append(target[5])
    
    ca_bmrb.append(bmrb_cs[resid]['CA'])
    cb_bmrb.append(bmrb_cs[resid]['CB'])
    co_bmrb.append(bmrb_cs[resid]['C'])
    n_bmrb.append(bmrb_cs[resid]['N'])
    hn_bmrb.append(bmrb_cs[resid]['HN'])
    if 'HA' in bmrb_cs[resid]:
        ha_bmrb.append(bmrb_cs[resid]['HA'])
    else:
        ha_bmrb.append(bmrb_cs[resid]['HA2'])

