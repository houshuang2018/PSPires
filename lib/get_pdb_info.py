#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import pandas as pd
from lib.Utility import *
from Bio import SeqIO
import itertools
import re
import numpy as np
from Bio import PDB

# ----------------------------------------
# get idr regions from AlphaFold PDB file
# ----------------------------------------
def get_seq_and_name(file):
    for record in SeqIO.parse(file, "pdb-seqres"):
        name = record.description.split(' ')[-1].split('_')[0]
    return str(record.seq), len(record.seq), name


def getIdr_pos(structure, threshold):
    '''position is 1-based'''
    idr = list()
    num = 0
    for model in structure.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                num += 1
                if residue.has_id("CA"):
                    ca = residue["CA"]
                    if ca.get_bfactor() < threshold:
                        idr.append(num)
    return idr


def intervals_extract(iterable):
    '''Convert list of sequential number into intervals'''
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable), lambda t: t[1] - t[0]):
        group = list(group)
        yield [group[0][1], group[-1][1]]


def matMorphology(seq, rmax=3):
    # One, two and three residues-long ID stretches flanked on both sides by one,
    # two and three residue-long ordered stretches are converted to order and vice- versa.

    # Disorder expansion
    seq = rmax*"D"+seq+rmax*"D"

    for r in range(1, rmax + 1):
        pattern = 'D'*r + 'S'*r + 'D'*r
        new_pattern = 'D'*r + 'D'*r + 'D'*r

        for i in range(0, r + 1):
            seq = seq.replace(pattern, new_pattern)

    # Disorder contraction
    seq = rmax*"S"+seq[rmax:-rmax]+rmax*"S"

    for r in range(1, rmax + 1):
        pattern = 'S'*r + 'D'*r + 'S'*r
        new_pattern = 'S'*r + 'S'*r + 'S'*r

        for i in range(0, r + 1):
            seq = seq.replace(pattern, new_pattern)

    return seq[rmax:-rmax]


def mobidb_run(seq, idr, length, cutoff=10):
    '''Modified algorithm of MobiDB_10-lite to get long disordered regions'''
    state = ['D' if i in idr else 'S' for i in range(1, length+1)]
    state = ''.join(state)

    consensus = matMorphology(state, 3)

    # Structured stretches of up to 10 consecutive residues are then converted to ID
    # if they are flanked by two disordered regions of at least 20 residues
    flag = True
    while flag:
        m = re.search("D{21,}S{1,10}D{21,}", consensus)
        if m:
            matchLength = m.end(0) - m.start(0)
            consensus = consensus[:m.start(
                0)] + "D" * matchLength + consensus[m.end(0):]
        else:
            flag = False

    position = [i for i in np.arange(1, length+1) if consensus[i-1] == 'D']

    idr_intervals = list(intervals_extract(position))
    idr_pos = list()
    idr_seq = list()
    for i, j in idr_intervals:
        if j-i >= cutoff-1:
            idr_seq.append(seq[(i-1):j])
            for pos in range(i, j+1):
                idr_pos.append(pos)

    idr_binary = np.zeros(length)
    idr_binary[list(np.array(idr_pos)-1)] = 1
    idr_binary = [str(int(i)) for i in idr_binary]

    return ''.join(idr_binary), ','.join(idr_seq)


def get_pdb_info(dssp, uids, files, threshold, tmpDir):
    rows = list()
    for uid, file in zip(uids, files):
        structure_binary = dssp.loc[uid, 'secondary_structure_binary'].replace('2', '1')
        structure_binary = np.array(list(map(int, list(structure_binary))))
        seq, length, name = get_seq_and_name(file)
        if len(structure_binary) != length:
            raise_error(f'SeqIO package can not get true number of amino acids from {file}.')
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure('_', file)
        idr = getIdr_pos(structure, threshold)
        idr_binary_mobidb_10, idr_seq_mobidb_10 = mobidb_run(seq, idr, length)
        idr_length_mobidb_10 = idr_binary_mobidb_10.count('1')
        idr_percentage_mobidb_10 = round(idr_length_mobidb_10/length, 3)

        rows.append([name, uid, seq, length, idr_length_mobidb_10,
                     idr_percentage_mobidb_10, idr_binary_mobidb_10, idr_seq_mobidb_10])

    df = pd.DataFrame(rows, columns=['protein_name', 'uniprot_id', 'sequence', 'length',
                                     'idr_length_mobidb_10', 'idr_percentage_mobidb_10', 'idr_binary_mobidb_10', 'idr_seq_mobidb_10'])
    df = df.set_index('uniprot_id')
    df.to_pickle(tmpDir + '/pdbInfo.pkl')
    df.to_csv(tmpDir + '/pdbInfo.csv')
    info('Get PDB info finished!')
    return df
