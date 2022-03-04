#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import os
import pandas as pd
from lib.Utility import *
from Bio.PDB.DSSP import make_dssp_dict

# ------------------------------------------------
# get secondary structure from dssp result
# ------------------------------------------------
def dssp_run(uids, files, tmpDir, nresume):
    dic = {'H': 1, 'G': 1, 'I': 1, 'B': 2, 'E': 2, 'T': 0, 'S': 0, '-': 0}
    rows = list()
    destDir = tmpDir + '/dssp_result'
    make_dir(destDir,nresume)
    for uid, file in zip(uids, files):
        dsspFile = destDir + '/' + os.path.basename(file).replace('.pdb', '.out')
        if not os.path.isfile(dsspFile):
            run_cmd('mkdssp -i '+file+' -o '+dsspFile)
        else:
            info(f'{file} DSSP result already exists.')
        out = make_dssp_dict(dsspFile)
        dssp = list(out)[0]
        out = [dssp[i][1] for i in dssp.keys()]
        binary = list(pd.Series(out).map(dic))
        structures = ''.join(map(str, binary))
        helix = structures.count('1')
        sheet = structures.count('2')
        helix_percentage = round(helix/len(structures), 3)
        sheet_percentage = round(sheet/len(structures), 3)
        loop_percentage = round(1 - helix_percentage - sheet_percentage, 3)

        rows.append([uid, helix_percentage, sheet_percentage,
                     loop_percentage, structures])

    df = pd.DataFrame(rows, columns=['uniprot_id', 'helix_percentage', 'sheet_percentage',
                                     'loop_percentage', 'secondary_structure_binary'])
    df = df.set_index('uniprot_id')
    df.to_pickle(destDir + '/dssp_result.pkl')
    df.to_csv(destDir + '/dssp_result.csv')
    info('DSSP finished!')
    return df
