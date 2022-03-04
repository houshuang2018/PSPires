#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import os
from lib.Utility import *

# ------------------------------------
# run PSAIA to calculate features
# ------------------------------------
def psaia_run(files, tmpDir, PACKAGEDIR, nresume):
    destDir = tmpDir + '/PSAIA_result'
    make_dir(destDir, nresume)
    newFiles = list()
    for file in files:
        pdbName = os.path.basename(file).replace('.pdb', '')
        out = run_cmd('ls '+destDir + '/' + pdbName+'*tbl', 'result')
        if len(out) == 0:
            newFiles.append(file)
        else:
            info(f'{file} PSAIA result already exists.')

    if newFiles:
        listFile = destDir + '/files.list'
        with open(listFile, 'w') as ofile:
            for i in newFiles:
                ofile.write(i+'\n')

        dataDir = PACKAGEDIR+'/software/PSAIA/'
        configFile = destDir + '/psaia.cfg'
        with open(configFile, 'w') as ofile:
            content = 'analyze_bound: 0\nanalyze_unbound: 1\ncalc_asa: 1\nwrite_asa: 1\ncalc_rasa: 1\n'
            content += 'standard_asa: ' + dataDir + 'natural_asa.asa\n'
            content += 'calc_dpx: 1\ncalc_cx: 1\ncalc_hydro: 1\n'
            content += 'hydro_file: ' + dataDir + 'hydrophobicity.hpb\n'
            content += 'radii_filename:' + dataDir + 'chothia.radii\n'
            content += 'write_xml: 1\nwrite_table: 1\n'
            content += 'output_dir: ' + destDir + '\n'
            ofile.write(content)

        simgFile = PACKAGEDIR + '/data/psaia.simg'
        psaExe = PACKAGEDIR+'/software/PSAIA/psa'
        run_cmd('chmod u+x '+psaExe)
        cmd = 'echo -e "y\n" | singularity exec ' + simgFile + \
            ' bash -c "' + psaExe + ' '+configFile + ' ' + listFile + '"'
        run_cmd(cmd, 'result')
    else:
        info('All PSAIA results already exist.')
    info('PSAIA finished!')
    
