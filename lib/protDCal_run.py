#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

import os
import pandas as pd
from lib.Utility import *
from math import ceil

# ------------------------------------
# run ProtDCal to calculate features
# ------------------------------------
def protDCal_run(files, tmpDir, PACKAGEDIR, nresume, jobs):
    protCol = ['DGfold','DGwat','DGconf','DGscr','DGHBd']
    destDir = tmpDir + '/protDCal_result'
    make_dir(destDir, nresume)
        
    protDCal_dir = PACKAGEDIR + '/software/ProtDCal_v4.5'
    os.chdir(protDCal_dir)
    
    ## get old results
    aaOld = destDir + '/aa_old.txt'
    protOld = destDir + '/prot_old.txt'
    
    if len(files) < jobs:
        jobs = len(files)
    
    for i in range(jobs):
        partname = destDir + '/part'+str(i)
        aatmp = partname + '/part' + str(i) + '_AA.txt'
        protmp = partname + '/part' + str(i) + '_Prot.txt'
        if os.path.isfile(aatmp) and os.path.isfile(protmp):
            if i == 0:
                run_cmd('cp '+aatmp+' '+aaOld)
                run_cmd('cp '+protmp+' '+protOld)
            else:
                run_cmd("sed -n '4,$'p " + aatmp + ">> " + aaOld)
                run_cmd("sed -n '2,$'p " + protmp + ">> " + protOld)

    ## prepare pdb files to calculate
    oldList = list()
    if os.path.isfile(protOld) and os.path.isfile(aaOld):
        run_cmd("sed -i 's/,//g' " + protOld)
        run_cmd("sed -i 's/,//g' " + aaOld)
        df_prot_old = pd.read_csv(protOld, sep='\t', index_col=0, low_memory=False)
        df_aa_old = pd.read_csv(aaOld, skiprows=2, sep='\t', index_col=0, low_memory=False)
        if not df_prot_old.empty:
            split = pd.Series(df_prot_old.index).str.split('_A_', expand=True)
            oldList = list(split.iloc[:, 0])

    allList = [os.path.basename(i) for i in files]
    pdbRootDir = destDir + '/protDCal_pdbFiles'
    aaFile = destDir + '/protDCal_AA.txt'
    protFile = destDir + '/protDCal_Prot.txt'
    if set(oldList) != set(allList):
        if os.path.isdir(pdbRootDir):
            run_cmd('rm -rf ' + pdbRootDir)
        run_cmd('mkdir '+pdbRootDir)
        allnum = len(files)
        fnum = ceil(allnum/jobs)
        template = PACKAGEDIR + '/software/ProtDCal_v4.5/example.proj'
        objs = list()
        for num in range(jobs):
            partname = pdbRootDir + '/part'+str(num)
            run_cmd('mkdir '+partname)
            right = (num+1)*fnum if (num+1)*fnum < allnum else allnum
            for i in range(num*fnum, right):
                if allList[i] not in oldList:
                    run_cmd('ln ' + tmpDir + '/pdbFiles/' + allList[i] + ' ' + partname)
            if os.listdir(partname):
                projName = destDir + '/part'+str(num)+'.proj'
                run_cmd('cp '+template+' '+projName)
                run_cmd("sed -i '2d' " + projName)
                run_cmd("sed -i '1a " + partname + "' " + projName)

                # run
                cmd = 'java -jar ProtDCal.jar' + ' -p ' + projName + ' -o ' + destDir
                obj = MyThread(run_cmd, args=(cmd,))
                objs.append(obj)
                obj.start()
        
        if len(objs) == 0:
            raise_error('Old ProtDCal result is not complete, but no job is running. Please check.')
        for i in objs:
            i.join()
        
        for i in range(jobs):
            partname = destDir + '/part'+str(i)
            aatmp = partname + '/part' + str(i) + '_AA.txt'
            protmp = partname + '/part' + str(i) + '_Prot.txt'
            if os.path.isfile(aatmp) and os.path.isfile(protmp):
                if i == 0:
                    run_cmd('cp '+aatmp+' '+aaFile)
                    run_cmd('cp '+protmp+' '+protFile)
                else:
                    run_cmd("sed -n '4,$'p " + aatmp + ">> " + aaFile)
                    run_cmd("sed -n '2,$'p " + protmp + ">> " + protFile)
        run_cmd("sed -i 's/,//g' " + aaFile)
        run_cmd("sed -i 's/,//g' " + protFile)
        df_aa = pd.read_csv(aaFile, skiprows=2, sep='\t', index_col=0, low_memory=False)
        df_prot = pd.read_csv(protFile, sep='\t', index_col=0, low_memory=False)
        if len(oldList) != 0:
            info('Merge old ProtDCal result.')
            df_aa = df_aa_old.append(df_aa, sort=False)
            df_prot = df_prot_old.append(df_prot, sort=False)
    else:
        os.rename(aaOld, aaFile)
        os.rename(protOld, protFile)
        df_aa = df_aa_old
        df_prot = df_prot_old

    split = pd.Series(df_aa.index).str.split('_A_', expand=True)
    df_aa.insert(0, 'uniprot_id', split.iloc[:, 0].str.replace('AF-', '', regex=False).str.replace('-F1-model_v2.pdb', '', regex=False).values)
    df_aa.insert(1, 'aa', split.iloc[:, 1].str.split('_', expand=True).iloc[:, 0].values)
    df_aa = df_aa.reset_index(drop=True)

    df_prot = df_prot[protCol]
    df_prot.insert(0, 'uniprot_id', df_prot.index.str.replace('AF-', '', regex=False).str.replace('-F1-model_v2.pdb', '', regex=False).values)
    df_prot = df_prot.reset_index(drop=True)
    
    df_aa.to_pickle(destDir + '/df_aa.pkl')
    df_prot.to_pickle(destDir + '/df_prot.pkl')

    info('ProtDCal finished!')
    return df_aa, df_prot

