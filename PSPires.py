#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

"""
Description: PSPires is a machine learning model based on integrated residue-level and structure-level features to predict phase-separating proteins.

Usage: -u/-f/-p/-d are required and you can only specify one of them.
1. ${SOFTWAREPATH}/PSPires.py -u P09651
2. ${SOFTWAREPATH}/PSPires.py -u P09651 O00444
3. ${SOFTWAREPATH}/PSPires.py -f ${SOFTWAREPATH}/demo/PDB_files_list.txt  #please change the PDB file path before running
4. ${SOFTWAREPATH}/PSPires.py -f ${SOFTWAREPATH}/demo/uniprotID_list.txt
5. ${SOFTWAREPATH}/PSPires.py -p ${SOFTWAREPATH}/demo/AF-A0A2R8QUZ1-F1-model_v2.pdb
6. ${SOFTWAREPATH}/PSPires.py -d ${SOFTWAREPATH}/demo
"""

import logging
import argparse
import os
import sys
import json
import re
import requests

import pandas as pd
from Bio import PDB
from Bio.SeqUtils import seq1

from lib.Utility import *
from lib.get_pdb_info import *
from lib.dssp_run import dssp_run
from lib.protDCal_run import protDCal_run
from lib.psaia_run import psaia_run
from lib.calculate_features import *
from lib.predict import *

# ------------------------------------
# parse arguments
# ------------------------------------
description = 'PSPires can predict phase-separating probability of proteins based on integrated residue-level and structure-level features.'
parser = argparse.ArgumentParser(description=description)
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-u', '--uniprot', type=str, nargs='+',
                   help='UniProt IDs. Multiple IDs should be separated by space.')
group.add_argument('-f', '--file', type=argparse.FileType('r'),
                   help="List file with UniProt IDs or absolute path of protein pdb files. Each ID or pdb file name should take one line.")
group.add_argument('-p', '--pdbfile', type=str,
                   help="PDB file of a protein.")
group.add_argument('-d', '--directory', type=str, default=os.getcwd(),
                   help="Absolute directory path of pdb files. The script will automatically search files with pdb suffix under the specified directory.")
parser.add_argument('-o', '--output', default=sys.stdout,
                    help="Output file name. If not specified, result would be sent to standard out.")
parser.add_argument('-n', '--name', type=str, default='PSPires',
                    help="Project name. PSPires would use this name to create temporary file directory. If not specified, the name would be PSPires_tmpDir.")
parser.add_argument('--dont_resume', dest="nresume", action="store_true",
                    help='By default, each re-run would use previous temporary files to resume from the step it crashed. When dont_resume is on, PSPires would clean up the temporary files and start from the beginning.')
parser.add_argument('--dont_remove', dest="nremove", action="store_true",
                    help='By default, PSPires would clean up temporary files. When dont_remove is on, PSPires would keep temporary files.')
parser.add_argument('--complete', action="store_true",
                    help='By default, PSPires would not include the features calculated by ProtDCal software as the calculation is time-consuming and contribute little to the model performance. When complex is on, PSPires would calculate ProtDCal features and it may take much longer time.')
parser.add_argument('-t', '--threshold', default=70, type=int,
                    help='Threshold of pLDDT score to get idr regions. The default value is 70.')
parser.add_argument('-c','--cutoff', default=34, type=int,
                    help='If the RSA percentage of a residue is greater than this cutoff, it will be assigned as exposed surface residue, otherwise as buried residue. The default value is 34.')
parser.add_argument('-r','--random', default=42, type=int,
                    help='Random seed. The model with random seed of 42 has been pre-trained. By default, PSPires would use this pre-trained model. If you specify other random seed, the calculation would take longer to train new model. The default value is 42.')
parser.add_argument('-j', '--jobs', default=10, type=int,
                    help='If random seed is set, PSPires would use the given number of jobs to train the Random Forest Classifier model. And if complete mode is on, PSPires would launch the given number of jobs to run ProtDCal. The default value is 10.')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)-5s @ %(asctime)s: %(message)s ',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filemode="w",
                    filename=args.name + '_running.log')

# ------------------------------------
# global parameters
# ------------------------------------
PWDIR = os.getcwd()
PACKAGEDIR = sys.path[0]
tmpDir = PWDIR + '/' + args.name
make_dir(tmpDir,args.nresume)


# ------------------------------------
# check input arguments
# ------------------------------------
def download_pdb_file(uid, dir):
    '''Download pdb file from AlphaFold database to a new folder'''
    filename = 'AF-' + uid + '-F1-model_v2.pdb'
    r = requests.get(f'https://alphafold.ebi.ac.uk/files/{filename}')
    if r.status_code != 200:
        raise_error(f'Could not find pdb file for {uid} in AlphaFold database.')
    content = r.content.decode('utf-8')
    with open(dir + '/' + filename, 'w') as f:
        f.write(content)


def check_pdb_format(file):
    '''check whether the input file is in valid pdb format'''
    if not os.path.isfile(file):
        return False
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('_', file)
    for model in structure.get_list():
        for chain in model.get_list():
            for residue in chain.get_list():
                if residue.has_id("CA"):
                    return True
                else:
                    return False


def get_lists():
    uniprot_ids = list()
    inputfiles = list()
    if args.uniprot:
        uniprot_ids = args.uniprot
    elif args.file:
        lists = [i.strip() for i in args.file.readlines()]
        if re.match(r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}', lists[0]):
            uniprot_ids.append(lists[0])
            if len(lists) > 1:
                for i in lists[1:]:
                    if re.match(r'[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}', i):
                        uniprot_ids.append(i)
                    else:
                        warn(f'{i} is not a valid UniProt ID. Skip this ID.')
        elif check_pdb_format(lists[0]):
            inputfiles.append(lists[0])
            if len(lists) > 1:
                for i in lists[1:]:
                    if check_pdb_format(i):
                        inputfiles.append(i)
                    else:
                        warn(f'{i} is not in valid pdb format. Skip this file.')
        else:
            raise_error(f'{args.file} is not valid. List file should contain UniProt IDs or absolute path of protein pdb files. Each ID or pdb file name should take one line.')
    elif args.pdbfile:
        if check_pdb_format(args.pdbfile):
            inputfiles = [args.pdbfile]
        else:
            raise_error(f'{args.pdbfile} is not in valid pdb format.')
    else:
        out = run_cmd('ls '+os.path.join(args.directory, '*pdb'), 'result')
        # tmpfiles = [i.strip() for i in os.popen('ls '+os.path.join(args.directory, '*pdb')).readlines()]
        if len(out) == 0:
            raise_error(f'There is no pdb file under the specified directory: {args.directory}')
        else:
            tmpfiles = out.split('\n')
            for i in tmpfiles:
                if check_pdb_format(i):
                    inputfiles.append(i)
                else:
                    warn(f'{i} is not in valid pdb format. Skip this file.')

    # download or link pdb files
    files = list()
    dirName = tmpDir + '/pdbFiles'
    make_dir(dirName,args.nresume)
    if inputfiles:
        for i in inputfiles:
            fileName = dirName + '/' + os.path.basename(i)
            if not os.path.isfile(fileName):
                run_cmd('ln '+i+' '+dirName)
            files.append(fileName)

    if uniprot_ids:
        for uid in uniprot_ids:
            fileName = dirName + '/AF-'+uid+'-F1-model_v2.pdb'
            if not os.path.isfile(fileName):
                download_pdb_file(uid, dirName)
            elif not check_pdb_format(fileName):
                warn(f'{fileName} already exists but is not a valid pdb file. Re-download the file.')
                download_pdb_file(uid, dirName)
            else:
                info(f'{fileName} already exists.')
            files.append(fileName)

    if type(args.output) == str:
        outputFile = open(args.output, 'w')
    else:
        outputFile = args.output
    uids = [os.path.basename(i).replace('AF-', '').replace('-F1-model_v2.pdb', '') for i in files]

    return uids, files, outputFile


# ------------------------------------
# main functions
# ------------------------------------
def main():
    uids, files, outputFile = get_lists()
    
    if os.path.isfile(tmpDir + '/final_features.pkl'):
        features = pd.read_pickle(tmpDir + '/final_features.pkl')
    else:
        psaia_job = MyThread(psaia_run, args=(files, tmpDir, PACKAGEDIR, args.nresume))
        psaia_job.start()
        info('PSAIA start running')

        if args.complete:
            destDir = tmpDir + '/protDCal_result'
            aaFile = destDir + '/df_aa.pkl'
            protFile = destDir + '/df_prot.pkl'
            if os.path.isfile(aaFile) and os.path.isfile(protFile):
                df_aa = pd.read_pickle(aaFile)
                df_prot = pd.read_pickle(protFile)
            else:
                protDCal_job = MyThread(protDCal_run, args=(files, tmpDir, PACKAGEDIR, args.nresume, args.jobs))
                protDCal_job.start()
                info('ProtDCal start running.')

        dsspFile = tmpDir + '/dssp_result/dssp_result.pkl'
        if os.path.isfile(dsspFile):
            dssp = pd.read_pickle(dsspFile)
        else:
            info('DSSP start running.')
            dssp = dssp_run(uids, files, tmpDir, args.nresume)

        pdbFile = tmpDir + '/pdbInfo.pkl'
        if os.path.isfile(pdbFile):
            pdbInfo = pd.read_pickle(pdbFile)
        else:
            info('PDB info start running.')
            pdbInfo = get_pdb_info(dssp, uids, files, args.threshold, tmpDir)

        if args.complete and not (os.path.isfile(aaFile) and os.path.isfile(protFile)):
            protDCal_job.join()
            df_aa, df_prot = protDCal_job.get_result()
        else:
            df_aa, df_prot = pd.DataFrame(), pd.DataFrame()

        psaia_job.join()

        features = calculate_features(tmpDir, args.cutoff, files, dssp, pdbInfo, df_aa, df_prot)

    if df_aa.empty or df_prot.empty:
        trainDataFile = PACKAGEDIR + '/data/RF_model_training_data1.bz2'
        modelFile = PACKAGEDIR + '/data/models1.sav'
    else:
        trainDataFile = PACKAGEDIR + '/data/RF_model_training_data2.bz2'
        modelFile = PACKAGEDIR + '/data/models2.sav'

    predict(outputFile, tmpDir, features, args.random, modelFile, trainDataFile, args.jobs)
    
    if not args.nremove:
        run_cmd('rm -rf '+tmpDir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        error("User interrupt me ^_^ \n")
        sys.exit(1)
