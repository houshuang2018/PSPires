#!/usr/bin/env python
# Date: 2022-02-25
# Author: Shuang Hou
# Contact: houshuang@tongji.edu.cn

"""
Description: calculate final features
"""

import logging
import argparse
import os
import sys
import json
import re
import requests
import itertools

import pandas as pd
import numpy as np

from lib.Utility import *


# ------------------------------------
# static parameters
# ------------------------------------
RESIDUES = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

residue_key = {'A':'ALA', 'C':'CYS', 'D':'ASP', 'E':'GLU', 
               'F':'PHE', 'G':'GLY', 'H':'HIS', 'I':'ILE', 
               'K':'LYS', 'L':'LEU', 'M':'MET', 'N':'ASN', 
               'P':'PRO', 'Q':'GLN', 'R':'ARG', 'S':'SER', 
               'T':'THR', 'V':'VAL', 'W':'TRP', 'Y':'TYR'}

residue_key2 = {residue_key[i]:i for i in residue_key.keys()}

psaia_standard_asa = {'A':107.24,'R':233.01,'N':150.85,'D':144.06,'C':131.46,
                      'Q':177.99,'E':171.53,'G':80.54,'H':180.93,'I':173.40,
                      'L':177.87,'K':196.14,'M':186.80,'F':200.93,'P':133.78,
                      'S':115.30,'T':136.59,'W':240.12,'Y':213.21,'V':149.34}

# dssp_standard_asa = {"A": 106.0, "R": 248.0, "N": 157.0, "D": 163.0,
#                      "C": 135.0, "Q": 198.0, "E": 194.0, "G": 84.0,
#                      "H": 184.0, "I": 169.0, "L": 164.0, "K": 205.0,
#                      "M": 188.0, "F": 197.0, "P": 136.0, "S": 130.0,
#                      "T": 142.0, "W": 227.0, "Y": 222.0, "V": 142.0}

# Kyte & Doolittle {kd} index of hydrophobicity, 1982
HP = {'A': 1.8, 'R':-4.5, 'N':-3.5, 'D':-3.5, 'C': 2.5,
      'Q':-3.5, 'E':-3.5, 'G':-0.4, 'H':-3.2, 'I': 4.5,
      'L': 3.8, 'K':-3.9, 'M': 1.9, 'F': 2.8, 'P':-1.6,
      'S':-0.8, 'T':-0.7, 'W':-0.9, 'Y':-1.3, 'V': 4.2, 'U': 0.0}

# AAindex: CHAM820101
Polarizability={'A':0.046, 'L':0.186, 'R':0.291, 'K':0.219, 
                'N':0.134, 'M':0.221, 'D':0.105, 'F':0.290,
                'C':0.128, 'P':0.131, 'Q':0.180, 'S':0.062,
                'E':0.151, 'T':0.108, 'G':0.000, 'W':0.409,
                'H':0.230, 'Y':0.298, 'I':0.186, 'V':0.140}

# AAindex: GRAR740102
Polarity = {'A':8.1, 'L':4.9, 'R':10.5, 'K':11.3,
            'N':11.6, 'M':5.7, 'D':13.0, 'F':5.2,
            'C':5.5, 'P':8.0, 'Q':10.5, 'S':9.2,
            'E':12.3, 'T':8.6, 'G':9.0, 'W':5.4,
            'H':10.4, 'Y':6.2, 'I':5.2, 'V':5.9}

# Hellberg et al. Journal of medicinal chemistry, 1987
Z1 = {'A':0.07, 'V':-2.69, 'L':-4.19, 'I':-4.44,
      'P':-1.22, 'F':-4.92, 'W':-4.75, 'M':-2.49,
      'K':2.84, 'R':2.88, 'H':2.41, 'G':2.23,
      'S':1.96, 'T':0.92, 'C':0.71, 'Y':-1.39,
      'N':3.22, 'Q':2.18, 'D':3.64, 'E':3.08}
Z2 = {'A':-1.73, 'V':-2.53, 'L':-1.03, 'I':-1.68,
      'P':0.88, 'F':1.30, 'W':3.65, 'M':-0.27,
      'K':1.41, 'R':2.52, 'H':1.74, 'G':-5.36,
      'S':-1.63, 'T':-2.09, 'C':-0.97, 'Y':2.32,
      'N':1.45, 'Q':0.53, 'D':1.13, 'E':0.39}
Z3 = {'A':0.09, 'V':-1.29, 'L':-0.98, 'I':-1.03,
      'P':2.23, 'F':0.45, 'W':0.85, 'M':-0.41,
      'K':-3.14, 'R':-3.44, 'H':1.11, 'G':0.30,
      'S':0.57, 'T':-1.40, 'C':4.13, 'Y':0.01,
      'N':0.84, 'Q':-1.14, 'D':2.36, 'E':-0.07}

# Collantes et al. Journal of medicinal chemistry, 1995
ISA = {'G':19.93, 'A':62.90, 'V':120.91, 'L':154.35,
       'I':149.77, 'F':189.42, 'Y':132.16, 'W':179.16,
       'P':122.35, 'D':18.46, 'N':17.87, 'E':30.19,
       'Q':19.53, 'S':19.75, 'T':59.44, 'C':78.51, 
       'M':132.22, 'K':102.78, 'R':52.98, 'H':87.38}

ECI = {'G':0.02, 'A':0.05, 'V':0.07, 'L':0.10,
       'I':0.09, 'F':0.14, 'Y':0.72, 'W':1.08,
       'P':0.16, 'D':1.25, 'N':1.31, 'E':1.31,
       'Q':1.36, 'S':0.56, 'T':0.65, 'C':0.15, 
       'M':0.34, 'K':0.53, 'R':1.69, 'H':0.56}

aaCol = ['DGc(F)_NO','Gw(F)_NO','W(F)_NO','DGs_NO','HBd_NO','DGel_NO','DGw_NO','DGLJ_NO','DGtor_NO',
         'lnFD_NO','Phi_NO','Psi_NO','wR2(HP)_NO','wDHBd(HP)_NO','wNc(HP)_NO','wNLC(HP)_NO']
protCol = ['DGfold','DGwat','DGconf','DGscr','DGHBd']

list1 = ['HP','ECI','Z1','Z2','Z3','ISA','Polarity','Polarizability']
list2 = [HP,ECI,Z1,Z2,Z3,ISA,Polarity,Polarizability]
list_weight = ['HP','ECI','Polarity','Polarizability']
list_psaia = ['DPX','s_chain_DPX','max_DPX','CX','s_chain_CX','max_CX','min_CX']
list_psaia_weight = ['DPX','CX']

asa_com = ['rsa','b_bone_rsa','s_chain_rsa','polar_rsa']
asa_com1 = ['rsa','b_bone_rsa','s_chain_rsa','polar_rsa','asa','b_bone_asa','s_chain_asa','polar_asa']
asa_com3 = ['rsa','polar_rsa','asa','polar_asa']
asa_com4 = ['asa','b_bone_asa','s_chain_asa','polar_asa']

psaia_col = ['chain_id','ch_total_ASA','ch_b_bone_ASA','ch_s_chain_ASA','ch_polar_ASA','ch_n_polar_ASA',
            'res_id','res_name','total_ASA','b_bone_ASA','s_chain_ASA','polar_ASA','n_polar_ASA',
            'total_RASA','b_bone_RASA','s_chain_RASA','polar_RASA','n_polar_RASA','average_DPX','s_avg_DPX',
            's_ch_avg_DPX','s_ch_s_avg_DPX','max_DPX','min_DPX','average_CX','s_avg_CX','s_ch_avg_CX',
            's_ch_s_avg_CX','max_CX','min_CX','Hydrophobicity']
psaia_col_keep = ['res_name',
                  'total_ASA','b_bone_ASA','s_chain_ASA','polar_ASA',
                  'total_RASA','b_bone_RASA','s_chain_RASA','polar_RASA',
                  'average_DPX','s_ch_avg_DPX','max_DPX','min_DPX',
                  'average_CX','s_ch_avg_CX','max_CX','min_CX']

group_percentage_col = ['Asx','Glx','Xle','Pos_charge','Neg_charge','Small','Hydrophilic','Aromatic','Aliphatic','Uncharge','Hydrophobic',
                         'Polar','NonPolar','UnFolding','Polarizability','Polarity','helix','sheet','turn','sp2','fraction_C','fraction_L']

## columns
columns = ['uniprot_id']

columns.extend(['sup_percent','ssup_percent'])

for col in list1+list_psaia:
    columns.append(col+'_'+'ssup')
        
for col2 in ['rsa','b_bone_rsa','s_chain_rsa','polar_rsa']:
    for col in list_weight+list_psaia_weight+['UN']:
        columns.append(col2+'('+col+')_'+'sup')

for col in group_percentage_col:
    columns.append('group_'+col+'_'+'sup')

for col in ['lcs_lowest_complexity', 'lcs_max_complexity', 'lcs_scores', 'lcs_fractions']:
    columns.append(col+'_PRT')


# ------------------------------------
# functions
# ------------------------------------
def group_percentage(seq):
    df = dict()
    for res in RESIDUES:
        res_count =seq.count(res)
        seq_len = len(seq)
        df['fraction_'+res] = res_count/seq_len

    # 16 features:
    Asx = df['fraction_D'] + df['fraction_N']
    Glx =df['fraction_E'] + df['fraction_Q']
    Xle =df['fraction_I'] + df['fraction_L']
    Pos_charge=df['fraction_K'] + df['fraction_R'] + df['fraction_H']
    Neg_charge=df['fraction_D'] + df['fraction_E']
    Small=df['fraction_P'] + df['fraction_G'] + df['fraction_A'] + df['fraction_S']
    Hydrophilic=(df['fraction_S'] + df['fraction_T'] + df['fraction_H'] 
                                          + df['fraction_N'] + df['fraction_Q'] + df['fraction_E'] 
                                          +df['fraction_D'] + df['fraction_K'] + df['fraction_R'])

    ### newly added
    # Calculates the aromaticity value of a protein according to Lobry, 1994. the same as iFeature
    Aromatic=df['fraction_F'] + df['fraction_W'] + df['fraction_Y']

    #iFeature
    Aliphatic=(df['fraction_V'] + df['fraction_I'] + df['fraction_L']
                                        + df['fraction_M']+ df['fraction_G'] + df['fraction_A'])
    #iFeature,preference
    Uncharge =(df['fraction_N'] + df['fraction_C'] + df['fraction_Q']
                                        + df['fraction_S'] + df['fraction_T'] + df['fraction_P'])

    #protr: PRAM900101:signal and nascent peptides, preference
    Hydrophobic = (df['fraction_V'] + df['fraction_I'] + df['fraction_L']
                                            + df['fraction_F'] + df['fraction_W'] + df['fraction_C'] 
                                            + df['fraction_M'])
    # ProtDCal
    Polar =(df['fraction_R'] + df['fraction_N'] + df['fraction_D']
                                     + df['fraction_C']+ df['fraction_Q'] + df['fraction_E']
                                     + df['fraction_H'] + df['fraction_K']+ df['fraction_S']
                                     + df['fraction_T'] + df['fraction_Y'])
    NonPolar=(df['fraction_A'] + df['fraction_G'] + df['fraction_I']
                                        + df['fraction_L'] + df['fraction_M'] + df['fraction_F']
                                        + df['fraction_P'] + df['fraction_W']+ df['fraction_V'])
    UnFolding=df['fraction_G'] + df['fraction_P']

    #protr: iFeature有些重复，显然是错误，因此用protr
    #protr: CHAM820101：The structural dependence of amino acid hydrophobicity parameters
    Polarizability=(df['fraction_K'] + df['fraction_M'] + df['fraction_H']
                                             + df['fraction_F'] + df['fraction_R'] + df['fraction_Y']
                                             + df['fraction_W'])
    #protr: GRAR740102: Amino acid difference formula to help explain protein evolution
    Polarity=(df['fraction_H'] + df['fraction_Q'] + df['fraction_R']+ df['fraction_K'] + df['fraction_N'] + df['fraction_E']+ df['fraction_D'])

    helix = (df['fraction_A'] + df['fraction_C'] + df['fraction_Q'] 
                                    + df['fraction_E'] + df['fraction_H'] + df['fraction_L'] 
                                    + df['fraction_K'] + df['fraction_M'])
    sheet =(df['fraction_I'] + df['fraction_F'] + df['fraction_T'] 
                                + df['fraction_W'] + df['fraction_Y'] + df['fraction_V'])
    turn =(df['fraction_N'] + df['fraction_D'] + df['fraction_G'] 
                            + df['fraction_P'] + df['fraction_S'])
    sp2 =(df['fraction_Y'] + df['fraction_F'] + df['fraction_W']
        + df['fraction_H'] + df['fraction_N'] + df['fraction_D']
        + df['fraction_Q'] + df['fraction_E'] + df['fraction_R'])

    out = [Asx,Glx,Xle,Pos_charge,Neg_charge,Small,Hydrophilic,Aromatic,Aliphatic,
           Uncharge,Hydrophobic,Polar,NonPolar,UnFolding,Polarizability,Polarity,helix,sheet,turn,sp2,df['fraction_C'],df['fraction_L']]
    return out


def add_lowcomplexity_features(seq):
    n_window = 20
    cutoff = 7       
    n_halfwindow = int(n_window / 2)        
    
    sig = list()
    lc_bool = [False] * len(seq)
    for i in range(len(seq)):
        if i < n_halfwindow:
            peptide = seq[:n_window]        
        elif i+n_halfwindow > int(len(seq)):
            peptide = seq[-n_window:]        
        else:
            peptide = seq[i-n_halfwindow:i+n_halfwindow]       
        complexity = (len(set(peptide)))
        if complexity <= cutoff:
            for bool_index in (i-n_halfwindow, i+n_halfwindow):
                try:
                    lc_bool[bool_index] = True
                except IndexError:
                    pass
        sig.append(complexity)            

    lcs_lowest_complexity = min(sig)
    lcs_max_complexity = max(sig)
    lcs_scores = sum(lc_bool)
    lcs_fractions = np.mean(lc_bool)

    out = [lcs_lowest_complexity, lcs_max_complexity, lcs_scores, lcs_fractions]
    
    return out


def calculate_features(tmpDir, cutoff, files, dssp = pd.DataFrame(), pdbInfo = pd.DataFrame(), df_aa = pd.DataFrame(), df_prot = pd.DataFrame()):
    ### load data
    if dssp.empty:
        raise_error('Please specify dssp data.')

    if pdbInfo.empty:
        raise_error('Please specify pdb information data.')
        
    if not (df_aa.empty and df_prot.empty):
        for struc in ['s']:
            for col in aaCol:
                columns.append(col+'_'+struc)
        # columns.append(protCol)
    
    ### calculate features
    psaiaDir = tmpDir + '/PSAIA_result/'
    final_df = list()
    for file in files:
        pdbName = os.path.basename(file).replace('.pdb','')
        info('PDB file: ' + file)
        # psaia_file = os.popen('ls '+psaiaDir+pdbName+'*unbound.tbl').read().strip()
        psaia_file = run_cmd('ls '+psaiaDir+pdbName+'*unbound.tbl', 'result')
        if not psaia_file:
            raise_error(f'{file} has no PSAIA result.')
        info('PSAIA result file: ' + psaia_file)
        uid = os.path.basename(file).replace('AF-', '').replace('-F1-model_v2.pdb', '')
        info('Inferred UniProt ID: ' + uid)
        data = pd.read_csv(psaia_file,skiprows=8,sep='\s+',header=None)
        data.columns=psaia_col
        data = data[psaia_col_keep]
        
        data['standard_asa']=data['res_name'].map(residue_key2).map(psaia_standard_asa).values
        data['DPX'] = data['average_DPX'].values
        data['s_chain_DPX'] = data['s_ch_avg_DPX'].values
        data['max_DPX'] = data['max_DPX'].values
        data['min_DPX'] = data['min_DPX'].values
        
        data['CX'] = data['average_CX'].values
        data['s_chain_CX'] = data['s_ch_avg_CX'].values
        data['max_CX'] = data['max_CX'].values
        data['min_CX'] = data['min_CX'].values
        
        data['asa'] = data['total_ASA'].values
        data['rsa'] = (data['total_RASA']/100).values
        data['b_bone_asa'] = data['b_bone_ASA'].values
        data['b_bone_rsa'] = (data['b_bone_RASA']/100).values
        data['s_chain_asa'] = data['s_chain_ASA'].values
        data['s_chain_rsa'] = (data['s_chain_RASA']/100).values
        data['polar_asa'] = data['polar_ASA'].values
        data['polar_rsa'] = (data['polar_RASA']/100).values
        
        idr_binary_mobidb_10 = list(map(int,list(pdbInfo.loc[uid, 'idr_binary_mobidb_10'])))
        data['idr_binary_mobidb_10'] = idr_binary_mobidb_10
        data['non_idr_binary_mobidb_10'] = np.logical_not(data['idr_binary_mobidb_10']).astype('int')
        
        for num, i in enumerate(list1):
            data[i] = data['res_name'].map(residue_key2).map(list2[num]).values 
        
        i = cutoff
        j = 'mobidb_10'
        superficial = '_'.join(['superficial',str(i)])
        data[superficial] = np.where(data['rsa']>i/100,1,0)
        idr = 'idr_binary_'+str(j)
        non_idr = 'non_idr_binary_'+str(j)
        s = non_idr
        ssup = '_'.join(['ssup',str(i),str(j)])
        sup = '_'.join(['sup',str(i),str(j)])
        data[ssup] = data[superficial] & data[non_idr]
        data[sup] = data[ssup] | data[idr]
        
        tmp = [uid]
        sup_percent = round(np.mean(data[sup]),3)
        ssup_percent = round(np.mean(data[ssup]),3)
        tmp.extend([sup_percent,ssup_percent])
        
        # type1
        struc = ssup
        for col in list1+list_psaia:
            transData = data[col][data[struc]==1]
            result = np.mean(transData) if len(transData)!=0 else 0
            tmp.append(round(result,3))
        
        # type2
        struc = sup
        for col2 in asa_com:
            for col in list_weight+list_psaia_weight+['UN']:
                if col=='UN':
                    transData = data[col2][data[struc]==1]
                else:
                    transData = data[col2][data[struc]==1] * data[col][data[struc]==1]
                result = np.mean(transData) if len(transData)!=0 else 0
                tmp.append(round(result,3))
        
        # type3
        # col2 = 'standard_asa'
        # for col in list1+['UN']:
        #     if col=='UN':
        #         transData = data[col2][data[idr]==1]
        #     else:
        #         transData = data[col2][data[idr]==1] * data[col][data[idr]==1]
        #     result = np.mean(transData) if len(transData)!=0 else 0
        #     tmp.append(round(result,3))
            
        # type3
        struc = sup
        seq = ''.join(list(data['res_name'][data[struc]==1].map(residue_key2)))
        if len(seq)==0:
            result = [0]*len(group_percentage_col)
        else:
            result = group_percentage(seq)
        tmp.extend(result)
        
        # type4
        wholeseq = ''.join(list(data['res_name'].map(residue_key2)))
        tmp.extend(add_lowcomplexity_features(wholeseq))
        
        # type5
        if not (df_aa.empty and df_prot.empty):
            struc = s
            data1 = df_aa.loc[df_aa['uniprot_id']==uid,:]
            data1 = data1.reset_index(drop=True)
            data2 = data.loc[:,['res_name',struc]]
            data2 = data2.reset_index(drop=True)
            tmp_data = pd.concat([data1, data2],axis=1)
            # tmp_data.to_csv(tmpDir + '/tmp_data.csv')
            # tmp_data = pd.merge(df_aa.loc[df_aa['uniprot_id']==uid,:], data.loc[:,['uniprot_id',struc]], how='inner',left_on='aa',right_on='res_name')
            for col in aaCol:
                if tmp_data[col].dtypes == 'object':
                    transData = tmp_data[col][tmp_data[struc]==1].str.replace(',','',regex = False).astype('float')
                else:
                    transData = tmp_data[col][tmp_data[struc]==1]
                result = np.mean(transData) if len(transData)!=0 else 0
                tmp.append(round(result,3))
            # tmp.extend(df_prot.loc[uid, protCol].tolist())
        
        final_df.append(tmp)
    
    ## merge result
    final_df = pd.DataFrame(final_df, columns=columns)
    if not df_prot.empty:
        final_df = pd.merge(final_df, df_prot, how='inner',left_on='uniprot_id',right_on='uniprot_id')
    final_df = pd.merge(final_df, pdbInfo[['idr_length_mobidb_10','idr_percentage_mobidb_10','idr_length_mobidb_10','idr_percentage_mobidb_10']], 
                    how='inner',left_on='uniprot_id',right_index=True)
    final_df = pd.merge(final_df, dssp[['helix_percentage','sheet_percentage','loop_percentage']], 
                    how='inner',left_on='uniprot_id',right_index=True)
                
    final_df.to_csv(tmpDir + '/final_features.csv',index=False)
    final_df.to_pickle(tmpDir + '/final_features.pkl')

    return final_df

