from time import sleep
import sys
import pandas as pd
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess before training.')
    parser.add_argument('inputPath', help='Address of Input files.(See FILES.MD)')
    parser.add_argument('outputPath', default='./', help='Address of output files.')
    args = parser.parse_args()

    infiles = args.inputPath
    directory = args.outputPath
    if directory[-1] == '/':
        directory = directory[:-1]
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        if not os.path.exists(infiles):
            print('[Error in Checking input directory] path: '+ infiles)
            os.exit()
    except OSError:
        print ('[Error in Checking output directory] path: ' +  directory)
        sys.exit()
    try:
        print('Loading EXP file...')
        EXP = pd.read_csv(infiles + '/EXP.csv')
    except:
        print('[Error in loading Expression] Check expression file is ready for extraction.')
        sys.exit()
    try:
        print('Loading PPI file...')
        PPI_INFO = pd.read_csv(infiles + '/PPI_INFO.txt', sep="\t", index_col = 0)
        PPI_LIST = pd.read_csv(infiles + '/PPI_LINK.txt', sep=' ')
    except:
        print('[Error in loading ppInteraction] Check ppinteraction file is ready for extraction.')
        sys.exit()
    try:
        print('Loading LIST file...')
        LIST = pd.read_csv(infiles + 'LIST.txt', sep='\t').columns.to_list()
        print('find', len(LIST), 'genes.')
    except:
        print('[Error in loading Genelist] Check genelist file is ready for extraction.')
        sys.exit()

    print('Generate preprocessed Expression data.')

    EXP_LIST = EXP[LIST]
    EXP1 = EXP[LIST]
    EXP2 = np.log(EXP / EXP.median())[LIST]
    EXP1.to_csv(directory + '/groupEXP.csv')
    EXP2.to_csv(directory + '/groupEXP_foldChange.csv')

    print('Start to struct protein-protein(gene-gene) interaction.')

    PPI = pd.DataFrame(0, index=EXP1.columns, columns=EXP1.columns)
    fullLength = len(PPI) * len(PPI) / 2

    check = []
    count = 0
    for index, value in PPI_INFO.iterrows():
        if value.preferred_name in PPI.columns:
            check.append(True)
        else:
            check.append(False)
    PPI_LEFT = PPI_INFO[check]

    for index, value in PPI_LIST.iterrows():
        if value.protein1 in PPI_LEFT.index and value.protein2 in PPI_LEFT.index:
            PPI.loc[PPI_LEFT.loc[value.protein1].preferred_name, PPI_LEFT.loc[value.protein2].preferred_name] = value.combined_score
            PPI.loc[PPI_LEFT.loc[value.protein2].preferred_name, PPI_LEFT.loc[value.protein1].preferred_name] = value.combined_score
            count += 1
            sys.stdout.write('\r')
            sys.stdout.write(directory + '] Interactions count: '+str(int(count / 2))+' / '+str(fullLength)+' (maximum interaction)')
            sys.stdout.flush()

    PPI.to_csv(directory + '/groupPPI.csv')
    print('End preprocess.')