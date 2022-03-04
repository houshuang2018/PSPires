# PSPires

PSPires is a machine learning model based on integrated residue-level and structure-level features to predict phase-separating proteins. It is written in Python and is available as a command line tool.

# Requirements

PSPires requires Java (version >=7.x) and Python (version 3.x.x).

# Setup

1. Install [Singularity](https://apptainer.org/admin-docs/master/installation.html#).
   
   + PSPires requires Singularity to run. As Singularity is written primarily in Go, you should install [Go](https://go.dev/doc/install) first. 
   
   + After installation, you can type the command below to check Singularity has been installed successfully.
     
     ```shell
     $ singularity -h
     ```
     
     > Note: you need to use the following command to source the Singularity bash completion file to make sure the usage of bash completion in new shells.
     > 
     > ```shell
     > $ echo ". Singularity_Installation_Path/etc/bash_completion.d/singularity" >> ~/.bashrc  # you should replce Singularity_Installation_Path with your installation path
     > ```

2. Install [DSSP](https://github.com/PDB-REDO/dssp). Conda installation is recommended. 
   
   ```
   conda install -c salilab dssp
   ```

3. Install the following python package:
   
   ```
   pip install biopython scikit-learn numpy pandas requests
   ```

4. Download required data and put them under the data folder: 
   
   + Qt 4.8.6 libraray container image needer for PSAIA software running: [psaia.simg](https://compbio-zhanglab.org/release/psaia.simg)

# Parameters

Required parameters: the following parameters are required and you can only specify one of them.

```python
-u UNIPROT [UNIPROT ...], --uniprot UNIPROT [UNIPROT ...]
                      UniProt IDs. Multiple IDs should be separated by space.
-f FILE, --file FILE  List file with UniProt IDs or absolute path of protein pdb files.
                        Each ID or pdb file name should take one line.
-p PDBFILE, --pdbfile PDBFILE
                      PDB file of a protein.
-d DIRECTORY, --directory DIRECTORY
                      Absolute directory path of pdb files. The script will automatically
                        search files with pdb suffix under the specified directory.
```

Optional parameters:

```python
-o OUTPUT, --output OUTPUT
                      Output file name. If not specified, result would be sent to standard
                        out.
-n NAME, --name NAME  Project name. PSPires would use this name to create temporary file
                        directory. If not specified, the name would be PSPires_tmpDir.
-t THRESHOLD, --threshold THRESHOLD
                      Threshold of pLDDT score to get idr regions. The default value is 70.
-c CUTOFF, --cutoff CUTOFF
                      If the RSA percentage of a residue is greater than this cutoff, it
                        will be assigned as exposed surface residue, otherwise as buried
                        residue. The default value is 34.
-r RANDOM, --random RANDOM
                      Random seed. The model with random seed of 42 has been pre-trained.
                        By default, PSPires would use this pre-trained model. If you specify
                        other random seed, the calculation would take longer to train new
                        model. The default value is 42.
-j JOBS, --jobs JOBS  If random seed is set, PSPires would use the given number of jobs to
                        train the Random Forest Classifier model. And if complete mode is on,
                        PSPires would launch the given number of jobs to run ProtDCal. The
                        default value is 10.
--dont_resume         By default, each re-run would use previous temporary files to resume
                        from the step it crashed. When dont_resume is on, PSPires would clean
                        up the temporary files and start from the beginning.
--dont_remove         By default, PSPires would clean up temporary files. When dont_remove
                        is on, PSPires would keep temporary files.
--complete            By default, PSPires would not include the features calculated by
                        ProtDCal software as the calculation is time-consuming and contribute
                        little to the model performance. When complex is on, PSPires would
                        calculate ProtDCal features and it may take much longer time.
```

## Usage

1. Specify uniprot ids:
   
   ```shell
   $ ${SOFTWAREPATH}/PSPires.py -u P09651
   $ ${SOFTWAREPATH}/PSPires.py -u P09651 O00444
   ```

2. Specify list file with UniProt IDs or absolute path of protein pdb files:
   
   ```shell
   $ ${SOFTWAREPATH}/PSPires.py -f ${SOFTWAREPATH}/demo/PDB_files_list.txt
   $ ${SOFTWAREPATH}/PSPires.py -f ${SOFTWAREPATH}/demo/uniprotID_list.txt
   ```

3. Specify PDB file of a protein:
   
   ```shell
   ${SOFTWAREPATH}/PSPires.py -p ${SOFTWAREPATH}/demo/AF-A0A2R8QUZ1-F1-model_v2.pdb
   ```

4. Specify absolute directory path of pdb files:
   
   ```shell
   ${SOFTWAREPATH}/PSPires.py -d ${SOFTWAREPATH}/demo
   ```
