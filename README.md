# P9-Code

# Information

## Models
- DE-TransE: transformation model
- DE-SimplE: tensor decompositional model
- DE-DistMult: tensor decompositional model
- ATiSE: tensor decompositional model
- TERO: transformation model
- TFLEX: neural network model

## Metrics
- MRR
- HITS@1
- HITS@3
- HITS@10


# Data Prepare
In this project, we use JSON as the file to transfer the data. 

The script is in `~/github/P9-Code/dataprepare/corruptedquadruple/corrupted_quadruple_generate.py`

## Usage on Terminal
1. Move the directory to `~/github/P9-Code/dataprepare/corruptedquadruple`
2. Run the script
```bash
python corrupted_quadruple_generate.py
```

## Usage on CLAAUDIA
1. Login CLAAUDIA
```bash
ssh -l <aau email> ai-fe02.srv.aau.dk
```
2. Transfer files into CLAAUDIA
```bash
scp some-file <aau email>@ai-fe02.srv.aau.dk:~
```
3. Run Singualrity
```bash
srun --pty singularity shell --fakeroot --writable <image name>
```
4. Run the script
```bash
python3 <script name>
```
5. Download files from CLAAUDIA (if it is a directory, add `-r` after scp)
```bash
scp <aau email>@ai-fe02.srv.aau.dk:~/some-folder/some-subfolder/some-file .
```

## Dataset Source

### ATISE
- ICEWS14
- $YAGO11K_{D}$

### TERO
- ICEWS14
- ICEWS05-15
- YAGO11K
- Wikidata12k

### Diachronic Embedding
- ICEWS14
- ICEWS05-15
- GDELT
