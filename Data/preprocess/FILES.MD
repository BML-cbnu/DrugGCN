# EXP & PPI
## EXP (FILENAME: EXP.csv)
    1. Columns name: Gene symbols.
    2. Index name: Cell lines name or number.
    3. Data: Gene expression
    4. We used GDSC datasets.

## PPI
   ### INFO (FILENAME: PPI_INFO.txt)
    1. Columns name: ['STRING ID', 'preferred_name', ...]
    2. No index
    3. column 1: STRING ID(ex. 9606.ENSP00000000233)
    4. column 2: gene symbol(ex. ARF5)

   ### LINK (FILENAME: PPI_LINK.txt)
    1. Columns name: ['protein1', 'protein2', 'combined_score', ...]
    2. No index
    3. columns 1, 2: STRING ID(ex. 9606.ENSP00000000233)
    4. column 3: combined score between column1 and column2

## LIST (FILENAME: LIST.txt)
    1. list of gene symbols(separator: tab)
    ex LIST.txt)
    TSPAN6	FGR	CFH	FUCA2	GCLC	NFYA	NIPAL3	ENPP4 ...

