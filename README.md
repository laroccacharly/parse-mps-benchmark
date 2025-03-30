# parse-mps-benchmark 

MPS files are the standard used in MIP to store mathematical problems on disk. 
However, this is a text-based file format which can be computationally expensive to parse for large instances. 
This project benchmarks the runtime required to parse MPS files in MIPLIB using the HiGHS solver. 
We then propose a new solution to store the problems as parquet files to accelerate the parsing. 