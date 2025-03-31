# parse-mps-benchmark 

Replacing MPS with Parquet, is it a good idea? 

[![Watch the presentation](https://img.youtube.com/vi/BQ9Mk00q-II/0.jpg)](https://youtu.be/BQ9Mk00q-II)

Research question: 
Does converting MIP problems from the text-based MPS format to the binary Parquet format help speed up the HiGHS solver? 

MPS files are the standard used in MIP to store mathematical problems on disk. 
However, MPS is a text-based file format that can be expensive to parse for large instances. 
This project benchmarks the runtime required to parse MPS files in MIPLIB using the HiGHS solver. 
We propose a new solution to store the problems as parquet files to accelerate the parsing. 