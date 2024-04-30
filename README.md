# TMS-TG
tms_tg is a python library for analysing electrophysiological (EPhys) data collected in Giese-Schwarz lab. The main objective is to perform data collation, processing and analysis of very large dataset in memory/computation -light manner.

## Features
<ins>Memory-light</ins> \
Necesary data is loaded into memory as one progresses through the processing / analysis steps. In addition, raw data in unloaded when a processing or analysis step requiring that data finishes executing.

<ins>Computation-light</ins> \
The "function caching" method is used for computation-heavy procedures. Behind the scenes and hidden from user, state variables are used to make decisions regarding uncaching, with the user experiencing just the increased performance. 

## Additional features
Parallel computing for computation heavy procedures.
A unified data-structure (built over pandas) for storing the results of processing / analysis steps.
Scripts for visualising popular analysis results. 
