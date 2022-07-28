#!/bin/bash
while IFS=$'\t' read n function failrandom failfraction method rep
do
    JOB=`sbatch - << EOJ

#!/bin/bash
#SBATCH -A <allocationID>
#SBATCH --partition=<queueName>
#SBATCH --time=<hh:mm:ss>
#SBATCH --mail-user=<emailAddress>
#SBATCH --output=<combined out and err file path>
#SBATCH -J <jobName>
#SBATCH --nodes=1
#SBATCH -n <core count>
#SBATCH --mem=<Total memory in MB>

# unload modules that may have been loaded when job was submitted
module purge all

# load the version of python you want to use
module load python/anaconda3.6
source activate surmise-venv

# By default all file paths are relative to the directory where you submitted the job.
# To change to another path, use `cd <path>`, for example:
# cd /projects/<allocationID>
cd ~/research/surmise/research/emucomp/quest-pcgpwm-surmise

python emucompare.py --n=${n} --function=${function} --failrandom=${failrandom} --failfraction=${failfraction} --method=${method} --rep=${rep}
EOJ
`

# print out the job id for reference later
echo "JobID = ${JOB} for parameters submitted on `date`"
done < params.txt
exit

