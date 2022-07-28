#!/bin/bash
while IFS=$'\t' read n function failrandom failfraction method rep
do
    JOB=`sbatch - << EOJ

#!/bin/bash
#SBATCH --account=p30845
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --mail-user=mosesyhc@u.northwestern.edu
#SBATCH -J emucompSurmise
#SBATCH --output=outlog ## standard out and standard error goes to this file
#SBATCH --nodes=1
#SBATCH -n 1

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

