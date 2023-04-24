python3 -m venv $1

source ./$1/bin/activate

python3 -m pip install -r requirements.txt

python3 cluster.py $2