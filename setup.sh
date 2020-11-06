conda create --name turkey

conda activate turkey

echo "INSTALLING REQUIRED PACKAGES"
      
echo "cwd="
pwd 
    
echo "VERIFYING 'python' location:"
which python
    
echo "VERIFYING 'pip' location:"
which pip

pip install numpy
conda install py-opencv
