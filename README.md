# mace-md-gui
GUI for running molecular dynamics simulations with MACE interatomic potential

### **Compile the app**  
Open your terminal console and write the following commands (the bold text):  
(Optional) Install Git:  
      **sudo apt update**  
      **sudo apt install git**    
      
1) Download the XRDlicious code from GitHub (or download it manually without Git on the following link by clicking on 'Code' and 'Download ZIP', then extract the ZIP. With Git, it is automatically extracted):  
      **git clone https://github.com/bracerino/mace-md-gui.git**

2) Navigate to the downloaded project folder:  
      **cd mace-md-gui/**

3) Create a Python virtual environment to prevent possible conflicts between packages:  
      **python3 -m venv mace_env**

4) Activate the Python virtual environment (before activating, make sure you are inside the xrdlicious folder):  
      **source mace_env/bin/activate**
   
5) Install all the necessary Python packages:  
      **pip install -r requirements.txt**

6) Run the XRDlicious app (always before running it, make sure to activate its Python virtual environment (Step 4):  
      **streamlit run app.py**
