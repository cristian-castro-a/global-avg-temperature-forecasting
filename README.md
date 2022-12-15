# World Poverty Forecasting

This is a project that aims at **forecasting world poverty** as a multivariate problem. 

Team members:
- [Cristian Castro](https://github.com/cristian-castro-a)
- [Gabrijela Juresic]()
- [Swapnali Sonkusale]()
- [Tejas Choudekar]()

## Virtual Environment and Dependencies

### Windows 
This project is based on Python 3.9. For windows operating system follow below steps to get the project started:
1. Install Anaconda
2. Go to anaconda and write `where conda` it will give you three paths
3. Add `C:\Users\user_\anaconda3` `C:\Users\user_\anaconda3\Scripts` `C:\Users\user_\anaconda3\Library\bin` path to environment variables in path variable
4. Install libraries using `conda install --file requirements_windows.txt`
5. For creating virtual environment and installing use below command: 
```bash 
conda create --name <env> --file requirements_windows.txt
```

### MacOS
This project is based on Python 3.9. If you own a MacOS computer with M1 chip, please use the `requirements_m1_macos.txt` file to install all dependencies necessary for this project in a Conda environment, by running the following command on your terminal:
```bash
conda create --name py39 --file requirements.txt
```

## Data
1. Data on Poverty by [Our World in Data](https://ourworldindata.org/): https://github.com/owid/poverty-data