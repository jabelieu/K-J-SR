import pysr as srp

targ_file = 'targ_path.txt'

with open ( targ_file , 'r' ) as f :
    path = f.readline()

table = srp.PySRRegressor.from_file(run_directory=path)

latex_table = table.latex_table()

with open ('latex_table.txt', 'w') as f:
    f.write(latex_table)