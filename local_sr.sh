echo
echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-="
echo "| Formatting data files... |"
echo "=-=-=-=-=-=-=-=-=-=-=-=-=-=-"
echo

python3 input_data_formatter.py

echo

python3 output_data_formatter.py

echo
echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-"
echo "| Activating regressor... |"
echo "=-=-=-=-=-=-=-=-=-=-=-=-=-="
echo

python3 local_sr.py

echo
echo "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-"
echo "| Generating latex table... |"
echo "=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
echo

python3 sr_latex_table.py

targ_direc=$(head -n 1 "targ_path.txt")

mv "latex_table.txt" "$targ_direc"
mv "sr_parameters.txt" "$targ_direc"

echo
echo "-=-=-=-=-=-=-"
echo "| Job Done! |"
echo "=-=-=-=-=-=-="
echo