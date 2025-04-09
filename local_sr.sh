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
echo "-=-=-=-=-=-=-"
echo "| Job Done! |"
echo "=-=-=-=-=-=-="
echo