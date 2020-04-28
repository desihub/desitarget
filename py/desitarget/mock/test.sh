date +"%T"

select_mock_targets --output_dir /global/cscratch1/sd/mjwilson/trash --nside 64 --config data/select-mock-targets-no-contam.yaml --healpixels 0 --no-spectra --overwrite

date +"%T"
