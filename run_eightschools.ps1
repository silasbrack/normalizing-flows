.\venv\Scripts\activate
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 4 --file-name eightschools_iaf_4
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 8 --file-name eightschools_iaf_8
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 16 --file-name eightschools_iaf_16
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 32 --file-name eightschools_iaf_32
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 64 --file-name eightschools_iaf_64
python .\src\experiments\run_eightschools.py --epochs 1000 --flow-type neural_autoregressive --num-flows 128 --file-name eightschools_iaf_128
