#for folder in $(ls /data/quo.vadis/data/pe.dataset/PeX86Exe/); do
for folder in $(ls /data/quo.vadis/data/pe.dataset/testset/); do
    python3 emulate_samples.py --debug --sample-prefix /data/quo.vadis/data/pe.dataset/testset/$folder --output ../testset_emulation/report_$folder > emulation_logs/testset_report_$folder.log 2>&1 &
done