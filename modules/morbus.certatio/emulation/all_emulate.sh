for folder in $(ls /data/quo.vadis/data/pe.dataset/PeX86Exe/); do
    python3 emulate_samples.py --debug --sample-prefix /data/quo.vadis/data/pe.dataset/PeX86Exe/$folder --output report_$folder > report_$folder.log 2>&1 &
done