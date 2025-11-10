


for warm_up_steps in 0 2 4; do
    run_number=10
uv run nsys profile -o result python method_timeit.py --model_cfg_path /home/niu/code/cs336/assignment2-systems/cs336_systems/benchmarking_script/timeit_strategy/model_configs.yaml \
                        --warm_up_steps $warm_up_steps \
                        --run_number $run_number > ./nvtx_warm_up_steps_$warm_up_steps.txt
done
