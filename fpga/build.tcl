set script_dir [file dirname [info script]]

set projectname "tpu_proj"

open_project $script_dir/build/$projectname

launch_runs synth_1 -jobs 4
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
