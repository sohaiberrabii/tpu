set script_dir [file dirname [info script]]

set_param board.repoPaths [list $script_dir/build/pynq-z2]

create_project -force -part xc7z020clg400-1 tpu_proj $script_dir/build/tpu_proj 
set_property board_part tul.com.tw:pynq-z2:part0:1.0 [current_project]

set bdname "tpu"
create_bd_design $bdname

set processing_system [
    create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7
]

apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {
    make_external "FIXED_IO, DDR"
    apply_board_preset "1"
    Master "Disable"
    Slave "Disable"
} $processing_system

set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
] $processing_system

set rst_ps7_0_100M [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7_0_100M ]
set axi_periph [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_periph ]
set_property CONFIG.NUM_MI {1} $axi_periph

add_files $script_dir/build/tpu.v
set TPU [create_bd_cell -type module -reference TPU TPU_0]

connect_bd_intf_net [get_bd_intf_pins processing_system7/S_AXI_HP0] [get_bd_intf_pins TPU_0/bus]
connect_bd_intf_net [get_bd_intf_pins processing_system7/M_AXI_GP0] [get_bd_intf_pins axi_periph/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_periph/M00_AXI] [get_bd_intf_pins TPU_0/ctrl]

connect_bd_net [get_bd_pins processing_system7/FCLK_CLK0] \
   [get_bd_pins processing_system7/S_AXI_HP0_ACLK] \
   [get_bd_pins rst_ps7_0_100M/slowest_sync_clk] \
   [get_bd_pins axi_periph/M00_ACLK] \
   [get_bd_pins processing_system7/M_AXI_GP0_ACLK] \
   [get_bd_pins axi_periph/S00_ACLK] \
   [get_bd_pins axi_periph/ACLK] \
   [get_bd_pins TPU_0/clk]
connect_bd_net [get_bd_pins processing_system7/FCLK_RESET0_N] \
   [get_bd_pins rst_ps7_0_100M/ext_reset_in]

set inv_rst [create_bd_cell -type ip -vlnv xilinx.com:ip:util_vector_logic:2.0 inv_rst]
set_property -dict [list CONFIG.C_OPERATION {not} CONFIG.C_SIZE {1}] $inv_rst
connect_bd_net [get_bd_pins inv_rst/Res] [get_bd_pins TPU_0/rst]

connect_bd_net [get_bd_pins rst_ps7_0_100M/peripheral_aresetn] \
   [get_bd_pins axi_periph/M00_ARESETN] \
   [get_bd_pins axi_periph/S00_ARESETN] \
   [get_bd_pins axi_periph/ARESETN] \
   [get_bd_pins inv_rst/Op1]

regenerate_bd_layout
save_bd_design

set bdpath [file dirname [get_files [get_property file_name [current_bd_design]]]]
make_wrapper -files [get_files tpu.bd] -top
file delete -force NA
add_files $bdpath/hdl/${bdname}_wrapper.v

launch_runs synth_1 -jobs 4
wait_on_run synth_1

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
