// SPDX-License-Identifier: GPL-2.0

use kernel::{
    device,
    io::Io,
    prelude::*,
    time::{
        delay::fsleep,
        Delta,
        Instant,
        Monotonic, //
    },
};

use crate::{
    driver::Bar0,
    gsp::nvkv,
    vgpu::{
        consts::{
            config_keys,
            plugin_rpc as consts,
            plugin_rpc::RpcMsg,
            set_bme_keys, //
        },
        Bar1Map,
        Gfid,
        VgpuInstance, //
    },
};

/// Plugin RPC channel for vGPU plugin communication.
pub(crate) struct PluginRpc {
    bar1_map: Bar1Map,
    ctrl_off: usize,
    resp_off: usize,
    msg_off: usize,
    msg_seq_num: u32,
}

impl PluginRpc {
    pub(crate) fn new(bar1_map: Bar1Map) -> Self {
        Self {
            bar1_map,
            ctrl_off: 0,
            resp_off: consts::CTRL_SIZE as usize,
            msg_off: (consts::CTRL_SIZE + consts::RESPONSE_SIZE) as usize,
            msg_seq_num: 0,
        }
    }

    /// Consume this `PluginRpc` and release the underlying BAR1 page table mapping.
    pub(crate) fn destroy(
        self,
        dev: &device::Device<device::Bound>,
        mm: &crate::mm::GpuMm,
    ) -> Result {
        self.bar1_map.destroy(dev, mm)
    }

    /// Poll ctrl_buf for plugin boot completion magic.
    pub(crate) fn wait_plugin_ready(&self, dev: &device::Device<device::Bound>) -> Result {
        let start = Instant::<Monotonic>::now();
        let timeout = Delta::from_millis(consts::PLUGIN_BOOT_TIMEOUT_MS as i64);
        loop {
            let val = self
                .bar1_map
                .read32(dev, consts::CTRL_BUF_MSG_SEQ_NUM_OFFSET)?;
            if val == consts::GSP_PLUGIN_BOOTLOADED {
                let elapsed = start.elapsed();
                pr_debug!(
                    "wait_plugin_ready: got magic {:#x} after {:?}\n",
                    val,
                    elapsed
                );
                return Ok(());
            }
            if start.elapsed() > timeout {
                pr_debug!(
                    "wait_plugin_ready: timeout after {}ms, last val={:#x}\n",
                    consts::PLUGIN_BOOT_TIMEOUT_MS,
                    val
                );
                return Err(ETIMEDOUT);
            }
            fsleep(Delta::from_millis(1));
        }
    }

    fn write_ctrl_u64(&self, dev: &device::Device<device::Bound>, off: usize, val: u64) -> Result {
        self.bar1_map
            .write32(dev, (self.ctrl_off + off) as u64, val as u32)?;
        self.bar1_map
            .write32(dev, (self.ctrl_off + off + 4) as u64, (val >> 32) as u32)
    }

    fn write_ctrl_u32(&self, dev: &device::Device<device::Bound>, off: usize, val: u32) -> Result {
        self.bar1_map.write32(dev, (self.ctrl_off + off) as u64, val)
    }

    /// Initialize the RPC control buffer with version and all buffer offsets.
    ///
    /// Must match the VGPU_CPU_GSP_CTRL_BUFF_REGION layout from dev_vgpu_gsp.h:
    ///   +0   version (u32)
    ///   +4   message_type (u32)
    ///   +8   message_seq_num (u32)
    ///   +16  response_buff_offset (u64)
    ///   +24  message_buff_offset (u64)
    ///   +32  migration_buff_offset (u64)
    ///   +40  error_buff_offset (u64)
    ///   +48  guest_rpc_trace_buff_offset (u64)
    ///   +56  migration_buf_cpu_access_offset (u32)
    ///   +60  is_migration_in_progress (u32/NvBool)
    ///   +64  error_buff_cpu_get_idx (u32)
    ///   +68  guest_rpc_trace_buff_cpu_get_idx (u32)
    ///   +72  attached_vgpu_count (u32)
    ///   +76  is_gr_init_done (u32/NvBool)
    pub(crate) fn init_rpc(&mut self, dev: &device::Device<device::Bound>) -> Result {
        self.write_ctrl_u32(dev, 0, consts::CTRL_BUFF_VERSION)?;
        self.write_ctrl_u64(dev, 16, self.resp_off as u64)?;
        self.write_ctrl_u64(dev, 24, self.msg_off as u64)?;
        self.write_ctrl_u64(dev, 32, consts::MIGRATION_BUFF_OFFSET)?;
        self.write_ctrl_u64(dev, 40, consts::ERROR_BUFF_OFFSET)?;
        self.write_ctrl_u64(dev, 48, consts::GUEST_RPC_TRACE_BUFF_OFFSET)?;
        self.write_ctrl_u32(dev, 56, 0)?; // migration_buf_cpu_access_offset
        self.write_ctrl_u32(dev, 60, 0)?; // is_migration_in_progress
        self.write_ctrl_u32(dev, 64, 0)?; // error_buff_cpu_get_idx
        self.write_ctrl_u32(dev, 68, 0)?; // guest_rpc_trace_buff_cpu_get_idx
        self.write_ctrl_u32(dev, 72, 1)?; // attached_vgpu_count
        self.write_ctrl_u32(dev, 76, 0)?; // is_gr_init_done
        Ok(())
    }

    /// Generic RPC call: write msg, set ctrl, doorbell, poll response.
    pub(crate) fn rpc_call(
        &mut self,
        dev: &device::Device<device::Bound>,
        bar0: &Bar0,
        gfid: Gfid,
        msg_type: RpcMsg,
        data: &[u8],
    ) -> Result {
        for (i, chunk) in data.chunks(4).enumerate() {
            let val = u32::from_le_bytes({
                let mut buf = [0u8; 4];
                buf[..chunk.len()].copy_from_slice(chunk);
                buf
            });
            self.bar1_map
                .write32(dev, (self.msg_off + i * 4) as u64, val)?;
        }

        self.msg_seq_num += 1;
        pr_debug!(
            "rpc_call: gfid={} msg_type={} data_len={} seq={}\n",
            gfid.0,
            msg_type as u32,
            data.len(),
            self.msg_seq_num
        );
        self.bar1_map
            .write32(dev, (self.ctrl_off + 4) as u64, msg_type as u32)?;
        self.bar1_map
            .write32(dev, (self.ctrl_off + 8) as u64, self.msg_seq_num)?;

        ring_doorbell(bar0, gfid);
        self.wait_response(dev)
    }

    fn wait_response(&self, dev: &device::Device<device::Bound>) -> Result {
        let start = Instant::<Monotonic>::now();
        let timeout = Delta::from_secs(120);
        loop {
            let processed = self.bar1_map.read32(dev, (self.resp_off + 4) as u64)?;
            if processed == self.msg_seq_num {
                let result = self.bar1_map.read32(dev, (self.resp_off + 8) as u64)?;
                if result != 0 {
                    pr_debug!(
                        "wait_response: seq={} plugin returned error {}\n",
                        self.msg_seq_num,
                        result
                    );
                    return Err(EIO);
                }
                let elapsed = start.elapsed();
                pr_debug!(
                    "wait_response: seq={} done after {:?}\n",
                    self.msg_seq_num,
                    elapsed
                );
                return Ok(());
            }
            if start.elapsed() > timeout {
                pr_debug!(
                    "wait_response: seq={} timeout, last processed={}\n",
                    self.msg_seq_num,
                    processed
                );
                return Err(ETIMEDOUT);
            }
            fsleep(Delta::from_millis(1));
        }
    }

    /// RPC version negotiation (no data).
    pub(crate) fn negotiate_rpc_version(
        &mut self,
        dev: &device::Device<device::Bound>,
        bar0: &Bar0,
        gfid: Gfid,
    ) -> Result {
        self.rpc_call(dev, bar0, gfid, RpcMsg::VersionNegotiation, &[])
    }

    /// Send configuration parameters as NVKV-encoded msg_buff_v2.
    pub(crate) fn send_config_params(
        &mut self,
        dev: &device::Device<device::Bound>,
        bar0: &Bar0,
        instance: &VgpuInstance,
    ) -> Result {
        let mut kvs: KVec<u64> = KVec::new();

        nvkv::nvkv_push_array8(&mut kvs, config_keys::UUID, &[0u8; 16])?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::DBDF, instance.dbdf.0)?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::DEV_INST, 0)?;
        nvkv::nvkv_push_imm32(
            &mut kvs,
            config_keys::VGPU_TYPE,
            instance.vgpu_type.vgpu_type_id,
        )?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::VM_PID, instance.vm_pid)?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::SWIZZ_ID, config_keys::SWIZZ_ID_NONE)?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::NUM_CHANNELS, instance.num_chid - 1)?;
        nvkv::nvkv_push_imm32(
            &mut kvs,
            config_keys::NUM_PLUGIN_CHANNELS,
            instance.num_plugin_channels,
        )?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::VMM_CAP, 0)?;
        nvkv::nvkv_push_imm32(&mut kvs, config_keys::MIGRATION_FEATURE, config_keys::MIGRATION_FEATURE_KVM)?;
        nvkv::nvkv_push_imm32(
            &mut kvs,
            config_keys::HYPERVISOR_TYPE,
            config_keys::HYPERVISOR_UNKNOWN,
        )?;
        nvkv::nvkv_push_imm32(
            &mut kvs,
            config_keys::CPU_ARCH,
            config_keys::CPU_ARCH_X86_64,
        )?;
        nvkv::nvkv_push_seq64(&mut kvs, config_keys::PAGE_SIZE, &[config_keys::PAGE_SIZE_4K])?;
        nvkv::nvkv_push_seq64(
            &mut kvs,
            config_keys::FEATURE_FLAGS,
            &[consts::FEATURE_FLAG_ENABLE_UVM | consts::FEATURE_FLAG_VMM_MIGRATION],
        )?;

        let mut msg: KVec<u64> = KVec::new();
        msg.push(kvs.len() as u64, GFP_KERNEL)?;
        for &v in kvs.as_slice() {
            msg.push(v, GFP_KERNEL)?;
        }

        // SAFETY: `msg` is a valid `KVec<u64>` and we reinterpret it as a byte
        // slice of the same total size. The pointer is valid for `len * 8` bytes.
        let payload: &[u8] =
            unsafe { core::slice::from_raw_parts(msg.as_ptr().cast::<u8>(), msg.len() * 8) };
        self.rpc_call(
            dev,
            bar0,
            instance.gfid,
            RpcMsg::SetupConfigParamsAndInit,
            payload,
        )
    }

    /// Send BME (Bus Master Enable) state update via NVKV-encoded payload.
    pub(crate) fn set_bme(
        &mut self,
        dev: &device::Device<device::Bound>,
        bar0: &Bar0,
        gfid: Gfid,
        enable: bool,
    ) -> Result {
        let mut kvs: KVec<u64> = KVec::new();
        nvkv::nvkv_push_imm32(&mut kvs, set_bme_keys::BME_ENABLE, enable as u32)?;

        let mut msg: KVec<u64> = KVec::new();
        msg.push(kvs.len() as u64, GFP_KERNEL)?;
        for &v in kvs.as_slice() {
            msg.push(v, GFP_KERNEL)?;
        }

        let payload: &[u8] =
            unsafe { core::slice::from_raw_parts(msg.as_ptr().cast::<u8>(), msg.len() * 8) };
        self.rpc_call(dev, bar0, gfid, RpcMsg::UpdateBmeState, payload)
    }
}

fn ring_doorbell(bar0: &Bar0, gfid: Gfid) {
    let v = gfid.0 * 32 + 17;
    let _ = bar0.try_write32(v, consts::NV_VIRTUAL_FUNCTION_PRIV_DOORBELL);
    let _ = bar0.try_read32(consts::NV_VIRTUAL_FUNCTION_PRIV_DOORBELL);
}
