// SPDX-License-Identifier: GPL-2.0-only
#include <linux/delay.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/vfio_pci_core.h>
#include <drm/nvidia_vgpu.h>

#define VF_FLR_WAIT_STEP_MS  100
#define VF_FLR_WAIT_MAX_MS  1000

static int nvidia_vgpu_fb_bar_index(struct pci_dev *pdev)
{
	if (pci_resource_flags(pdev, 0) & IORESOURCE_MEM_64)
		return 2;
	return 1;
}

struct nvidia_vgpu_pci_core_device {
	struct vfio_pci_core_device core_device;
	struct nvidia_vgpu_type_info type_info;
};

static inline unsigned int nvidia_vgpu_vf_gfid(struct pci_dev *vf)
{
	return pci_iov_vf_id(vf) + 1;
}

static inline unsigned int nvidia_vgpu_vf_dbdf(struct pci_dev *vf)
{
	return ((u32)pci_domain_nr(vf->bus) << 16) | pci_dev_id(vf);
}

static int do_vf_flr(struct pci_dev *pdev)
{
	u32 data, elapsed = 0;

	if (!pdev->is_virtfn)
		return 0;

	pcie_capability_read_dword(pdev, PCI_EXP_DEVCAP, &data);
	if (!(data & PCI_EXP_DEVCAP_FLR)) {
		pci_err(pdev, "FLR capability not present\n");
		return -EINVAL;
	}

	device_lock(&pdev->dev);
	pci_set_power_state(pdev, PCI_D0);
	pci_save_state(pdev);

	if (!pci_wait_for_pending_transaction(pdev))
		pci_err(pdev, "timed out waiting for pending transactions\n");

	pcie_capability_set_word(pdev, PCI_EXP_DEVCTL, PCI_EXP_DEVCTL_BCR_FLR);

	do {
		msleep(VF_FLR_WAIT_STEP_MS);
		elapsed += VF_FLR_WAIT_STEP_MS;
		pci_read_config_dword(pdev, PCI_VENDOR_ID, &data);
	} while ((data & 0xffff) == 0x0001 && elapsed < VF_FLR_WAIT_MAX_MS);

	if (elapsed < VF_FLR_WAIT_MAX_MS) {
		do {
			pci_read_config_dword(pdev, PCI_COMMAND, &data);
			if (data != ~0u)
				goto done;
			msleep(VF_FLR_WAIT_STEP_MS);
			elapsed += VF_FLR_WAIT_STEP_MS;
		} while (elapsed < VF_FLR_WAIT_MAX_MS);
		pci_err(pdev, "FLR failed (non-CRS), waited %u ms\n", elapsed);
	} else {
		pci_err(pdev, "FLR failed (CRS), waited %u ms\n", elapsed);
	}

	device_unlock(&pdev->dev);
	return -ETIMEDOUT;

done:
	pci_restore_state(pdev);
	device_unlock(&pdev->dev);
	pci_info(pdev, "FLR done in %u ms\n", elapsed);
	return 0;
}

static int nvidia_vgpu_open_device(struct vfio_device *core_vdev)
{
	struct nvidia_vgpu_pci_core_device *nvdev = container_of(
		core_vdev, struct nvidia_vgpu_pci_core_device, core_device.vdev);
	struct pci_dev *vf = to_pci_dev(core_vdev->dev);
	struct nvidia_vgpu_type_info type_info;
	int ret;

	ret = vfio_pci_core_enable(&nvdev->core_device);
	if (ret)
		return ret;

	ret = do_vf_flr(vf);
	if (ret) {
		vfio_pci_core_disable(&nvdev->core_device);
		return ret;
	}

	ret = nvidia_vgpu_open(pci_physfn(vf), nvidia_vgpu_vf_gfid(vf),
			       nvidia_vgpu_vf_dbdf(vf), &type_info);
	if (ret) {
		vfio_pci_core_disable(&nvdev->core_device);
		return ret;
	}

	nvdev->type_info = type_info;
	pci_info(vf, "vgpu open: dev_id=0x%x subsys_id=0x%x bar1_length=0x%llx\n",
		 type_info.pci_dev_id, type_info.pci_subsys_id,
		 type_info.bar1_length);
	vfio_pci_core_finish_enable(&nvdev->core_device);
	return 0;
}

static void nvidia_vgpu_close_device(struct vfio_device *core_vdev)
{
	struct pci_dev *vf = to_pci_dev(core_vdev->dev);

	nvidia_vgpu_close(pci_physfn(vf), nvidia_vgpu_vf_gfid(vf));
	vfio_pci_core_close_device(core_vdev);
}

static ssize_t nvidia_vgpu_pci_read_config(struct vfio_device *core_vdev,
					   char __user *buf, size_t count,
					   loff_t *ppos)
{
	struct nvidia_vgpu_pci_core_device *nvdev = container_of(
		core_vdev, struct nvidia_vgpu_pci_core_device, core_device.vdev);
	struct nvidia_vgpu_type_info *ti = &nvdev->type_info;
	loff_t pos = *ppos & VFIO_PCI_OFFSET_MASK;
	size_t register_offset;
	loff_t copy_offset;
	size_t copy_count;
	__le16 val16;
	int ret;

	ret = vfio_pci_core_read(core_vdev, buf, count, ppos);
	if (ret < 0)
		return ret;

	if (vfio_pci_core_range_intersect_range(pos, count, PCI_DEVICE_ID,
						sizeof(val16), &copy_offset,
						&copy_count, &register_offset)) {
		val16 = cpu_to_le16(ti->pci_dev_id);
		if (copy_to_user(buf + copy_offset,
				 (void *)&val16 + register_offset, copy_count))
			return -EFAULT;
	}

	if (vfio_pci_core_range_intersect_range(pos, count, PCI_SUBSYSTEM_VENDOR_ID,
						sizeof(val16), &copy_offset,
						&copy_count, &register_offset)) {
		val16 = cpu_to_le16(PCI_VENDOR_ID_NVIDIA);
		if (copy_to_user(buf + copy_offset,
				 (void *)&val16 + register_offset, copy_count))
			return -EFAULT;
	}

	if (vfio_pci_core_range_intersect_range(pos, count, PCI_SUBSYSTEM_ID,
						sizeof(val16), &copy_offset,
						&copy_count, &register_offset)) {
		val16 = cpu_to_le16(ti->pci_subsys_id);
		if (copy_to_user(buf + copy_offset,
				 (void *)&val16 + register_offset, copy_count))
			return -EFAULT;
	}

	return count;
}

static ssize_t nvidia_vgpu_pci_read(struct vfio_device *core_vdev,
				    char __user *buf, size_t count,
				    loff_t *ppos)
{
	unsigned int index = VFIO_PCI_OFFSET_TO_INDEX(*ppos);

	if (index == VFIO_PCI_CONFIG_REGION_INDEX)
		return nvidia_vgpu_pci_read_config(core_vdev, buf, count, ppos);

	return vfio_pci_core_read(core_vdev, buf, count, ppos);
}

static u64 nvidia_vgpu_bar1_size(struct nvidia_vgpu_pci_core_device *nvdev)
{
	return nvdev->type_info.bar1_length << 20;
}

static int nvidia_vgpu_get_region_info(struct vfio_device *core_vdev,
				       struct vfio_region_info *info,
				       struct vfio_info_cap *caps)
{
	int ret;

	ret = vfio_pci_ioctl_get_region_info(core_vdev, info, caps);
	if (ret)
		return ret;

	if (info->index == nvidia_vgpu_fb_bar_index(
		to_pci_dev(core_vdev->dev)) && info->size) {
		struct nvidia_vgpu_pci_core_device *nvdev = container_of(
			core_vdev, struct nvidia_vgpu_pci_core_device,
			core_device.vdev);
		u64 vgpu_bar1 = nvidia_vgpu_bar1_size(nvdev);

		if (vgpu_bar1 && vgpu_bar1 < info->size)
			info->size = vgpu_bar1;
	}

	return 0;
}

static long nvidia_vgpu_pci_ioctl(struct vfio_device *core_vdev,
				  unsigned int cmd, unsigned long arg)
{
	if (cmd == VFIO_DEVICE_RESET) {
		struct pci_dev *vf = to_pci_dev(core_vdev->dev);

		nvidia_vgpu_reset(pci_physfn(vf), nvidia_vgpu_vf_gfid(vf));
	}

	return vfio_pci_core_ioctl(core_vdev, cmd, arg);
}

static const struct vfio_device_ops nvidia_vgpu_pci_ops = {
	.name		= "nvidia-vgpu-pci",
	.init		= vfio_pci_core_init_dev,
	.release	= vfio_pci_core_release_dev,
	.open_device	= nvidia_vgpu_open_device,
	.close_device	= nvidia_vgpu_close_device,
	.ioctl		= nvidia_vgpu_pci_ioctl,
	.get_region_info_caps = nvidia_vgpu_get_region_info,
	.device_feature	= vfio_pci_core_ioctl_feature,
	.read		= nvidia_vgpu_pci_read,
	.write		= vfio_pci_core_write,
	.mmap		= vfio_pci_core_mmap,
	.request	= vfio_pci_core_request,
	.match		= vfio_pci_core_match,
	.match_token_uuid = vfio_pci_core_match_token_uuid,
	.bind_iommufd	= vfio_iommufd_physical_bind,
	.unbind_iommufd	= vfio_iommufd_physical_unbind,
	.attach_ioas	= vfio_iommufd_physical_attach_ioas,
	.detach_ioas	= vfio_iommufd_physical_detach_ioas,
};

static int nvidia_vgpu_pci_probe(struct pci_dev *pdev,
			       const struct pci_device_id *id)
{
	struct nvidia_vgpu_pci_core_device *nvdev;
	int ret;

	nvdev = vfio_alloc_device(nvidia_vgpu_pci_core_device, core_device.vdev,
				  &pdev->dev, &nvidia_vgpu_pci_ops);
	if (IS_ERR(nvdev))
		return PTR_ERR(nvdev);

	dev_set_drvdata(&pdev->dev, nvdev);
	ret = vfio_pci_core_register_device(&nvdev->core_device);
	if (ret)
		goto out_put_vdev;

	return 0;

out_put_vdev:
	vfio_put_device(&nvdev->core_device.vdev);
	return ret;
}

static void nvidia_vgpu_pci_remove(struct pci_dev *pdev)
{
	struct nvidia_vgpu_pci_core_device *nvdev = dev_get_drvdata(&pdev->dev);

	vfio_pci_core_unregister_device(&nvdev->core_device);
	vfio_put_device(&nvdev->core_device.vdev);
}

static const struct pci_device_id nvidia_vgpu_pci_table[] = {
	/* Placeholder: match all NVIDIA VFs (vendor 0x10de) */
	{ PCI_DRIVER_OVERRIDE_DEVICE_VFIO(PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID) },
	{}
};
MODULE_DEVICE_TABLE(pci, nvidia_vgpu_pci_table);

static struct pci_driver nvidia_vgpu_pci_driver = {
	.name		= "nvidia-vgpu-pci",
	.id_table	= nvidia_vgpu_pci_table,
	.probe		= nvidia_vgpu_pci_probe,
	.remove		= nvidia_vgpu_pci_remove,
	.driver_managed_dma = true,
};
module_pci_driver(nvidia_vgpu_pci_driver);

MODULE_DESCRIPTION("NVIDIA vGPU vfio-pci driver");
MODULE_LICENSE("GPL");
MODULE_IMPORT_NS("NOVA_CORE_VGPU");
