// SPDX-License-Identifier: GPL-2.0-only
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/vfio_pci_core.h>
#include <drm/nvidia_vgpu.h>

struct nvidia_vgpu_pci_core_device {
	struct vfio_pci_core_device core_device;
};

static inline unsigned int nvidia_vgpu_vf_gfid(struct pci_dev *vf)
{
	return pci_iov_vf_id(vf) + 1;
}

static inline unsigned int nvidia_vgpu_vf_dbdf(struct pci_dev *vf)
{
	return ((u32)pci_domain_nr(vf->bus) << 16) | pci_dev_id(vf);
}

static int nvidia_vgpu_open_device(struct vfio_device *core_vdev)
{
	struct nvidia_vgpu_pci_core_device *nvdev = container_of(
		core_vdev, struct nvidia_vgpu_pci_core_device, core_device.vdev);
	struct pci_dev *vf = to_pci_dev(core_vdev->dev);
	int ret;

	ret = vfio_pci_core_enable(&nvdev->core_device);
	if (ret)
		return ret;

	ret = nvidia_vgpu_open(pci_physfn(vf), nvidia_vgpu_vf_gfid(vf),
			       nvidia_vgpu_vf_dbdf(vf));
	if (ret) {
		vfio_pci_core_disable(&nvdev->core_device);
		return ret;
	}

	vfio_pci_core_finish_enable(&nvdev->core_device);
	return 0;
}

static void nvidia_vgpu_close_device(struct vfio_device *core_vdev)
{
	struct pci_dev *vf = to_pci_dev(core_vdev->dev);

	nvidia_vgpu_close(pci_physfn(vf), nvidia_vgpu_vf_gfid(vf));
	vfio_pci_core_close_device(core_vdev);
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
	.device_feature	= vfio_pci_core_ioctl_feature,
	.read		= vfio_pci_core_read,
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
