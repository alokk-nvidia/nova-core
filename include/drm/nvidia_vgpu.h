/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __DRM_NVIDIA_VGPU_H__
#define __DRM_NVIDIA_VGPU_H__

#include <linux/types.h>

struct pci_dev;

/**
 * struct nvidia_vgpu_type_info - vGPU type descriptor returned by open
 * @pci_dev_id:   PCI device ID to present to the guest
 * @pci_subsys_id: PCI subsystem ID to present to the guest
 * @bar1_length:  BAR1 aperture size in bytes
 */
struct nvidia_vgpu_type_info {
	u32 pci_dev_id;
	u32 pci_subsys_id;
	u64 bar1_length;
};

int nvidia_vgpu_open(struct pci_dev *pf_pdev, int vf_id, u32 dbdf,
		     struct nvidia_vgpu_type_info *type_info);
void nvidia_vgpu_close(struct pci_dev *pf_pdev, int vf_id);
int nvidia_vgpu_reset(struct pci_dev *pf_pdev, int vf_id);

#endif /* __DRM_NVIDIA_VGPU_H__ */
