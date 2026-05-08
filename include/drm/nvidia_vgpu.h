/* SPDX-License-Identifier: GPL-2.0 */
#ifndef __DRM_NVIDIA_VGPU_H__
#define __DRM_NVIDIA_VGPU_H__

#include <linux/types.h>

struct pci_dev;

int nvidia_vgpu_open(struct pci_dev *pf_pdev, int vf_id, u32 dbdf);
void nvidia_vgpu_close(struct pci_dev *pf_pdev, int vf_id);
int nvidia_vgpu_reset(struct pci_dev *pf_pdev, int vf_id);

#endif /* __DRM_NVIDIA_VGPU_H__ */
