// SPDX-License-Identifier: GPL-2.0

#include <linux/export.h>
#include <linux/pci.h>

extern int nvidia_vgpu_open(struct pci_dev *pf_pdev, unsigned int gfid,
			    unsigned int dbdf);
extern void nvidia_vgpu_close(struct pci_dev *pf_pdev, unsigned int gfid);
extern int nvidia_vgpu_reset(struct pci_dev *pf_pdev, unsigned int gfid);

EXPORT_SYMBOL_NS_GPL(nvidia_vgpu_open, "NOVA_CORE_VGPU");
EXPORT_SYMBOL_NS_GPL(nvidia_vgpu_close, "NOVA_CORE_VGPU");
EXPORT_SYMBOL_NS_GPL(nvidia_vgpu_reset, "NOVA_CORE_VGPU");
