from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

from caer_research import devices


class AcceleratorDeviceTests(unittest.TestCase):
    def test_parse_device_ids_rejects_duplicates(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            devices.parse_device_ids("0,0")

    def test_rocm_visibility_uses_rocr_variable_only(self) -> None:
        with (
            patch.object(devices.torch.version, "hip", "7.2.0"),
            patch.dict(
                os.environ,
                {"CUDA_VISIBLE_DEVICES": "3", "HIP_VISIBLE_DEVICES": "3"},
                clear=False,
            ),
        ):
            devices.configure_visible_devices("1,2")

            self.assertEqual(os.environ["ROCR_VISIBLE_DEVICES"], "1,2")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            self.assertNotIn("HIP_VISIBLE_DEVICES", os.environ)

    def test_snapshot_maps_requested_to_visible_indices(self) -> None:
        properties = Mock(name="properties")
        properties.name = "AMD Test GPU"
        with (
            patch.object(devices.torch.version, "hip", "7.2.0"),
            patch.object(devices.torch.cuda, "is_available", return_value=True),
            patch.object(devices.torch.cuda, "device_count", return_value=2),
            patch.object(
                devices.torch.cuda,
                "mem_get_info",
                side_effect=[(8 * devices.MIB, 16 * devices.MIB), (7 * devices.MIB, 16 * devices.MIB)],
            ),
            patch.object(devices.torch.cuda, "get_device_properties", return_value=properties),
        ):
            snapshot = devices.accelerator_snapshot("2,3", requested_count=2)

        self.assertEqual(snapshot["backend"], "rocm")
        self.assertIsNone(snapshot["hsa_override_gfx_version"])
        self.assertEqual(snapshot["devices"][0]["requested_index"], 2)
        self.assertEqual(snapshot["devices"][0]["logical_index"], 0)
        self.assertEqual(snapshot["devices"][1]["memory_free_mib"], 7)


if __name__ == "__main__":
    unittest.main()
