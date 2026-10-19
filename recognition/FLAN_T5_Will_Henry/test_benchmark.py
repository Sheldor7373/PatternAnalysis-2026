import unittest
from benchmark import long_case_indices, parse_telemetry


class TelemetryTests(unittest.TestCase):
    def test_missing_power_retains_memory_sample(self):
        sample = parse_telemetry("3072, 4096, 55, N/A\n")
        self.assertEqual(sample["used_mib"], 3072)
        self.assertEqual(sample["total_mib"], 4096)
        self.assertEqual(sample["temperature_c"], 55)
        self.assertIsNone(sample["power_w"])

    def test_missing_optional_fields_are_null(self):
        sample = parse_telemetry("1024, 4096, [Not Supported], [N/A]")
        self.assertIsNone(sample["temperature_c"])
        self.assertIsNone(sample["power_w"])

    def test_missing_memory_is_not_a_valid_sample(self):
        with self.assertRaises(ValueError):
            parse_telemetry("N/A, 4096, 55, 20")

    def test_long_sources_targets_and_joint_lengths_are_selected(self):
        self.assertEqual(long_case_indices([1000, 10, 450, 5], [10, 1000, 500, 5]), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
