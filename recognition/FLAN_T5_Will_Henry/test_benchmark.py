import unittest
from benchmark import parse_telemetry


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


if __name__ == "__main__":
    unittest.main()
