"""
Unit tests for TerraPave SustainabilityAnalyzer
"""

import unittest
from terrapave import SustainabilityAnalyzer


class TestSustainabilityAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SustainabilityAnalyzer(base_fuel_rate=8.0)

    def test_pristine_road_efficiency(self):
        metrics = self.analyzer.calculate_route_efficiency(current_road_quality=1.0, distance_km=10.0)
        self.assertEqual(metrics["route_efficiency_score"], 1.0)
        self.assertEqual(metrics["current_fuel_liters"], 0.8)
        self.assertEqual(metrics["optimal_fuel_liters"], 0.76)
        self.assertAlmostEqual(metrics["excess_fuel_wasted_liters"], 0.04, places=3)
        self.assertFalse(metrics["suggested_reroute"])
        self.assertEqual(metrics["pavement_condition"], "Excellent")

    def test_degraded_road_efficiency(self):
        metrics = self.analyzer.calculate_route_efficiency(current_road_quality=0.2, distance_km=10.0)
        self.assertEqual(metrics["route_efficiency_score"], 0.2)
        # fuel multiplier: 1.0 + (1 - 0.2) * 0.35 = 1.28
        # actual fuel: 0.8 * 1.28 = 1.024
        self.assertAlmostEqual(metrics["current_fuel_liters"], 1.024, places=3)
        self.assertGreater(metrics["excess_fuel_wasted_liters"], 0.2)
        self.assertGreater(metrics["excess_co2_kg"], 0.5)
        self.assertTrue(metrics["suggested_reroute"])
        self.assertEqual(metrics["pavement_condition"], "Critical")

    def test_suggest_alternative_route(self):
        candidates = [
            {"name": "Pristine Highway", "quality": 0.95, "distance_km": 10.0, "time_minutes": 12.0},
            {"name": "Broken Street", "quality": 0.30, "distance_km": 9.0, "time_minutes": 16.0},
        ]
        suggestion = self.analyzer.suggest_alternative_route(current_quality=0.30, available_routes=candidates)
        self.assertIsNotNone(suggestion)
        self.assertEqual(suggestion["best_route"]["name"], "Pristine Highway")
        self.assertGreater(suggestion["best_route"]["estimated_fuel_saving_liters"], 0.0)
        self.assertGreater(suggestion["best_route"]["co2_saved_kg"], 0.0)

    def test_cumulative_metrics(self):
        self.analyzer.calculate_route_efficiency(1.0, 10.0)
        self.analyzer.calculate_route_efficiency(0.5, 10.0)
        stats = self.analyzer.get_sustainability_metrics()
        self.assertEqual(stats["samples_analyzed"], 2)
        self.assertGreater(stats["total_fuel_wasted_liters"], 0.0)
        self.assertAlmostEqual(stats["average_route_quality"], 0.75, places=2)


if __name__ == "__main__":
    unittest.main()
