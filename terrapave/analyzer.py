"""
TerraPave Sustainability & Eco-Routing Analyzer
Calculates fuel consumption penalties, excess emissions, carbon reduction, and optimal route trade-offs.
"""

from collections import deque
import numpy as np


class SustainabilityAnalyzer:
    # Industry constants for standard passenger/light commercial vehicles
    DEFAULT_FUEL_CONSUMPTION_BASE = 8.0   # Liters per 100 km (at optimal speed on smooth asphalt)
    CO2_PER_LITER_GASOLINE = 2.31         # kg CO2 emitted per liter of gasoline combusted
    ANNUAL_TREE_CO2_ABSORPTION_KG = 21.77 # Average kg CO2 sequestered per mature urban tree/year

    def __init__(self, base_fuel_rate=DEFAULT_FUEL_CONSUMPTION_BASE):
        """
        Initialize the sustainability & fuel efficiency analyzer.

        Args:
            base_fuel_rate (float): Base vehicle fuel consumption in L/100km on perfect pavement.
        """
        self.base_fuel_rate = float(base_fuel_rate)
        self.fuel_wasted_history = deque(maxlen=200)
        self.fuel_saved_history = deque(maxlen=200)
        self.co2_reduced_history = deque(maxlen=200)
        self.route_quality_history = deque(maxlen=200)

    def calculate_route_efficiency(self, current_road_quality, distance_km=10.0):
        """
        Calculate fuel efficiency penalty and carbon emissions based on road surface quality.

        Args:
            current_road_quality (float): Road quality score (0.0 to 1.0, where 1.0 is pristine).
            distance_km (float): Segment distance in kilometers.

        Returns:
            dict: Detailed fuel consumption, excess wastage, and carbon penalty metrics.
        """
        quality = float(np.clip(current_road_quality, 0.0, 1.0))
        dist = max(0.01, float(distance_km))

        # Degraded roads increase rolling resistance, frequent braking, and acceleration
        # Multiplier ranges from 1.0 (pristine) to 1.35 (severe potholes/cracks)
        fuel_multiplier = 1.0 + (1.0 - quality) * 0.35

        base_fuel = (dist / 100.0) * self.base_fuel_rate
        actual_fuel = base_fuel * fuel_multiplier
        optimal_fuel = base_fuel * 0.95  # Ideal aerodynamic and smooth cruising baseline

        # Excess fuel wasted on this segment compared to pristine road
        excess_fuel_wasted = max(0.0, actual_fuel - optimal_fuel)
        excess_co2_kg = excess_fuel_wasted * self.CO2_PER_LITER_GASOLINE

        # Record metrics for historical analytics
        self.fuel_wasted_history.append(excess_fuel_wasted)
        self.route_quality_history.append(quality)

        return {
            "current_fuel_liters": round(actual_fuel, 4),
            "optimal_fuel_liters": round(optimal_fuel, 4),
            "excess_fuel_wasted_liters": round(excess_fuel_wasted, 4),
            "excess_co2_kg": round(excess_co2_kg, 4),
            "route_efficiency_score": round(quality, 3),
            "suggested_reroute": quality < 0.70,
            "pavement_condition": (
                "Excellent" if quality > 0.85 else
                "Good" if quality > 0.70 else
                "Fair" if quality > 0.50 else
                "Poor" if quality > 0.30 else "Critical"
            ),
        }

    def suggest_alternative_route(self, current_quality, available_routes):
        """
        Evaluate alternative route candidates and recommend the eco-optimal choice.

        Args:
            current_quality (float): Current route road quality score (0-1).
            available_routes (list): List of candidate route dictionaries.

        Returns:
            dict: Evaluated candidates with rankings and potential fuel/CO2 savings.
        """
        if not available_routes:
            return None

        scored_routes = []
        current_efficiency = self.calculate_route_efficiency(current_quality, 10.0)
        current_fuel = current_efficiency["current_fuel_liters"]

        for route in available_routes:
            quality = float(route.get("quality", 0.5))
            dist = float(route.get("distance_km", 10.0))
            time_minutes = max(0.1, float(route.get("time_minutes", 15.0)))

            route_metrics = self.calculate_route_efficiency(quality, dist)
            candidate_fuel = route_metrics["current_fuel_liters"]

            # Potential savings if switching from degraded current road to this route
            fuel_saving = max(0.0, current_fuel - candidate_fuel)
            co2_saving = fuel_saving * self.CO2_PER_LITER_GASOLINE

            # Composite score: 55% road quality, 30% time efficiency, 15% distance penalty
            time_score = min(1.0, 15.0 / time_minutes)
            dist_score = min(1.0, 10.0 / dist)
            efficiency_score = (quality * 0.55) + (time_score * 0.30) + (dist_score * 0.15)

            recommendation = (
                "Highly Recommended" if efficiency_score >= 0.80 else
                "Recommended" if efficiency_score >= 0.65 else
                "Neutral" if efficiency_score >= 0.50 else "Avoid"
            )

            scored_routes.append({
                **route,
                "efficiency_score": round(efficiency_score, 3),
                "estimated_fuel_liters": round(candidate_fuel, 3),
                "estimated_fuel_saving_liters": round(fuel_saving, 3),
                "co2_saved_kg": round(co2_saving, 3),
                "recommendation": recommendation,
            })

        # Sort descending by composite efficiency
        scored_routes.sort(key=lambda r: r["efficiency_score"], reverse=True)
        best_route = scored_routes[0]

        # Record savings if best route improves over current
        if best_route["estimated_fuel_saving_liters"] > 0:
            self.fuel_saved_history.append(best_route["estimated_fuel_saving_liters"])
            self.co2_reduced_history.append(best_route["co2_saved_kg"])

        return {
            "best_route": best_route,
            "all_routes": scored_routes,
            "current_route_quality": round(current_quality, 3),
            "improvement_potential": round(best_route["efficiency_score"] - current_quality, 3),
        }

    def get_sustainability_metrics(self):
        """Get aggregate fleet and trip sustainability metrics."""
        total_wasted = float(sum(self.fuel_wasted_history)) if self.fuel_wasted_history else 0.0
        total_saved = float(sum(self.fuel_saved_history)) if self.fuel_saved_history else 0.0
        total_co2_reduced = float(sum(self.co2_reduced_history)) if self.co2_reduced_history else (total_saved * self.CO2_PER_LITER_GASOLINE)
        avg_quality = float(np.mean(list(self.route_quality_history))) if self.route_quality_history else 1.0

        trees_equiv = total_co2_reduced / self.ANNUAL_TREE_CO2_ABSORPTION_KG

        return {
            "total_fuel_wasted_liters": round(total_wasted, 3),
            "total_fuel_saved_liters": round(total_saved, 3),
            "total_co2_reduced_kg": round(total_co2_reduced, 3),
            "average_route_quality": round(avg_quality, 3),
            "trees_equivalent": round(trees_equiv, 2),
            "samples_analyzed": len(self.route_quality_history),
        }
