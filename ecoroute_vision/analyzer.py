import numpy as np
from collections import deque
import math


class SustainabilityAnalyzer:
    def __init__(self):
        """Initialize sustainability impact analyzer"""
        self.fuel_savings_history = deque(maxlen=100)
        self.co2_reduction_history = deque(maxlen=100)
        self.route_quality_history = deque(maxlen=100)

        # Constants for calculations
        self.FUEL_CONSUMPTION_BASE = 8.0  # liters/100km
        self.CO2_PER_LITER = 2.31  # kg CO2 per liter of gasoline
        self.AVG_SPEED = 50  # km/h

    def calculate_route_efficiency(self, current_road_quality, distance_km):
        """
        Calculate fuel efficiency based on road quality

        Args:
            current_road_quality: Road quality score (0-1, 1=best)
            distance_km: Route distance in kilometers

        Returns:
            dict: Efficiency metrics
        """
        # Road quality affects fuel consumption (poor roads = more fuel)
        fuel_multiplier = 1.0 + (1.0 - current_road_quality) * 0.3

        base_fuel = (distance_km / 100) * self.FUEL_CONSUMPTION_BASE
        actual_fuel = base_fuel * fuel_multiplier
        optimal_fuel = base_fuel * 0.95  # Assume 5% better with perfect roads

        fuel_saving = optimal_fuel - actual_fuel
        co2_reduction = fuel_saving * self.CO2_PER_LITER

        # Store for analytics
        self.fuel_savings_history.append(fuel_saving)
        self.co2_reduction_history.append(co2_reduction)
        self.route_quality_history.append(current_road_quality)

        return {
            'current_fuel_liters': actual_fuel,
            'optimal_fuel_liters': optimal_fuel,
            'fuel_saving_liters': max(0, fuel_saving),
            'co2_reduction_kg': max(0, co2_reduction),
            'route_efficiency_score': current_road_quality,
            'suggested_reroute': current_road_quality < 0.7
        }

    def suggest_alternative_route(self, current_quality, available_routes):
        """
        Suggest best route based on road quality and efficiency

        Args:
            current_quality: Current route quality score
            available_routes: List of alternative routes with their qualities

        Returns:
            dict: Best route suggestion
        """
        if not available_routes:
            return None

        # Score routes based on quality and estimated time
        scored_routes = []
        for route in available_routes:
            quality = route.get('quality', 0.5)
            distance = route.get('distance_km', 10)
            time_minutes = route.get('time_minutes', 15)

            # Efficiency score (higher is better)
            efficiency_score = quality * 0.7 + (1 / time_minutes) * 30 * 0.3

            # Fuel saving estimate
            fuel_saving = self.calculate_route_efficiency(
                quality, distance)['fuel_saving_liters']

            scored_routes.append({
                **route,
                'efficiency_score': efficiency_score,
                'estimated_fuel_saving': fuel_saving,
                'recommendation': 'Highly Recommended' if efficiency_score > 0.8 else
                'Recommended' if efficiency_score > 0.6 else 'Consider'
            })

        # Sort by efficiency score
        scored_routes.sort(key=lambda x: x['efficiency_score'], reverse=True)

        best_route = scored_routes[0] if scored_routes else None

        return {
            'best_route': best_route,
            'all_routes': scored_routes,
            'current_route_quality': current_quality,
            'improvement_potential': best_route['efficiency_score'] - current_quality if best_route else 0
        }

    def get_sustainability_metrics(self):
        """Get cumulative sustainability impact metrics"""
        total_fuel_saved = sum(self.fuel_savings_history)
        total_co2_reduced = sum(self.co2_reduction_history)
        avg_route_quality = np.mean(
            list(self.route_quality_history)) if self.route_quality_history else 0

        return {
            'total_fuel_saved_liters': total_fuel_saved,
            'total_co2_reduced_kg': total_co2_reduced,
            'average_route_quality': avg_route_quality,
            # kg CO2 absorbed by one tree per year
            'trees_equivalent': total_co2_reduced / 21.77,
            'trips_analyzed': len(self.fuel_savings_history)
        }
