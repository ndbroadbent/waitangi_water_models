"""Physical and mathematical constants."""

import math

# Gravitational acceleration (m/s²)
G = 9.81

# Water density (kg/m³)
RHO_WATER = 1025.0  # Seawater

# Air density at sea level (kg/m³)
RHO_AIR = 1.225

# Earth's angular velocity (rad/s)
OMEGA_EARTH = 7.2921e-5

# Tidal constituent frequencies (rad/hour)
# M2: Principal lunar semidiurnal (period ~12.42 hours)
OMEGA_M2 = 2 * math.pi / 12.4206

# S2: Principal solar semidiurnal (period 12 hours)
OMEGA_S2 = 2 * math.pi / 12.0

# N2: Larger lunar elliptic (period ~12.66 hours)
OMEGA_N2 = 2 * math.pi / 12.6583

# K1: Lunar diurnal (period ~23.93 hours)
OMEGA_K1 = 2 * math.pi / 23.9345

# O1: Lunar diurnal (period ~25.82 hours)
OMEGA_O1 = 2 * math.pi / 25.8193

# Spring/neap modulation period (days)
SPRING_NEAP_PERIOD_DAYS = 14.77

# Unit conversions
MS_TO_KMH = 3.6
KMH_TO_MS = 1 / 3.6
DEG_TO_RAD = math.pi / 180
RAD_TO_DEG = 180 / math.pi
HOURS_TO_SECONDS = 3600
DAYS_TO_SECONDS = 86400

# Coordinate reference system parameters
# NZTM2000 (EPSG:2193) parameters
NZTM_FALSE_EASTING = 1600000.0
NZTM_FALSE_NORTHING = 10000000.0
NZTM_SCALE_FACTOR = 0.9996
NZTM_CENTRAL_MERIDIAN = 173.0
