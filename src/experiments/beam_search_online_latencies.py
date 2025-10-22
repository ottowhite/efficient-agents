import matplotlib.pyplot as plt
import numpy as np

latency_percentiles = [2.21, 2.47, 2.73, 2.89, 3.04, 3.12, 3.25, 3.31, 3.41, 3.54, 3.71, 3.78, 3.86, 3.99, 4.06, 4.11, 4.22, 4.29, 4.41, 4.47, 4.61, 4.78, 5.01, 5.13, 5.32, 5.45, 5.58, 5.66, 5.93, 6.01, 6.10, 6.23, 6.37, 6.43, 6.52, 6.67, 6.91, 7.04, 7.14, 7.29, 7.45, 7.67, 7.77, 8.01, 8.29, 8.65, 8.77, 9.03, 9.15, 9.36, 9.62, 9.78, 10.00, 10.47, 10.63, 10.94, 11.43, 11.73, 12.10, 12.31, 12.82, 13.25, 13.50, 13.85, 14.51, 15.00, 15.66, 16.08, 16.34, 16.63, 17.87, 19.09, 19.78, 21.48, 22.42, 23.23, 23.84, 24.87, 26.45, 28.34, 29.98, 31.37, 33.81, 34.77, 35.94, 38.60, 39.23, 41.26, 42.01, 43.18, 44.02, 44.95, 46.86, 48.72, 50.16, 52.53, 57.26, 65.72, 75.96]

# Calculate CDF values
sorted_latencies = np.sort(latency_percentiles)
cdf_values = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)

# Calculate percentile values
p50 = latency_percentiles[49]
p90 = latency_percentiles[89]
p95 = latency_percentiles[94]
p99 = latency_percentiles[98]

# Calculate cumulative latency percentages
total_latency = sum(latency_percentiles)
cum_latency_50 = sum(latency_percentiles[:50]) / total_latency * 100
cum_latency_90 = sum(latency_percentiles[:90]) / total_latency * 100
cum_latency_95 = sum(latency_percentiles[:95]) / total_latency * 100
cum_latency_99 = sum(latency_percentiles[:99]) / total_latency * 100

# Create CDF plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sorted_latencies, cdf_values, linewidth=2, color='steelblue')
ax.set_xlabel('Latency (seconds)', fontsize=22)
ax.set_ylabel('CDF', fontsize=22)

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=18)

# Set axis limits
ax.set_xlim(0, None)  # Start x-axis at 0
ax.set_ylim(0, 1)

# Add vertical percentile reference lines through entire plot
ax.axvline(p50, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label=f'P50: {p50:.1f}s')
ax.axvline(p90, color='orange', linestyle='--', alpha=0.7, linewidth=1.5, label=f'P90: {p90:.1f}s')
ax.axvline(p95, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label=f'P95: {p95:.1f}s')
ax.axvline(p99, color='purple', linestyle='--', alpha=0.7, linewidth=1.5, label=f'P99: {p99:.1f}s')

# Add legend
ax.legend(fontsize=14, loc='lower right')

# Add grid and styling
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Save plot
plt.savefig('beam_search_online_latencies.png', dpi=300, bbox_inches='tight')