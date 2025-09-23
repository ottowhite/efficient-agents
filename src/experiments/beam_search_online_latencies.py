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
plt.figure(figsize=(10, 6))
plt.plot(sorted_latencies, cdf_values, linewidth=2, color='steelblue')
plt.xlabel('Latency (seconds)', fontsize=12, fontweight='bold')
plt.ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
plt.title('Cumulative Distribution Function of Beam Search Online Latencies', fontsize=14, fontweight='bold')

# Set axis limits
plt.xlim(0, None)  # Start x-axis at 0
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1), [f'{int(x*100)}%' for x in np.arange(0, 1.1, 0.1)])

# Add percentile reference lines
# Horizontal lines that stop at the CDF curve intersection
plt.plot([0, p50], [0.5, 0.5], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([0, p90], [0.9, 0.9], color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([0, p95], [0.95, 0.95], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([0, p99], [0.99, 0.99], color='purple', linestyle='--', alpha=0.7, linewidth=1.5)

# Vertical lines that stop at the CDF curve intersection
plt.plot([p50, p50], [0, 0.5], color='red', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([p90, p90], [0, 0.9], color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([p95, p95], [0, 0.95], color='green', linestyle='--', alpha=0.7, linewidth=1.5)
plt.plot([p99, p99], [0, 0.99], color='purple', linestyle='--', alpha=0.7, linewidth=1.5)

# Add text labels for the percentile values with cumulative latency percentages
plt.text(p50, 0.02, f'{p50:.1f}s ({cum_latency_50:.1f}%)', rotation=90, verticalalignment='bottom', 
         horizontalalignment='right', color='red', fontweight='bold')
plt.text(p90, 0.02, f'{p90:.1f}s ({cum_latency_90:.1f}%)', rotation=90, verticalalignment='bottom', 
         horizontalalignment='right', color='orange', fontweight='bold')
plt.text(p95, 0.02, f'{p95:.1f}s ({cum_latency_95:.1f}%)', rotation=90, verticalalignment='bottom', 
         horizontalalignment='right', color='green', fontweight='bold')
plt.text(p99, 0.02, f'{p99:.1f}s ({cum_latency_99:.1f}%)', rotation=90, verticalalignment='bottom', 
         horizontalalignment='right', color='purple', fontweight='bold')

# Add grid and styling
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()

# Show plot
plt.savefig('beam_search_online_latencies.png')