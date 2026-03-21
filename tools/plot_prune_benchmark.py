"""Plot pruning methods benchmark comparison - 2 separate charts."""
import matplotlib.pyplot as plt
import numpy as np

methods  = ['Baseline\n(unpruned)', 'L1 Norm', 'FPGM', 'Taylor', 'BN Gamma\n+ CWD', 'BN Gamma', 'Random']
map50    = [0.890, 0.878,  0.873, 0.872, 0.869,  0.865, 0.865]
map50_95 = [0.725, 0.701,  0.698, 0.693, 0.693,  0.689, 0.689]
colors   = ['#607D8B', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#E91E63', '#F44336']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- mAP50 ---
bars1 = ax1.bar(methods, map50, color=colors, edgecolor='white', linewidth=0.5)
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0003,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax1.set_title('mAP50', fontsize=14, fontweight='bold')
ax1.set_ylim(0.84, 0.900)
ax1.set_ylabel('Score', fontsize=12)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.tick_params(axis='x', labelsize=9)

# --- mAP50-95 ---
bars2 = ax2.bar(methods, map50_95, color=colors, edgecolor='white', linewidth=0.5)
for bar in bars2:
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0003,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
ax2.set_title('mAP50-95', fontsize=14, fontweight='bold')
ax2.set_ylim(0.65, 0.735)
ax2.set_ylabel('Score', fontsize=12)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.tick_params(axis='x', labelsize=9)

fig.suptitle('Pruning Methods Benchmark ', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('prune_bm/prune_benchmark.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved to prune_bm/prune_benchmark.png")
