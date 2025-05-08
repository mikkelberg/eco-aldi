import json

metrics_path = "experiments/oracle-model/first-training/output/metrics.json"
'''
with open(metrics_path, "r") as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        if i < 10:  # print only the first few keys to check
            print("Example keys in metrics.json:")
            print(data.keys())
        if "grad" in str(data.keys()).lower():
            print(f"Found gradient info at line {i}:")
            print(data)
'''

grad_norms = []
with open(metrics_path, "r") as f:
    for line in f:
        data = json.loads(line)
        if "grad_norm" in data:
            grad_norms.append(data["grad_norm"])

plt.plot(grad_norms)
plt.xlabel("Iterations (filtered)")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm During Training")
plt.grid()
plt.savefig("experiments/oracle-model/first-training/results/grad_norms.png")
plt.close()