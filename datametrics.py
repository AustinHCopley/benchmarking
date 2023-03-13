import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rand

model_names = ["yolov7", "yolov7e6e", "mobilenetv1", "mobilenetv3"]

def metrics(inference_times):
    for i, m in enumerate(inference_times):
        mean = np.mean(m)
        variance = np.var(m)
        min_latency = np.min(m)
        max_latency = np.max(m)
        data = {
            "metric" : ["mean", "variance", "min", "max"],
            "val" : [np.mean(m), np.var(m), np.min(m), np.max(m)]
        }
        df = pd.DataFrame(data)
        print(df)
        # TODO save dataframe to file
        
        frames = list(range(1, len(m)+1))
        plt.plot(frames, m, label=model_names[i])
    plt.xlabel("Frame")
    plt.ylabel("Latency (ms)")
    plt.title("Inference times")
    plt.legend()
    plt.show()


def main():
    rand.seed(0)
    inference_times = []
    for x in range(4):
        inf = [round(rand.uniform(1, 4), 1) for x in range(10)]
        inference_times.append(inf)
    metrics(inference_times)

if __name__ == "__main__":
    main()
