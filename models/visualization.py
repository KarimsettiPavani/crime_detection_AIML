import matplotlib.pyplot as plt
import seaborn as sns

def generate_zone_chart(zone_risk):

    plt.figure()
    plt.bar(zone_risk["Zone"], zone_risk["Predicted"])
    plt.xticks(rotation=45)
    plt.title("Predicted Crime Cases by Zone (2026)")
    plt.tight_layout()
    plt.savefig("static/zone_chart.png")
    plt.close()

def generate_heatmap(zone_risk):

    plt.figure()
    sns.heatmap(zone_risk.set_index("Zone"),
                annot=True)
    plt.title("Crime Risk Heatmap")
    plt.tight_layout()
    plt.savefig("static/heatmap.png")
    plt.close()