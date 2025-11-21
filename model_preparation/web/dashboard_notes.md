# Dashboard User Guide

## Understanding Model Metrics

This guide explains how the Jharkhand-IDS dashboard works and what each metric and visualization means. The dashboard allows you to interact with the intrusion detection model, view performance metrics, and understand how the model makes decisions.

### Accuracy
**What it means:** Accuracy tells you the overall percentage of correct predictions. If accuracy is 95%, it means the model correctly classified 95 out of 100 network flows.

**Why it matters:** High accuracy means the model is generally good at distinguishing between normal and attack traffic. However, accuracy alone doesn't tell the full story - you also need to consider precision and recall.

### Precision
**What it means:** Precision answers the question: "When the model says 'attack', how often is it right?" If precision is 90%, it means 9 out of 10 times the model flags something as an attack, it actually is an attack.

**Why it matters:** High precision means fewer false alarms. This is important because you don't want to waste time investigating false positives. However, high precision might come at the cost of missing some real attacks.

### Recall
**What it means:** Recall answers the question: "Of all the real attacks, how many did the model catch?" If recall is 85%, it means the model detected 85 out of 100 actual attacks.

**Why it matters:** High recall means the model catches most real threats. This is crucial for security - you want to minimize the number of attacks that slip through undetected. However, high recall might mean more false alarms.

### F1 Score
**What it means:** F1 score is a balance between precision and recall. It's the harmonic mean of both metrics, giving you a single number that considers both.

**Why it matters:** F1 score is useful when you need to balance both precision and recall. It's especially helpful when comparing different models or configurations.

### ROC-AUC
**What it means:** ROC-AUC (Area Under the ROC Curve) measures how well the model can distinguish between normal and attack traffic. A score of 1.0 means perfect separation, while 0.5 means the model is no better than random guessing.

**Why it matters:** ROC-AUC gives you an overall measure of model performance that's independent of the classification threshold. Higher AUC means better model performance.

## Understanding Dashboard Options

### Metric Selection
**What it does:** Selecting a metric changes which performance measure is displayed in the charts.

**Example:** If you select "Recall", the charts will focus on showing how well the model detects attacks, rather than overall accuracy.

### Chart Type Selection
**What it does:** This changes how the data is visualized.

- **Bar Chart:** Best for comparing different metrics side by side
- **Pie Chart:** Shows proportions (e.g., 70% normal, 30% attacks)
- **Line Chart:** Shows trends over time
- **Heatmap:** Uses color intensity to show patterns (great for confusion matrices)
- **ROC Curve:** Shows the trade-off between true positives and false positives

### Time Range Filter
**What it does:** Filters data to show only results from a specific time period.

**Example:** Selecting "Last 7 days" will show only attacks and metrics from the past week, helping you see recent trends.

### Attack Type Filter
**What it does:** Shows only data related to specific attack types.

**Example:** Filtering by "DDoS" will show only DDoS-related predictions and metrics, helping you analyze how well the model handles this specific threat.

## Understanding Visualizations

### Attack Distribution
**What it shows:** A breakdown of different attack types detected in your network.

**How to read it:** Each segment or bar represents a different attack type. Larger segments mean more of that attack type was detected. This helps you understand which threats are most common in your environment. The CICIDS2017 dataset includes attacks like DDoS, PortScan, Botnet, Web Attacks (XSS, SQL Injection), and various DoS variants. Understanding attack distribution helps prioritize security measures.

### Feature Importance
**What it shows:** Which features (network flow characteristics) are most important for the model's predictions.

**How to read it:** Features with higher importance scores have more influence on the model's decisions. For example, if "Flow Duration" has high importance, the model relies heavily on how long network connections last to make predictions. Features like packet counts, flow duration, and protocol flags are often strong indicators of attacks. Understanding feature importance helps you know which network characteristics to monitor most closely.

**Note:** Feature importance is only available for tree-based models like RandomForest. Other models may not provide this information.

**How model decisions depend on features:** The model analyzes all features together to make predictions. A single unusual value may not trigger an attack classification, but combinations of suspicious characteristics increase the likelihood. For instance, high packet rates combined with unusual flow duration and suspicious TCP flags create a pattern that strongly indicates an attack. Adjusting feature values in the dashboard shows how different network flow patterns affect the prediction.

### Confusion Matrix
**What it shows:** A detailed breakdown of predictions showing:
- **True Positives:** Correctly identified attacks
- **True Negatives:** Correctly identified normal traffic
- **False Positives:** Normal traffic incorrectly flagged as attacks (false alarms)
- **False Negatives:** Attacks missed by the model

**How to read it:**
- High values on the diagonal (top-left and bottom-right) = good performance
- High values off the diagonal = model mistakes
- The goal is to maximize diagonal values and minimize off-diagonal values

### ROC Curve
**What it shows:** A graph plotting True Positive Rate (how many attacks are caught) against False Positive Rate (how many false alarms occur).

**How to read it:**
- A curve closer to the top-left corner = better performance
- The diagonal line represents random guessing
- The area under the curve (AUC) is a single number summarizing performance (closer to 1.0 is better)

**Why it's useful:** ROC curves help you understand the trade-off between catching attacks and avoiding false alarms. You can adjust the model's threshold based on whether you prioritize catching all attacks or reducing false alarms.

## Tips for Using the Dashboard

1. **Start with Accuracy:** Begin by checking overall accuracy to get a sense of model performance.

2. **Check Precision and Recall Together:** Don't just look at one - high precision with low recall means you're missing attacks, while high recall with low precision means too many false alarms.

3. **Use Filters Strategically:** Filter by time range to see recent trends, or by attack type to analyze specific threats.

4. **Compare Chart Types:** Try different chart types to see which visualization helps you understand the data best.

5. **Monitor Over Time:** Regularly check the dashboard to spot trends - increasing false positives might indicate a need to retrain the model.

6. **Use Feature Importance:** Understanding which features matter most can help you focus monitoring efforts on the most important network characteristics.

## Common Questions

**Q: Why is my accuracy high but I'm still getting false alarms?**
A: High accuracy doesn't guarantee low false positives. Check your precision metric - low precision means more false alarms even with good overall accuracy.

**Q: Should I prioritize precision or recall?**
A: It depends on your priorities. If missing attacks is costly, prioritize recall. If false alarms are disruptive, prioritize precision. F1 score balances both.

**Q: What does a low ROC-AUC mean?**
A: ROC-AUC below 0.7 suggests the model struggles to distinguish between normal and attack traffic. Consider retraining with different features or algorithms.

**Q: How often should I check the dashboard?**
A: Daily monitoring is recommended for active security systems. Weekly reviews are sufficient for less critical environments.

