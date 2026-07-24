["Sample Mean", "Average of sample values", r"$\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$"],
["Sample Variance", "Measure of spread in the sample", r"$s^2 = \frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$"],
["Bias", "Difference between the estimator's expected value and the true parameter value", r"$\text{Bias}(\hat{\theta}) = E(\hat{\theta}) - \theta$"],
["Consistency", "An estimator's convergence in probability to the parameter as sample size increases", r"$\hat{\theta}_n \xrightarrow{p} \theta$ as $n \to \infty$"],
["Efficiency", "An unbiased estimator with the smallest variance among all unbiased estimators", r"$\text{Var}(\hat{\theta}_{\text{eff}}) \leq \text{Var}(\hat{\theta})$ for any unbiased $\hat{\theta}$"],
["Sufficiency", "An estimator that captures all information about the parameter from the sample", r"A statistic $T(X)$ is sufficient for $\theta$ if the conditional distribution of $X$ given $T(X)$ does not depend on $\theta$"],
["Point Estimator", "A single value that serves as a best guess or best estimate of a population parameter", r"A statistic $\hat{\theta}$ used to estimate $\theta$"],
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
# Ensure matplotlib is configured to use LaTeX for text rendering
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# Combined data for various estimation methods, with adjusted Delta Method row
data_combined = [
    ["Regularity: Identifiability", "Different parameter values lead to different probability distributions", r"If $\theta_1 \neq \theta_2$, then $f(x|\theta_1) \neq f(x|\theta_2)$ for at least one $x$"],
    ["Regularity: Differentiability", "The likelihood function is differentiable as a function of the parameter", r"The score function $U(\theta) = \frac{\partial}{\partial \theta} \log L(\theta)$ exists"],
    ["Regularity: Support Does Not Depend on Parameters", "The set of possible observations does not depend on the parameter", r"The support of $f(x|\theta)$ is the same for all $\theta$"],
    ["Regularity: Existence of Moments", "Sufficient moments of the estimator exist", r"$E|\hat{\theta}^k| < \infty$ for some $k \geq 1$"],
    ["Regularity: Independence and Identically Distributed (i.i.d.)", "Samples are drawn independently from the same distribution", r"Observations $X_1, X_2, \ldots, X_n$ are i.i.d."],
    ["Regularity: Parameter Space is Open", "The parameter space is an open subset of the Euclidean space", r"$\theta \subseteq R^k$"],

]

fig, ax = plt.subplots(figsize=(16, 10))  # Adjusted for content
ax.axis('off')
table = ax.table(cellText=data_combined[1:], colLabels=data_combined[0], cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)  # Adjust scaling for visibility
plt.title("Estimation Methods and Their Formulas", fontsize=16, pad=20)
plt.tight_layout()
plt.show()
