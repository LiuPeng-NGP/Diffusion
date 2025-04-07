Okay, let's describe the loss function used to train the vector field model v_θ(x, t) for each of the Conditional Flow Matching methods implemented in the code.

General Loss Formulation

All these methods aim to minimize the expected squared L2 distance between the output of the neural network v_θ(x_t, t) and the target conditional vector field u_t(x_t | x_0, x_1) specific to that method. The expectation is taken over:

Time t, typically sampled uniformly from [0, 1].

The pair of samples (x_0, x_1), drawn according to the specific strategy (independent or from an OT plan).

The sample x_t drawn from the conditional probability path p_t(x | x_0, x_1) defined by the method.

The general form of the loss function L(θ) is:

L(θ) = E_{t ~ U[0,1], (x_0, x_1) ~ π, x_t ~ p_t(x | x_0, x_1)} [ || v_θ(x_t, t) - u_t(x_t | x_0, x_1) ||^2 ]

Where:

v_θ(x_t, t) is the output of the neural network (the learned vector field).

π denotes the joint distribution or sampling procedure for the pair (x_0, x_1).

p_t(x | x_0, x_1) is the conditional probability path N(x; μ_t(x_0, x_1, t), σ_t(t)^2 * I).

u_t(x_t | x_0, x_1) is the target conditional vector field.

In practice, this expectation is approximated using Monte Carlo estimation over minibatches. The sample_location_and_conditional_flow (or guided_sample_location_and_conditional_flow) method in each class is designed to compute the t, x_t, and u_t needed to evaluate one term inside the expectation for the loss calculation.

Loss Function for Each Method

Let's specify π, μ_t, σ_t, and u_t for each method to define its specific loss:

1. ConditionalFlowMatcher (Independent CFM)

Sampling (x_0, x_1): Independent samples: x_0 ~ p_0, x_1 ~ p_1.

Target Vector Field u_t: x_1 - x_0.

Loss:
L(θ) = E_{t ~ U[0,1], x_0 ~ p_0, x_1 ~ p_1, x_t ~ N(x; (1-t)x_0 + tx_1, σ^2 I)} [ || v_θ(x_t, t) - (x_1 - x_0) ||^2 ]

2. ExactOptimalTransportConditionalFlowMatcher (OT-CFM)

Sampling (x_0, x_1): Jointly sampled from the exact Optimal Transport plan π_OT between minibatches of p_0 and p_1.

Target Vector Field u_t: x_1 - x_0.

Loss:
L(θ) = E_{t ~ U[0,1], (x_0, x_1) ~ π_OT, x_t ~ N(x; (1-t)x_0 + tx_1, σ^2 I)} [ || v_θ(x_t, t) - (x_1 - x_0) ||^2 ]
(Note: The only difference from Independent CFM is how the (x_0, x_1) pairs are obtained)

3. TargetConditionalFlowMatcher

Sampling (x_0, x_1): Independent samples: x_0 ~ p_0, x_1 ~ p_1.

Target Vector Field u_t: (x_1 - (1 - σ) x_t) / (1 - (1 - σ) t).

Loss:
L(θ) = E_{t ~ U[0,1], x_0 ~ p_0, x_1 ~ p_1, x_t ~ N(x; tx_1, (1-(1-σ)t)^2 I)} [ || v_θ(x_t, t) - (x_1 - (1 - σ) x_t) / (1 - (1 - σ) t) ||^2 ]

4. SchrodingerBridgeConditionalFlowMatcher (SB-CFM)

Sampling (x_0, x_1): Jointly sampled from the entropic Optimal Transport plan π_γ (with γ=2σ^2) between minibatches of p_0 and p_1.

Target Vector Field u_t: (x_1 - x_0) + ( (1 - 2t) / (2t(1 - t)) ) * (x_t - μ_t) where μ_t = (1-t)x_0 + tx_1.

Loss:
L(θ) = E_{t ~ U[0,1], (x_0, x_1) ~ π_γ, x_t ~ N(x; μ_t, σ^2 t(1-t) I)} [ || v_θ(x_t, t) - [(x_1 - x_0) + ( (1 - 2t) / (2t(1 - t)) ) * (x_t - μ_t)] ||^2 ]

5. VariancePreservingConditionalFlowMatcher (VP-CFM)

Sampling (x_0, x_1): Independent samples: x_0 ~ p_0, x_1 ~ p_1.

Target Vector Field u_t: (π / 2) * (cos(πt / 2) * x_1 - sin(πt / 2) * x_0).

Loss:
L(θ) = E_{t ~ U[0,1], x_0 ~ p_0, x_1 ~ p_1, x_t ~ N(x; cos(πt/2)x_0 + sin(πt/2)x_1, σ^2 I)} [ || v_θ(x_t, t) - (π / 2) * (cos(πt / 2) * x_1 - sin(πt / 2) * x_0) ||^2 ]

In essence, the training procedure involves repeatedly:

Sampling x_0, x_1, and t.

Possibly resampling (x_0, x_1) based on an OT plan if using OT-CFM or SB-CFM.

Computing x_t by adding the appropriate noise to the specific μ_t.

Computing the target vector field u_t using the formula for the chosen method.

Evaluating the model v_θ(x_t, t).

Calculating the squared difference and averaging over the minibatch to estimate the loss L(θ).

Performing a gradient descent step on θ based on this loss.